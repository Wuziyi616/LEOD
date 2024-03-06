import copy
from typing import Tuple, List

import numpy as np

from .tracker import Tracker
from .utils import clamp_bbox, greedy_matching, iou_batch_xywh, xywh2xyxy


class LinearBoxTracker(object):
    """
    Simple linear velocity bbox tracker.
    Inspired by paper: Towards Unsupervised Object Detection from LiDAR Point Clouds.
    """

    def __init__(self,
                 track_id: int,
                 bbox: np.ndarray,
                 bbox_idx: int,
                 is_gt: bool,
                 img_hw: Tuple[int, int],
                 q: float = 0.9):
        """
        Initialises a tracker using initial bounding box.
        """
        self.img_hw = img_hw  # need to clamp bbox to image boundaries

        # we only track bbox center (x,y) and assume (w,h) unchanged
        # bbox: [x,y,w,h,cls_id], a single detected object
        self.last_bbox, self.bbox, self.class_id = None, bbox[:4], bbox[4]
        self.vxvy = np.zeros(2)
        self.clamp_t, self.clamp_d, self.clamp_l, self.clamp_r = [False] * 4
        self.bbox_idx = [bbox_idx]  # record all matched bbox of this tracklet
        self.missed_bbox = {}  # pred_bbox at missed detection frames
        self.missed_bbox_cache = {}  # cache, clear after each update
        self.is_gt = is_gt  # whether this bbox is from GT labels

        # tracklet confidence
        # init as `q`, and decay by `q` every frame
        self.q = q  # usually 0.9
        self.conf, self.all_conf = q, [q]

        self.id = track_id
        self.age = 0  # number of frames since the track is detected
        self.hits = 1  # number of detections matched to this tracklet
        self.all_hits = [1]
        self.time_since_update = 0  # number of frames since last detection
        self.done = False  # whether this tracklet is finished

    def _conf_update_weight(self) -> float:
        # \Sum_i^{self.age} q^i
        # first predict, then update, so `self.age` is already +1
        return self.q * (1. - self.q**self.age) / (1. - self.q)

    def get_state(self) -> np.ndarray:
        """
        Returns the current bounding box estimate (with class_id).
        """
        # this is getting the corrected state bbox
        # instead of the last observed bbox
        bbox = np.zeros(5, dtype=self.bbox.dtype)
        bbox[:4], self.clamp_t, self.clamp_d, self.clamp_l, self.clamp_r = \
            clamp_bbox(self.bbox, self.img_hw, format_='xywh')
        bbox[4] = self.class_id
        return bbox

    def predict(self) -> np.ndarray:
        """
        Advances the state vector and returns the predicted bbox estimate.
        """
        # update time stats
        self.age += 1
        self.time_since_update += 1
        # linear velocity model
        self.last_bbox = copy.deepcopy(self.bbox)  # bbox_{t-1}
        self.bbox[:2] += self.vxvy
        self.pred_bbox = self.get_state()
        return copy.deepcopy(self.pred_bbox)  # bbox format: [x,y,w,h,cls_id]

    def update(self, new_bbox: np.ndarray, bbox_idx: int, is_gt: bool = False):
        """
        Updates the state vector with observed bbox in the form [x,y,w,h].
        """
        assert new_bbox[4] == self.class_id, 'Tracklet class_id mismatch'
        # update time stats
        self.hits = self.age + 1
        self.all_hits.append(self.hits)
        self.time_since_update = 0
        # update velocity
        self.vxvy = self._robust_velocity(new_bbox)
        # update bbox
        self.bbox = new_bbox[:4]
        self.bbox_idx.append(bbox_idx)
        self.is_gt = self.is_gt or is_gt  # if any bbox is GT, then is_gt=True
        # update confidence
        w = self._conf_update_weight()
        self.conf = (w * self.conf + 1.) / (w + 1.)
        self.all_conf.append(self.conf)
        # clear inpaint bbox cache
        self.missed_bbox.update(self.missed_bbox_cache)
        self.missed_bbox_cache = {}

    def _robust_velocity(self, new_bbox) -> np.ndarray:
        """Compute bbox_clamp aware velocity."""
        # The naive way is just `new_bbox[:2] - self.last_bbox[:2]`
        # However, if the bbox is moving out of image, the new_bbox's center
        #   will be clamped slightly inwards, which will cause a wrong velocity
        # In this case, we should use the bbox's edge instead of center
        vxvy = new_bbox[:2] - self.last_bbox[:2]
        if not any([self.clamp_t, self.clamp_d, self.clamp_l, self.clamp_r]):
            return vxvy
        assert not (self.clamp_t and self.clamp_d), 'Clamp top and bottom'
        assert not (self.clamp_l and self.clamp_r), 'Clamp left and right'
        old_x1, old_y1, old_x2, old_y2 = xywh2xyxy(self.last_bbox[:4])
        new_x1, new_y1, new_x2, new_y2 = xywh2xyxy(new_bbox[:4])
        if self.clamp_t:  # clamp top, so use the bottom edge for vy
            vxvy[1] = new_y2 - old_y2
        if self.clamp_d:  # clamp bottom, so use the top edge for vy
            vxvy[1] = new_y1 - old_y1
        if self.clamp_l:  # clamp left, so use the right edge for vx
            vxvy[0] = new_x2 - old_x2
        if self.clamp_r:  # clamp right, so use the left edge for vx
            vxvy[0] = new_x1 - old_x1
        return vxvy

    def miss(self, frame_idx: int, has_gt: bool = False):
        """
        Decay the tracklet's confidence.
        """
        self.conf *= self.q
        # save the predicted bbox at this step
        # we might add it later ("inpaint" the tracklet)
        if not has_gt:
            self.missed_bbox_cache[frame_idx] = copy.deepcopy(self.pred_bbox)

    def finish(self, done: bool = True):
        self.bbox_idx = np.array(self.bbox_idx)
        self.all_conf = np.array(self.all_conf)
        self.all_hits = np.array(self.all_hits)
        self.done = done
        del self.missed_bbox_cache

    def get_conf(self, bbox_idx: int) -> float:
        return self.all_conf[self.bbox_idx == bbox_idx][0]

    def get_hits(self, bbox_idx: int) -> int:
        return self.all_hits[self.bbox_idx == bbox_idx][0]

    @property
    def area(self) -> float:
        return self.bbox[2] * self.bbox[3]


def associate_tracking(
    trackers: np.ndarray,  # (N, 5): [x,y,w,h,cls_id]
    tracker_order: np.ndarray,  # from most confident to least confident
    detections: np.ndarray,  # (N, 5): [x,y,w,h,cls_id]
    iou_threshold: float = 0.3,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """
    For each tracker_bbox, find its closest detection_bbox in a greedy manner.

    Returns 3 lists of matches, unmatched_trackers, and unmatched_detections
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), [], list(range(len(detections)))
    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), list(range(len(trackers))), []

    # both: (N, 5), each row as [x,y,w,h,cls_id]
    iou_matrix = iou_batch_xywh(trackers, detections)
    # iou_matrix: (num_trk, num_det)

    if min(iou_matrix.shape) > 0 and iou_matrix.max() > 0:
        matched_indices = greedy_matching(
            iou_matrix, tracker_order, thresh=iou_threshold)
        if len(matched_indices) == 0:
            matched_indices = np.empty((0, 2))
    else:
        matched_indices = np.empty((0, 2))

    unmatched_trackers, unmatched_detections = [], []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 0]:
            unmatched_trackers.append(t)
    for d in range(len(detections)):
        if d not in matched_indices[:, 1]:
            unmatched_detections.append(d)

    # matched_indices: (N, 2), each if (trk_idx, det_idx)
    # unmatched_trackers: idx of the tracks that are not matched
    # unmatched_detections: idx of the dets that are not matched
    return matched_indices, unmatched_trackers, unmatched_detections


class LinearTracker(Tracker):
    """Simple linear online tracker."""

    def __init__(
            self,
            img_hw: Tuple[int, int],
            min_conf: float = 0.55,  # 0.9**6 = 0.53
            iou_threshold: float = 0.45,
            q: float = 0.9):
        """
        Sets key parameters for the tracker.
        """
        super().__init__(img_hw=img_hw, iou_threshold=iou_threshold)

        self.min_conf = min_conf  # thresh to delete a track
        self.q = q  # usually 0.9

    def update(self,
               frame_idx: int,
               dets: np.ndarray = np.empty((0, 5)),
               is_gt: np.ndarray = np.empty((0, ))):
        """
        Params:
          frame_idx - int
          dets - a numpy array of detections in shape (N, 4/5),
            each in the format of [x,y,w,h(,cls_id)].
          is_gt - a numpy array specifying whether the bbox is a GT label or
            model's predicted pseudo label

        This method must be called for each frame even with empty detections
          (use np.empty((0, 5)) for frames without detections).

        Returns the a similar array, where the last column is the object ID.
        """
        assert not self.done, 'Please create a new tracker'
        if len(dets) == 0 and len(self.trackers) == 0:
            return
        if is_gt is None or len(is_gt) == 0:
            is_gt = np.zeros((len(dets), ), dtype=bool)
        if dets.shape[1] == 4:
            dets = np.concatenate([dets, np.zeros((len(dets), 1))], axis=1)
        # get predicted locations from existing trackers
        to_del, trks, trks_conf = [], [], []
        for t, trk in enumerate(self.trackers):
            if trk.area <= 0.:
                to_del.append(t)
                continue
            trks.append(trk.predict())  # [x,y,w,h,cls_id]
            trks_conf.append(-trk.conf)
        if len(trks) > 0:
            trks = np.stack(trks, axis=0)  # (num_trk, 5)
        for t in reversed(to_del):
            self._del_tracker(t)
        trks_conf_order = np.argsort(trks_conf)
        matched, unmatched_trks, unmatched_dets = associate_tracking(
            trks, trks_conf_order, dets, self.iou_threshold)
        # matched: (N, 2), each if (trk_idx, det_idx)
        # unmatched_trks: idx of the tracks that are not matched
        # unmatched_dets: idx of the dets that are not matched

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[0]].update(
                dets[m[1], :],
                bbox_idx=self.bbox_count + m[1],
                is_gt=is_gt[m[1]])

        # decay unmatched trackers
        for t in unmatched_trks:
            self.trackers[t].miss(frame_idx=frame_idx, has_gt=is_gt.any())

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = LinearBoxTracker(
                track_id=self.track_count,
                bbox=dets[i, :],
                bbox_idx=self.bbox_count + i,
                is_gt=is_gt[i],
                img_hw=self.img_hw,
                q=self.q)
            self.trackers.append(trk)
            self.track_count += 1
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            # remove dead tracklet
            if trk.conf < self.min_conf:
                self._del_tracker(i)
        self.bbox_count += len(dets)

    def new(self):
        """Create a new tracker."""
        return LinearTracker(
            img_hw=self.img_hw,
            min_conf=self.min_conf,
            iou_threshold=self.iou_threshold,
            q=self.q)
