from typing import Tuple

import numpy as np


class Tracker(object):
    """Base class for multi-object video tracker."""

    def __init__(self, img_hw: Tuple[int, int], iou_threshold: float = 0.45):
        """
        Sets key parameters for the tracker.
        """
        self.img_hw = img_hw
        self.iou_threshold = iou_threshold

        self.trackers, self.prev_trackers, self.bbox_idx2tracker = [], [], {}
        self.track_count, self.bbox_count = 0, 0
        self.done = False

    def update(self,
               dets: np.ndarray = np.empty((0, 4)),
               is_gt: np.ndarray = np.empty((0, ))):
        raise NotImplementedError

    def _del_tracker(self, idx: int, done: bool = True):
        """Delete self.trackers[idx], move it to self.del_trackers."""
        tracker = self.trackers.pop(idx)
        tracker.finish(done=done)
        self.prev_trackers.append(tracker)
        for idx in tracker.bbox_idx:
            self.bbox_idx2tracker[idx] = tracker

    def finish(self):
        """Delete all remaining trackers."""
        for idx in reversed(range(len(self.trackers))):
            # don't filter out unfinished tracklets!
            self._del_tracker(idx, done=False)
        self.done = True

    def new(self):
        """Create a new tracker."""
        raise NotImplementedError

    def get_bbox_tracker(self, bbox_idx: int):
        """Get the bbox_tracker for the given bbox_idx."""
        assert self.done, 'Please call Tracker.finish() first.'
        return self.bbox_idx2tracker[bbox_idx]
