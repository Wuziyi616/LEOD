import os
import os.path as osp
import copy
from typing import Any, Optional, Dict, Tuple, List

import h5py
import numpy as np
from omegaconf import DictConfig
import torch
import torch as th
import torchvision.ops as ops
from pytorch_lightning.utilities.types import STEP_OUTPUT

from data.genx_utils.labels import ObjectLabels, SparselyBatchedObjectLabels
from data.utils.types import DataType
from data.utils.misc import get_ev_dir, get_ev_h5_fn, get_labels_npz_fn
from models.detection.yolox.utils.boxes import postprocess
from utils.bbox import xyxy2xywh
from utils.evaluation.prophesee.io.box_loading import to_prophesee
from .detection import Module
from .tracking import LinearTracker
from .utils.ssod import pred2label, filter_pred_boxes, evaluate_label, get_scores_ious
from .utils.detection import BackboneFeatureSelector, SeqLens, Mode, DATA_KEY

from nerv.utils import AverageMeter

BBOX_DTYPE = np.dtype({
    'names':
    ['t', 'x', 'y', 'w', 'h', 'class_id', 'class_confidence', 'objectness'],
    'formats': ['<i8', '<f4', '<f4', '<f4', '<f4', '<u4', '<f4', '<f4'],
    'offsets': [0, 8, 12, 16, 20, 24, 28, 32],
    'itemsize':
    40,
})


def tta_postprocess(preds: List[ObjectLabels],
                    conf_thre: float = 0.7,
                    nms_thre: float = 0.45,
                    class_agnostic: bool = False) -> List[ObjectLabels]:
    """Apply NMS on predicted bboxes."""
    if len(preds) == 0:
        return preds

    pad = preds[0].new_zeros()  # empty bbox
    output = [pad] * len(preds)
    for i, pred in enumerate(preds):
        # no need to postprocess GT labels
        if pred.is_gt_label().any():
            output[i] = pred
            continue

        # first convert to [(xyxy), obj_conf, cls_conf, cls_idx]
        t = pred.t.unsqueeze(1)
        pred = pred.get_labels_as_tensors(format_='prophesee')

        # If none are remaining => process next image
        if not pred.size(0):
            continue

        obj_conf, class_conf = pred[:, 4], pred[:, 5]
        conf_mask = ((obj_conf * class_conf) >= conf_thre)  # (N,)
        detections = pred
        detections = detections[conf_mask]
        t = t[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )
        detections = detections[nms_out_index]
        t = t[nms_out_index]

        # convert back to `ObjectLabels`
        xywh = xyxy2xywh(detections[:, :4], format_='corner', last4=True)
        obj_conf, class_conf, cls_id = torch.split(detections[:, 4:], 1, dim=1)
        detections = th.cat([t, xywh, cls_id, class_conf, obj_conf], dim=1)
        output[i] = ObjectLabels(detections, pad.input_size_hw)

    return output


class EventSeqData:
    """Record the labels of an event sequence."""

    def __init__(self, path: str, scale_ratio: int,
                 filter_config: DictConfig, postproc_cfg: DictConfig):
        self.path = path
        self.scale_ratio = scale_ratio
        self.filter_config = filter_config
        self.postproc_cfg = postproc_cfg
        self._eoe, self._aug = False, False
        # remove empty bbox or padded ones, only record idx of labeled frames
        self.frame_idx_2_labels: Dict[int, ObjectLabels] = {}

    def update(self, labels: List[ObjectLabels], ev_idx: List[int],
               is_last_sample: bool, is_padded_mask: List[bool],
               is_hflip: bool, is_tflip: bool, tflip_offset: int) -> None:
        """Append new model predictions to self."""
        self._eoe = is_last_sample
        if is_hflip:
            labels = self._hflip_bbox(labels)
            self._aug = True
        if is_tflip:
            ev_idx = [i + tflip_offset for i in ev_idx]
            self._aug = True
        self._update(labels, ev_idx, is_padded_mask)

    def _hflip_bbox(self, bboxes: List[ObjectLabels]) -> List[ObjectLabels]:
        """Apply horizontal flip to bboxes."""
        if len(bboxes) == 0:
            return bboxes
        # flip back, then to xyxy format
        for i, bbox in enumerate(bboxes):
            if bbox is not None:
                bbox.flip_lr_()
                bboxes[i] = bbox
        return bboxes

    def _update(self, labels: List[ObjectLabels], ev_idx: List[int],
                is_padded_mask: List[bool]) -> None:
        for tidx, (label, frame_idx) in enumerate(zip(labels, ev_idx)):
            if frame_idx < 0 or label is None or len(label) == 0:
                continue
            assert not is_padded_mask[tidx]
            # the labels are saved without downsampling
            # need to scale them back to original size when saving
            label.scale_(self.scale_ratio)
            # store labels to the frame
            if frame_idx in self.frame_idx_2_labels:
                # we only add once for GT labels!
                if label.is_gt_label().any():
                    try:
                        assert label == self.frame_idx_2_labels[frame_idx], \
                            'Different GT on the same frame!'
                    except AssertionError:
                        gt = self.frame_idx_2_labels[frame_idx].object_labels
                        lbl = label.object_labels
                        diff = (gt - lbl).abs().max().item()
                        print(f'Warning: GT max difference: {diff:.6f}')
                    continue
                # append predicted labels for later aggregation
                self.frame_idx_2_labels[frame_idx] += label
            else:
                self.frame_idx_2_labels[frame_idx] = label

    def _aggregate_results(self, num_frames: int) -> None:
        """Merge TTA predictions if applicable."""
        assert self._eoe, 'Cannot aggregate results before the sequence ends.'
        # convert `frame_idx_2_labels` to `frame_idx` and `labels`
        if len(self.frame_idx_2_labels) == 0:
            self.frame_idx, self.labels = [], []
            return
        # sort labels by frame_idx
        frame_idx = [
            idx for idx in self.frame_idx_2_labels.keys()
            if 0 <= idx < num_frames
        ]
        self.frame_idx = sorted(frame_idx)
        self.labels = [self.frame_idx_2_labels[idx] for idx in self.frame_idx]
        if not self._aug:  # no need to aggregate TTA results
            return
        self.labels = tta_postprocess(
            self.labels,
            conf_thre=self.postproc_cfg.confidence_threshold,
            nms_thre=self.postproc_cfg.nms_threshold)

    def _summarize(self) -> Tuple[np.ndarray]:
        """Convert to BBOX_DTYPE. Compute `objframe_idx_2_repr/label_idx`."""
        # concat labels, record the start and end idx of labels at each frame
        labels, lbl_cnt = [], 0
        objframe_idx_2_repr_idx, objframe_idx_2_label_idx = [], []
        for label, frame_idx in zip(self.labels, self.frame_idx):
            objframe_idx_2_label_idx.append(lbl_cnt)
            lbl_cnt += len(label)
            np_labels = label.to_structured_array()
            assert np_labels.dtype == BBOX_DTYPE
            labels.append(np_labels)
            objframe_idx_2_repr_idx.append(frame_idx)
        if len(labels) == 0:
            labels = np.zeros((0,), dtype=BBOX_DTYPE)
        else:
            labels = np.concatenate(labels)
        objframe_idx_2_label_idx = \
            np.array(objframe_idx_2_label_idx, dtype=np.int64)
        objframe_idx_2_repr_idx = \
            np.array(objframe_idx_2_repr_idx, dtype=np.int64)
        return labels, objframe_idx_2_label_idx, objframe_idx_2_repr_idx

    @staticmethod
    def _track(labels: List[ObjectLabels],
               frame_idx: List[int],
               min_track_len: int = 6,
               inpaint: bool = False) -> List[int]:
        """We track the bbox and filter out those from short tracklets."""
        assert min_track_len > 0, f'{min_track_len=} <= 0'
        assert len(labels) == len(frame_idx)
        if len(labels) == 0:
            return []
        model = LinearTracker(img_hw=labels[0].input_size_hw)
        for f_idx in range(max(frame_idx) + 1):
            if f_idx not in frame_idx:
                model.update(f_idx)  # call tracker even on empty label frames
                continue
            idx = frame_idx.index(f_idx)
            obj_label: ObjectLabels = labels[idx]
            assert obj_label is not None and len(obj_label) > 0
            # get bbox in [x,y,w,h,cls_id] format, (N, 5)
            obj_label.numpy_()
            bboxes = obj_label.get_xywh(format_='center', add_class_id=True)
            is_gt = obj_label.is_gt_label()
            model.update(frame_idx=f_idx, dets=bboxes, is_gt=is_gt)
        model.finish()
        # filter out short tracklets by looking at each bbox's track length
        bbox_idx, remove_idx = 0, []
        for obj_label in labels:
            for _ in range(len(obj_label)):
                tracker = model.get_bbox_tracker(bbox_idx)
                # keep if 1) unfinished, 2) GT, 3) many hits
                if (not tracker.done) or tracker.is_gt or \
                        tracker.hits >= min_track_len:
                    pass
                else:  # remove
                    remove_idx.append(bbox_idx)
                bbox_idx += 1
        if not inpaint:
            return remove_idx, {}

        # hallucinate bbox at frames where the tracklet has no matching bbox
        inpainted_bbox = {}
        for tracker in model.prev_trackers:
            if tracker.done and (not tracker.is_gt) and \
                    tracker.hits < min_track_len:
                continue
            for f_idx, bbox in tracker.missed_bbox.items():
                if f_idx not in inpainted_bbox:
                    inpainted_bbox[f_idx] = []
                inpainted_bbox[f_idx].append(bbox)  # [x,y,w,h,cls_id]

        def _postproc_inpaint_bbox(bbox: List[np.ndarray]) -> np.ndarray:
            """Post-process inpainted bbox."""
            bbox_ = np.stack(bbox)  # [N, 5], [x,y,w,h,cls_id], center format
            # [t,x,y,w,h,cls_id,cls_conf,obj_conf], (xywh) in corner format
            # leave t,cls_conf,obj_conf as 0
            bbox = np.zeros((bbox_.shape[0], 8), dtype=np.float32)
            bbox[:, 1] = bbox_[:, 0] - bbox_[:, 2] / 2.  # x1
            bbox[:, 2] = bbox_[:, 1] - bbox_[:, 3] / 2.  # y1
            bbox[:, 3:6] = bbox_[:, 2:5]  # w,h,cls_id
            return bbox

        inpainted_bbox = {
            k: _postproc_inpaint_bbox(v)
            for k, v in inpainted_bbox.items()
        }
        return remove_idx, inpainted_bbox

    def _track_filter(self) -> None:
        """We might track in both directions, and take and/or."""
        if len(self.labels) == 0:
            return
        min_track_len = self.filter_config.min_track_len
        if min_track_len <= 0:
            return
        track_method = self.filter_config.track_method
        assert track_method in ['forward', 'forward or backward'], \
            f'Unknown tracking post-processing {track_method}'
        # forward tracking
        remove_idx, inpainted_bbox = self._track(
            self.labels,
            self.frame_idx,
            min_track_len=min_track_len,
            inpaint=self.filter_config.inpaint)
        # backward tracking
        if 'backward' in track_method:
            rev_labels = [label.get_reverse() for label in self.labels[::-1]]
            rev_frame_idx = [
                max(self.frame_idx) - idx for idx in self.frame_idx[::-1]
            ]
            bg_remove_idx, _ = self._track(
                rev_labels,
                rev_frame_idx,
                min_track_len=min_track_len,
                inpaint=False)
            nlabels = sum(len(label) for label in self.labels)
            bg_remove_idx = [nlabels - idx - 1 for idx in bg_remove_idx[::-1]]
            # "or": both dir to remove --> remove
            remove_idx = list(set(remove_idx) & set(bg_remove_idx))
        # remove by setting class_id to 1024 (cannot use -1 as it is uint32)
        # will be ignored in loss computation during model training
        bbox_idx = 0
        for idx, obj_label in enumerate(self.labels):
            new_class_id = copy.deepcopy(obj_label.class_id)
            for i in range(len(obj_label)):
                if bbox_idx in remove_idx:
                    new_class_id[i] = self.filter_config.ignore_label
                    assert obj_label.is_pseudo_label().all(), 'Ignoring GT!'
                bbox_idx += 1
            self.labels[idx].class_id = new_class_id
        if not inpainted_bbox:
            return

        # inpaint bbox, also set the class_id to 1024 as ignore during training
        for f_idx in range(max(self.frame_idx) + 1):
            if f_idx not in inpainted_bbox:
                continue
            obj_label = inpainted_bbox[f_idx]  # [n, 8]
            obj_label[:, 5] = self.filter_config.ignore_label
            obj_label = ObjectLabels(obj_label, self.labels[0].input_size_hw)
            # find the right position to insert, or append to existing labels
            if f_idx in self.frame_idx:
                idx = self.frame_idx.index(f_idx)
                assert self.labels[idx].is_pseudo_label().all(), \
                    'Inpaint ignored bbox at labeled frames!'
                self.labels[idx] += obj_label
            else:
                self.frame_idx.append(f_idx)
                self.labels.append(obj_label)
        # sort by frame_idx
        self.labels = [
            label for _, label in sorted(zip(self.frame_idx, self.labels))
        ]
        self.frame_idx = sorted(self.frame_idx)

    def save(self, save_dir: str, dst_name: str) -> None:
        """Save labels, soft-link ev_repr.
        save_dir + self.path
            ├── event_representations_v2
            │ └── ev_representation_name
            │     ├── event_representations.h5  # soft-link
            │     ├── objframe_idx_2_repr_idx.npy
            └── labels_v2
                └── labels.npz  # have `labels` and `objframe_idx_2_label_idx`
        """
        # save_dir: path/to/dataset/train/
        # self.path: path/to/dataset/train/18-03-29_13-15-02_500000_60500000
        assert dst_name in ['gen1', 'gen4']
        assert 'train' in save_dir and dst_name in save_dir
        assert 'train' in self.path and dst_name in self.path

        # get original path/to/dataset
        base_dir = osp.dirname(osp.dirname(self.path))
        ev_dir = get_ev_dir(self.path)
        ev_h5_fn = get_ev_h5_fn(ev_dir)
        while osp.islink(ev_h5_fn):
            ev_h5_fn = os.readlink(ev_h5_fn)
        with h5py.File(ev_h5_fn, 'r') as h5f:
            num_ev_repr = h5f['data'].shape[0]

        # get new path/to/dataset
        new_base_dir = osp.dirname(save_dir)
        new_seq_dir = osp.join(save_dir, osp.basename(self.path))
        new_ev_dir = get_ev_dir(new_seq_dir)
        new_ev_h5_fn = get_ev_h5_fn(new_ev_dir)
        new_labels_npz_fn = get_labels_npz_fn(new_seq_dir)
        os.makedirs(new_ev_dir, exist_ok=False)
        os.makedirs(os.path.dirname(new_labels_npz_fn), exist_ok=False)

        # soft-link ev_repr
        os.symlink(ev_h5_fn, new_ev_h5_fn)
        # post-process and gather labels
        self._aggregate_results(num_frames=num_ev_repr)
        self._track_filter()
        labels, objframe_idx_2_label_idx, objframe_idx_2_repr_idx = \
            self._summarize()
        # save objframe_idx_2_repr_idx
        np.save(
            osp.join(new_ev_dir, 'objframe_idx_2_repr_idx.npy'),
            objframe_idx_2_repr_idx)
        # save labels
        np.savez(
            new_labels_npz_fn,
            labels=labels,
            objframe_idx_2_label_idx=objframe_idx_2_label_idx)

        # also link the val/test set for completeness
        val_dir = osp.join(base_dir, 'val')
        test_dir = osp.join(base_dir, 'test')
        while osp.islink(val_dir):
            val_dir = os.readlink(val_dir)
        while osp.islink(test_dir):
            test_dir = os.readlink(test_dir)
        if osp.islink(osp.join(new_base_dir, 'val')):
            assert osp.islink(osp.join(new_base_dir, 'test'))
            return
        os.symlink(val_dir, osp.join(new_base_dir, 'val'))
        os.symlink(test_dir, osp.join(new_base_dir, 'test'))

    @property
    def eoe(self) -> bool:
        """Whether the sequence has ended."""
        return self._eoe

    @property
    def aug(self) -> bool:
        """Whether the sequence has been applied TTA."""
        return self._aug


class PseudoLabeler(Module):
    """Generate pseudo labels on training data."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mode_2_seq_lens = SeqLens()  # record seq lens

        self.ev_path_2_ev_data: Dict[str, EventSeqData] = {}  # store events
        self.ev_cnt = 0

        self.dst_name = dst_name = self.dst_config.name
        self.ds_by2 = self.dst_config.downsample_by_factor_2
        assert dst_name in ['gen1', 'gen4'], f'Unknown dataset {dst_name}'
        self.save_dir = self.full_config.save_dir
        assert dst_name in self.save_dir, \
            'save_dir must contain the name of the dataset'
        assert 'train' in self.save_dir, \
            'we are creating pseudo labels for the training set'
        assert not osp.exists(self.save_dir), f'{self.save_dir} already exists'
        # whether to take GT labels or generated pseudo labels on labeled frame
        # should be True on Gen1 as their GT labels are manually annotated
        # but on Gen4, the GT labels are generated by another trained model
        # so they could be noisy, we might want to use the pseudo labels
        self.use_gt = self.full_config.get('use_gt', True)
        if not self.use_gt:
            assert 'all_pse' in self.save_dir
            assert dst_name == 'gen4', 'only Gen4 has noisy GT labels'

        # post-processing & filtering config
        print('\nobj_thresh:', self.mdl_config.pseudo_label.obj_thresh)
        print('cls_thresh:', self.mdl_config.pseudo_label.cls_thresh, '\n')

        # Test-Time-Augmentation (TTA) to enhance the generated label quality
        self.tta_cfg = self.full_config.tta
        if self.tta_cfg.enable:
            assert 'tta' in self.save_dir.lower(), 'tta must be in save_dir'

        # record metrics measuring pseudo label quality
        self.metrics: Dict[str, AverageMeter] = {}
        self.results: Dict[str, List[float]] = {}

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage=stage)
        # might need to filter pred bbox according to dataset criterion
        self.filter_bbox_fn = lambda bbox: filter_pred_boxes(
            bbox, dataset_name=self.dst_name, downsampled_by_2=self.ds_by2)

    def get_data_from_batch(self, batch: Any):
        data = batch[DATA_KEY]
        assert DataType.AUGM_STATE not in data, \
            'should not apply data augmentation in testing'
        # pad ev_repr to desired HxW here
        ev_repr = th.stack(data[DataType.EV_REPR]).to(dtype=self.dtype)
        # [L, B, C, H, W], event reprs
        B = ev_repr.shape[1]
        data['is_hflip'] = np.array([False] * B, dtype=bool)  # all False
        # if apply hflip TTA, manually apply it here
        if self.tta_cfg.enable and self.tta_cfg.hflip:
            hflip_ev_repr = th.flip(ev_repr, dims=[-1])
            ev_repr = th.cat([ev_repr, hflip_ev_repr], dim=1)  # 2B
            # simply duplicate other data items
            new_data = {}
            # for th.Tensor with shape [B] or [L, B]
            for k in (DataType.IS_FIRST_SAMPLE, DataType.IS_LAST_SAMPLE,
                      DataType.IS_REVERSED):
                new_data[k] = th.cat([data[k]] * 2, dim=-1)
            # for List[th.Tensor]
            for k in (DataType.EV_IDX, DataType.IS_PADDED_MASK):
                new_data[k] = [th.cat([d] * 2, dim=-1) for d in data[k]]
            # for List[str]
            new_data[DataType.PATH] = data[DataType.PATH] * 2
            # for `L`-len List[SparselyBatchedObjectLabels]
            for k in (DataType.OBJLABELS_SEQ, DataType.SKIPPED_OBJLABELS_SEQ):
                labels, labels_flip = data[k], copy.deepcopy(data[k])
                for i, (lbl, lbl_flip) in enumerate(zip(labels, labels_flip)):
                    lbl_flip.flip_lr_()
                    labels[i] = lbl + lbl_flip
                new_data[k] = labels
            # indicator
            is_hflip = np.array([False] * B + [True] * B, dtype=bool)
            new_data['is_hflip'] = is_hflip
            data = new_data
        ev_repr = self.input_padder.pad_tensor_ev_repr(ev_repr)
        data[DataType.EV_REPR] = [ev for ev in ev_repr]  # back to list
        return data

    def collect_data(self, data):
        # make them all list of B or BxL
        ev_paths = data[DataType.PATH]
        ev_idx = torch.stack(data[DataType.EV_IDX]).transpose(1, 0).\
            cpu().numpy().tolist()
        is_first_sample = data[DataType.IS_FIRST_SAMPLE].cpu().numpy().tolist()
        is_last_sample = data[DataType.IS_LAST_SAMPLE].cpu().numpy().tolist()
        padding = torch.stack(data[DataType.IS_PADDED_MASK]).transpose(1, 0).\
            cpu().numpy().tolist()
        if 'is_hflip' in data:
            is_hflip = data['is_hflip']
        else:
            is_hflip = [False] * len(ev_paths)
        is_tflip = data[DataType.IS_REVERSED].cpu().numpy().tolist()
        return ev_paths, ev_idx, is_first_sample, is_last_sample, \
            padding, is_hflip, is_tflip

    def _get_pred_mask(self, worker_id: int, data: Dict) -> Tuple[np.ndarray]:
        """Get frames (timesteps) we need to predict bbox."""
        obj_labels = data[DataType.OBJLABELS_SEQ]
        # a `L`-len list of `SparselyBatchedObjectLabels`
        #   each contains a `B`-len list of `ObjectLabels` or None
        skipped_obj_labels = data[DataType.SKIPPED_OBJLABELS_SEQ]
        # similar to `obj_labels`
        L, B = len(obj_labels), len(obj_labels[0])
        skip_mask = np.zeros((L, B), dtype=bool)  # True --> skip
        gt_mask = np.zeros((L, B), dtype=bool)  # True --> has GT label
        skipped_gt_mask = np.zeros((L, B), dtype=bool)  # True --> skipped GT
        # 1. skip the first X frames
        skip_len = max(self.mdl_config.pseudo_label.skip_first_t, 1)
        prev_lens = self.mode_2_seq_lens.get_lens(worker_id=worker_id)  # [B]
        # `prev_lens` is after reset!
        for bidx in range(B):
            if prev_lens[bidx] < skip_len:
                skip_mask[:skip_len - prev_lens[bidx], bidx] = True
        # 2. skip frames with GT labels (NOT including skipped GT)
        for tidx in range(L):
            for bidx in range(B):
                has_gt = (obj_labels[tidx][bidx] is not None) and self.use_gt
                has_skipped_gt = (skipped_obj_labels[tidx][bidx] is not None)
                assert not (has_gt and has_skipped_gt)
                gt_mask[tidx, bidx] = has_gt
                skip_mask[tidx, bidx] = has_gt
                skipped_gt_mask[tidx, bidx] = has_skipped_gt
        # 3. skip padded frames
        padded_mask = torch.stack(data[DataType.IS_PADDED_MASK]).cpu().numpy()
        # [L, B], bool. True --> padded --> skip
        skip_mask[padded_mask] = True
        # True --> need to generate pseudo labels
        # True --> has (skipped) GT label
        return (~skip_mask), gt_mask, skipped_gt_mask

    def _get_label_and_index(self, gt_obj_label: SparselyBatchedObjectLabels,
                             skipped_gt_obj_label: SparselyBatchedObjectLabels,
                             pse_mask: np.ndarray):
        """Get the GT and pseudo labels and their indices."""
        if self.use_gt:
            gt_labels, gt_indices = \
                gt_obj_label.get_valid_labels_and_batch_indices()
        else:
            gt_labels, gt_indices = [], []
        skipped_gt_labels, skipped_gt_indices = \
            skipped_gt_obj_label.get_valid_labels_and_batch_indices()
        # pred_mask: [B], True --> need to generate pseudo labels on this frame
        pse_indices = np.where(pse_mask)[0]
        return gt_labels, gt_indices, skipped_gt_labels, skipped_gt_indices, \
            pse_indices

    def _predict_bbox(self,
                      backbone_feature_selector: BackboneFeatureSelector) -> \
            List[th.Tensor]:
        """Run detection head on backbone features and do post-processing."""
        # backbone features are now dict: {stage_id: [B', C, h, w]}
        selected_backbone_features = \
            backbone_feature_selector.get_batched_backbone_features()
        if selected_backbone_features is None:  # no frames to predict
            return None

        # predictions: (B', N, 4 + 1 + num_cls), [(x, y, w, h), obj_conf, cls]
        predictions, _ = self.mdl.forward_detect(
            backbone_features=selected_backbone_features)

        # pred_processed: `B'`-len List[(N_i, 7)] (empty bbox is [0, 7])
        #   7: [(x1, y1, x2, y2), obj_conf, cls_conf, cls_idx]
        pad_bbox = th.zeros((0, 7)).type_as(predictions).detach()
        pred_processed = postprocess(
            prediction=predictions,
            num_classes=self.mdl_config.head.num_classes,
            conf_thre=self.mdl_config.postprocess.confidence_threshold,
            nms_thre=self.mdl_config.postprocess.nms_threshold,
            pad=pad_bbox)

        return pred_processed

    def _evaluate_pseudo_label(self, gt_obj_labels: List[ObjectLabels],
                               pse_obj_labels: List[ObjectLabels]) -> None:
        # Precision & Recall
        pred_mask = np.ones(len(gt_obj_labels), dtype=bool)
        metrics = evaluate_label(
            gt_obj_labels,
            pse_obj_labels,
            pred_mask=pred_mask,
            num_cls=self.num_classes,
            prefix='ssod/')
        for k, v in metrics.items():
            if k.startswith('num_'):
                continue
            if k not in self.metrics:
                self.metrics[k] = AverageMeter()
            cls_name = k.split('_')[-1]  # xxx_car
            self.metrics[k].update(v, n=metrics[f'num_{cls_name}'])
        # collect predicted bbox's IoUs, cls/obj scores
        if self.results and len(self.results['ssod/true_ious_all']) > 1e5:
            return
        result_dict = get_scores_ious(
            gt_label=gt_obj_labels,
            pseudo_label=pse_obj_labels,
            pred_mask=pred_mask,
            num_cls=self.num_classes,
            prefix='ssod/')
        for k, v in result_dict.items():
            if k not in self.results:
                self.results[k] = []
            self.results[k] += v

    @torch.inference_mode()
    def _predict_step_impl(self, batch: Any, mode: Mode) -> STEP_OUTPUT:
        """Predict bbox on a batch of event sequences."""
        # TODO: an ugly hack for tracking-only post-processing
        if self.dst_config.only_load_labels:
            data = batch[DATA_KEY]
            skipped_obj_labels = data[DataType.SKIPPED_OBJLABELS_SEQ]
            assert all([lbl.is_empty() for lbl in skipped_obj_labels])
            # make LxB to BxL
            obj_labels = data[DataType.OBJLABELS_SEQ]
            obj_labels = [list(lbl) for lbl in zip(*obj_labels)]
            # prepare for saving pseudo labels
            ev_paths, ev_idx, is_first_sample, is_last_sample, \
                padding, is_hflip, is_tflip = self.collect_data(data)
            return obj_labels, ev_paths, ev_idx, \
                is_first_sample, is_last_sample, padding, is_hflip, is_tflip

        # normal prediction
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        assert mode == Mode.TEST  # but we're actually running on training data
        ev_tensor_sequence = data[DataType.EV_REPR]
        # `L`-len list of [B, C, H, W], torch.Tensor event reprs
        obj_labels = data[DataType.OBJLABELS_SEQ]
        # a `L`-len list of `SparselyBatchedObjectLabels`
        #   each contains a `B`-len list of `ObjectLabels` or None
        skipped_obj_labels = data[DataType.SKIPPED_OBJLABELS_SEQ]
        # similar to `obj_labels`, they are exclusive
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        # [B], bool. Will reset RNN state if True

        L, B = len(obj_labels), len(obj_labels[0])
        assert L > 0 and B > 0
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = B
        else:
            assert self.mode_2_batch_size[mode] == B

        # update RNN states, seq_len stats
        self.mode_2_rnn_states[mode].reset(
            worker_id=worker_id, indices_or_bool_tensor=is_first_sample)
        prev_states = \
            self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        self.mode_2_seq_lens.reset(
            worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        # store backbone features at frames to be pseudo-labeled
        pse_backbone_feature_selector = BackboneFeatureSelector()
        gt_obj_labels, skipped_gt_obj_labels = list(), list()
        pse_mask, gt_mask, skipped_gt_mask = self._get_pred_mask(
            worker_id=worker_id, data=data)  # np.bool
        if not self.use_gt:
            assert gt_mask.sum() == 0, 'should not use GT labels'
        for tidx in range(L):
            ev_tensors = ev_tensor_sequence[tidx]
            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            backbone_features, states = self.mdl.forward_backbone(
                x=ev_tensors, previous_states=prev_states)
            # `backbone_features`: dict{stage_id: feats, [B, C, h, w]}
            # `states`: list[(lstm_h, lstm_c), same shape]
            prev_states = states

            # gather features on frames we need to predict
            # 1. frames with GT labels, just store them
            # 2. frames with skipped GT, generate pseudo labels + evaluate
            # 3. frames without labels, to generate pseudo labels
            gt_labels, gt_indices, skipped_gt_labels, skipped_gt_indices, \
                pse_indices = self._get_label_and_index(
                    obj_labels[tidx], skipped_obj_labels[tidx], pse_mask[tidx])
            if len(gt_indices) > 0:
                assert self.use_gt
                gt_obj_labels.extend(gt_labels)
            if len(skipped_gt_indices) > 0:
                skipped_gt_obj_labels.extend(skipped_gt_labels)
            if len(pse_indices) > 0:
                pse_backbone_feature_selector.add_backbone_features(
                    backbone_features=backbone_features,
                    selected_indices=pse_indices)

        self.mode_2_rnn_states[mode].save_states_and_detach(
            worker_id=worker_id, states=prev_states)
        self.mode_2_seq_lens.update_lens(
            worker_id=worker_id, lens=torch.ones(B).long() * L)

        # predict on frames to be pseudo-labeled
        # pred_processed: `B'`-len List[(N_i, 7)] (empty bbox is [0, 7])
        #   7: [(x1, y1, x2, y2), obj_conf, cls_conf, cls_idx]
        pse_pred_processed = self._predict_bbox(pse_backbone_feature_selector)
        # filter out low-quality bbox
        if pse_pred_processed:
            pse_labels = pred2label(
                pred=pse_pred_processed,
                obj_thresh=self.mdl_config.pseudo_label.obj_thresh,
                cls_thresh=self.mdl_config.pseudo_label.cls_thresh,
                filter_bbox_fn=self.filter_bbox_fn,
                hw=tuple(self.dst_config.ev_repr_hw))
        else:
            pse_labels = []  # no frames to be pseudo-labeled
        # pse_labels: `B'`-len list of `ObjectLabels`
        # (skipped_)gt_obj_labels: `num_gt`-len list of `ObjectLabels`
        # the field `t` are all set as 0

        # 1. select frames with skipped GT labels to evaluate
        # 2. rearrange back to `BxL` List[List[`ObjectLabels`]]
        skipped_gt_pse_labels = []
        all_labels, gt_cnt, pse_cnt = np.zeros((B, L)).tolist(), 0, 0
        for tidx in range(L):
            for bidx in range(B):
                is_pse, is_gt, is_skipped_gt = pse_mask[tidx, bidx], \
                    gt_mask[tidx, bidx], skipped_gt_mask[tidx, bidx]
                if is_skipped_gt:
                    assert is_pse, 'should predict on skipped GT frames'
                    skipped_gt_pse_labels.append(pse_labels[pse_cnt])
                assert not (is_pse and is_gt), 'do not predict on GT frames'
                if is_pse:
                    all_labels[bidx][tidx] = pse_labels[pse_cnt]
                    pse_cnt += 1
                elif is_gt:
                    all_labels[bidx][tidx] = gt_obj_labels[gt_cnt]
                    gt_cnt += 1
                else:
                    all_labels[bidx][tidx] = None

        assert pse_cnt == pse_mask.sum() and gt_cnt == gt_mask.sum() and \
            len(skipped_gt_pse_labels) == skipped_gt_mask.sum()
        # save for eval later
        if len(skipped_gt_pse_labels) == 0:  # no frame has skipped GT
            assert len(skipped_gt_obj_labels) == 0
        else:
            # eval precision, recall
            self._evaluate_pseudo_label(skipped_gt_obj_labels,
                                        skipped_gt_pse_labels)
            # evaluate real COCO-based AP, AR
            loaded_labels_proph, yolox_preds_proph = to_prophesee(
                skipped_gt_obj_labels, skipped_gt_pse_labels)
            self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
            self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)

        # prepare for saving pseudo labels
        ev_paths, ev_idx, is_first_sample, is_last_sample, \
            padding, is_hflip, is_tflip = self.collect_data(data)

        return all_labels, ev_paths, ev_idx, \
            is_first_sample, is_last_sample, padding, is_hflip, is_tflip

    def predict_step(self, batch: Any, batch_idx: int) -> None:
        out = self._predict_step_impl(batch=batch, mode=Mode.TEST)
        # create new event data or update existing ones
        for (labels, ev_path, ev_idx, is_first, is_last, padded, is_hflip,
             is_tflip) in zip(*out):
            if not ev_path:
                assert not is_first and not is_last and all(padded) and \
                    all(i == -1 for i in ev_idx), 'invalid empty data'
                continue
            if ev_path not in self.ev_path_2_ev_data:
                assert is_first, 'should load the first sample first'
                self.ev_path_2_ev_data[ev_path] = EventSeqData(
                    path=ev_path,
                    scale_ratio=2. if self.ds_by2 else 1,
                    filter_config=self.mdl_config.pseudo_label,
                    postproc_cfg=self.mdl_config.postprocess)
                self.ev_cnt += 1
            self.ev_path_2_ev_data[ev_path].update(
                labels=labels,
                ev_idx=ev_idx,
                is_last_sample=is_last,
                is_padded_mask=padded,
                is_hflip=is_hflip,
                is_tflip=is_tflip,
                tflip_offset=self.dst_config.data_augmentation.tflip_offset)
