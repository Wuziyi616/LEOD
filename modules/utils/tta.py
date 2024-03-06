import os
from typing import Any, Dict, Tuple, List

import numpy as np
from omegaconf import DictConfig
import torch as th
import torchvision.ops as ops
from pytorch_lightning.utilities.types import STEP_OUTPUT

from data.genx_utils.labels import ObjectLabels
from data.utils.types import DataType, DatasetSamplingMode
from models.detection.yolox.utils.boxes import postprocess
from utils.evaluation.prophesee.io.box_loading import to_prophesee
from modules.detection import Module
from .detection import BackboneFeatureSelector, Mode, DATA_KEY


def tta_postprocess(preds: List[th.Tensor],
                    conf_thre: float = 0.7,
                    nms_thre: float = 0.45,
                    class_agnostic: bool = False,
                    pad=None) -> List[th.Tensor]:
    """Apply NMS on predicted bboxes.

    Input:
        preds: List[(N_i, 7)], [(xyxy), obj_conf, cls_conf, cls_idx]

    Returns:
        output: same as `preds` but with NMS applied
    """
    output = [pad] * len(preds)
    for i, pred in enumerate(preds):
        # If none are remaining => process next image
        if not pred.size(0):
            continue

        obj_conf, class_conf = pred[:, 4], pred[:, 5]
        conf_mask = ((obj_conf * class_conf) >= conf_thre)  # (N,)
        detections = pred
        detections = detections[conf_mask]
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
        output[i] = detections

    return output


class EventSeqResult:
    """Aggregate model predictions from TTA (Test-Time-Augmentation)."""

    def __init__(self, path: str, img_hw: Tuple[int, int],
                 postproc_cfg: DictConfig):
        self.path = path
        self.img_hw = img_hw
        self.postproc_cfg = postproc_cfg
        self._eoe, self._aug = False, False  # just started
        # record idx of the GT labels & model predictions
        self.ev_idx_2_pred: Dict[int, th.Tensor] = {}
        self.ev_idx_2_gt: Dict[int, ObjectLabels] = {}

    def update(self, is_hflip: bool, is_tflip: bool, preds: List[th.Tensor],
               gts: List[ObjectLabels], ev_idx: List[int],
               is_last_sample: bool, tflip_offset: int) -> None:
        """General interface for updating new predictions and labels."""
        # get only prediction frames with GT
        preds_, gts_, ev_idx_ = [], [], []
        for pred, gt, idx in zip(preds, gts, ev_idx):
            if not isinstance(gt, ObjectLabels) or len(gt) == 0:
                continue
            if isinstance(pred, ObjectLabels):
                # NMS requires xyxy format bbox
                pred = pred.get_labels_as_tensors(format_='prophesee')
            preds_.append(pred)
            gts_.append(gt)
            ev_idx_.append(idx)
        # handle different TTA cases
        if not is_hflip:
            if not is_tflip:
                self._update(preds_, gts_, ev_idx_, is_last_sample)
            else:
                self._update_tflip(preds_, ev_idx_, tflip_offset)
        else:
            if not is_tflip:
                self._update_hflip(preds_, ev_idx_)
            else:
                self._update_tflip_hflip(preds_, ev_idx_, tflip_offset)

    def _update_gt(self, gts: List[ObjectLabels], ev_idx: List[int]) -> None:
        """Append new GT labels to self."""
        assert not self._eoe, 'Cannot update a finished sequence.'
        for idx, gt in zip(ev_idx, gts):
            assert isinstance(gt, ObjectLabels) and len(gt) > 0, \
                'GT must be non-empty ObjectLabels.'
            assert idx not in self.ev_idx_2_gt, 'Duplicate label.'
            self.ev_idx_2_gt[idx] = gt  # ObjectLabels
            assert self.img_hw == gt.input_size_hw, 'Inconsistent image size.'

    def _update_pred(self, preds: List[th.Tensor], ev_idx: List[int]) -> None:
        """Append new model predictions to self."""
        for idx, pred in zip(ev_idx, preds):
            if idx not in self.ev_idx_2_pred:
                self.ev_idx_2_pred[idx] = pred
            else:
                self.ev_idx_2_pred[idx] = th.cat(
                    [self.ev_idx_2_pred[idx], pred], dim=0)

    def _update(self, preds: List[th.Tensor], gts: List[ObjectLabels],
                ev_idx: List[int], is_last_sample: bool) -> None:
        """Update labels without any TTA."""
        self._update_gt(gts=gts, ev_idx=ev_idx)
        self._update_pred(preds=preds, ev_idx=ev_idx)
        self._eoe = is_last_sample

    def _hflip_bbox(self, bboxes: List[th.Tensor]) -> List[th.Tensor]:
        """Apply horizontal flip to bboxes."""
        if len(bboxes) == 0:
            return bboxes
        if isinstance(bboxes[0], ObjectLabels):
            # flip back, then to xyxy format
            for i, bbox in enumerate(bboxes):
                bbox.flip_lr_()
                bboxes[i] = bbox.get_labels_as_tensors(format_='prophesee')
        else:
            # already xyxy, manually flip it
            for i, bbox in enumerate(bboxes):
                W = self.img_hw[1]
                w = bbox[:, 2] - bbox[:, 0]
                bbox[:, 0] = W - 1 - bbox[:, 0] - w
                bbox[:, 2] = bbox[:, 0] + w
                bboxes[i] = bbox
        return bboxes

    def _update_hflip(self, preds: List[th.Tensor], ev_idx: List[int]) -> None:
        """Update labels that undergo horizontal flip TTA."""
        self._aug = True
        preds = self._hflip_bbox(preds)
        self._update_pred(preds=preds, ev_idx=ev_idx)

    def _update_tflip(self,
                      preds: List[th.Tensor],
                      ev_idx: List[int],
                      offset: int = -1) -> None:
        """Update labels that undergo time-flip TTA."""
        self._aug = True
        # rev_pred with ev_idx i is actually the pred for ev_idx (i + offset)
        ev_idx = [i + offset for i in ev_idx]
        self._update_pred(preds=preds, ev_idx=ev_idx)

    def _update_tflip_hflip(self,
                            preds: List[th.Tensor],
                            ev_idx: List[int],
                            offset: int = -1) -> None:
        """Update labels that undergo time-flip and hflip TTA."""
        self._aug = True
        preds = self._hflip_bbox(preds)  # hflip back
        self._update_tflip(preds=preds, ev_idx=ev_idx, offset=offset)

    def aggregate_results(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Aggregate predictions and postprocess them."""
        assert self._eoe, 'Cannot aggregate results before the sequence ends.'
        ev_idx = sorted(self.ev_idx_2_pred.keys())
        assert ev_idx == sorted(self.ev_idx_2_gt.keys()), 'Missing labels.'
        gts = [self.ev_idx_2_gt[idx] for idx in ev_idx]
        preds = [self.ev_idx_2_pred[idx] for idx in ev_idx]
        if self._aug:  # aggregate predictions from multiple TTA views
            preds = tta_postprocess(
                preds,
                conf_thre=self.postproc_cfg.confidence_threshold,
                nms_thre=self.postproc_cfg.nms_threshold)
        loaded_labels_proph, yolox_preds_proph = to_prophesee(gts, preds)
        return loaded_labels_proph, yolox_preds_proph

    @property
    def aug(self) -> bool:
        """Whether the sequence has been augmented."""
        return self._aug

    @property
    def eoe(self) -> bool:
        """Whether the sequence has ended."""
        return self._eoe


class TTAModule(Module):
    """A wrapper for model inference with TTA."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ev_path_2_ev_pred: Dict[str, EventSeqResult] = {}  # store preds
        self.tta_cfg = self.full_config.tta
        self.postproc_cfg = self.mdl_config.postprocess

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
            new_data[DataType.EV_IDX] = [
                th.cat([idx] * 2, dim=-1) for idx in data[DataType.EV_IDX]
            ]
            # for List[str]
            new_data[DataType.PATH] = data[DataType.PATH] * 2
            # for `L`-len List[SparselyBatchedObjectLabels]
            sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
            for i, obj_labels in enumerate(sparse_obj_labels):
                sparse_obj_labels[i] = obj_labels + obj_labels
            new_data[DataType.OBJLABELS_SEQ] = sparse_obj_labels
            # indicator
            is_hflip = np.array([False] * B + [True] * B, dtype=bool)
            new_data['is_hflip'] = is_hflip
            data = new_data
        ev_repr = self.input_padder.pad_tensor_ev_repr(ev_repr)
        data[DataType.EV_REPR] = [ev for ev in ev_repr]  # back to list
        return data

    def _test_step_impl(self, batch: Any, mode: Mode) -> STEP_OUTPUT:
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        assert mode in (Mode.VAL, Mode.TEST)
        assert self.mode_2_sampling_mode[mode] == DatasetSamplingMode.STREAM, \
            'Should always test on streaming mode event sequences'
        ev_tensor_sequence = data[DataType.EV_REPR]
        # a `L`-len list, each [B, C, H, W], event reprs
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        # a `L`-len list of `SparselyBatchedObjectLabels`
        #   each contains a `B`-len list of `ObjectLabels` or None
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        # [B], bool. Will reset RNN state if True, i.e. the first sample of seq

        L, B = len(sparse_obj_labels), len(sparse_obj_labels[0])
        assert L > 0 and B > 0
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = B
        else:
            assert self.mode_2_batch_size[mode] == B

        # update RNN states
        self.mode_2_rnn_states[mode].reset(
            worker_id=worker_id, indices_or_bool_tensor=is_first_sample)
        prev_states = \
            self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)

        # store backbone features at labeled frames to apply detection head
        backbone_feature_selector = BackboneFeatureSelector()
        obj_labels, pred_idx = [], [[], []]
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

            # in streaming, we load the entire event seq only once
            #   so have to predict for all labeled frames in-between
            current_labels, valid_batch_indices = \
                sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
            # `current_labels`: list[ObjectLabels], length is num_valid_frames
            # `valid_batch_indices`: list[int], idx of valid frames in a batch
            # Store backbone features that correspond to the available labels.
            if len(current_labels) > 0:
                backbone_feature_selector.add_backbone_features(
                    backbone_features=backbone_features,
                    selected_indices=valid_batch_indices)
                obj_labels.extend(current_labels)
                pred_idx[0].extend([tidx] * len(current_labels))
                pred_idx[1].extend(valid_batch_indices)

        self.mode_2_rnn_states[mode].save_states_and_detach(
            worker_id=worker_id, states=prev_states)

        # no valid labels in this batch
        if backbone_feature_selector.is_empty():
            assert len(obj_labels) == 0
            return tuple([[]] * 8)

        # backbone features are now dict: {stage_id: [B, C, h, w]}
        selected_backbone_features = \
            backbone_feature_selector.get_batched_backbone_features()

        # predictions: (B, N, 4 + 1 + num_cls), [(x, y, w, h), obj_conf, cls]
        predictions, _ = self.mdl.forward_detect(
            backbone_features=selected_backbone_features)

        # pred_processed: `B`-len List[(N_i, 7)], [(xyxy), obj, cls, cls_idx]
        pad_bbox = th.zeros((0, 7)).type_as(predictions).detach()
        pred_processed = postprocess(
            prediction=predictions,
            num_classes=self.mdl_config.head.num_classes,
            conf_thre=self.mdl_config.postprocess.confidence_threshold,
            nms_thre=self.mdl_config.postprocess.nms_threshold,
            pad=pad_bbox)

        # organize the results to B or BxL, lists
        all_preds, all_gts = np.ones((B, L)).tolist(), np.ones((B, L)).tolist()
        for i, (tidx, bidx) in enumerate(zip(*pred_idx)):
            all_preds[bidx][tidx] = pred_processed[i]
            all_gts[bidx][tidx] = obj_labels[i]
        ev_paths = data[DataType.PATH]
        ev_idx = th.stack(data[DataType.EV_IDX]).transpose(1, 0).\
            cpu().numpy().tolist()
        is_first_sample = data[DataType.IS_FIRST_SAMPLE].cpu().numpy().tolist()
        is_last_sample = data[DataType.IS_LAST_SAMPLE].cpu().numpy().tolist()
        is_hflip = data['is_hflip'].tolist()
        is_tflip = data[DataType.IS_REVERSED].cpu().numpy().tolist()

        return all_preds, all_gts, ev_paths, ev_idx, is_first_sample, \
            is_last_sample, is_hflip, is_tflip

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        raise NotImplementedError('Only used for testing')

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        raise NotImplementedError('Only used for testing')

    def test_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        out = self._test_step_impl(batch=batch, mode=Mode.TEST)
        # create new event predictions or update existing ones
        for (preds, gts, ev_path, ev_idx, is_first_sample, is_last_sample,
             is_hflip, is_tflip) in zip(*out):
            if not ev_path:
                assert not is_first_sample and not is_last_sample and \
                    all(i == -1 for i in ev_idx), 'invalid empty data'
                continue
            ev_path = os.path.basename(ev_path)
            if ev_path not in self.ev_path_2_ev_pred:
                assert is_first_sample, 'should load the first sample first'
                self.ev_path_2_ev_pred[ev_path] = EventSeqResult(
                    path=ev_path,
                    img_hw=self.dst_config.ev_repr_hw,
                    postproc_cfg=self.postproc_cfg)
            self.ev_path_2_ev_pred[ev_path].update(
                is_hflip=is_hflip,
                is_tflip=is_tflip,
                preds=preds,
                gts=gts,
                ev_idx=ev_idx,
                is_last_sample=is_last_sample,
                tflip_offset=self.dst_config.data_augmentation.tflip_offset)

    def on_test_epoch_end(self) -> None:
        mode = Mode.TEST
        assert not self.mode_2_psee_evaluator[mode].has_data()
        # aggregate TTA results
        for ev_pred in self.ev_path_2_ev_pred.values():
            loaded_labels, yolox_preds = ev_pred.aggregate_results()
            self.mode_2_psee_evaluator[mode].add_labels(loaded_labels)
            self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds)
        self.run_psee_evaluator(mode=mode)
