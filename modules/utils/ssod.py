from typing import List, Union, Tuple, Callable

import numpy as np
import torch
import torch as th

from data.genx_utils.labels import ObjectLabels
from data.utils.types import BatchAugmState
from data.utils.ssod_augmentor import LabelAugmentorGenX
from utils.helpers import temporal_wrapper, th_cat
from utils.bbox import xyxy2xywh, get_bbox_coords, construct_bbox
from models.detection.yolox.utils import bboxes_iou
from utils.evaluation.prophesee.evaluator import get_labelmap

DATASET2HEIGHT = {'gen1': 240, 'gen4': 720}
DATASET2WIDTH = {'gen1': 304, 'gen4': 1280}


def get_subsample_label_idx(L: int,
                            use_every: int = -1,
                            remove_every: int = -1) -> List[int]:
    """Sub-sample labels from a long sequence."""
    assert use_every == -1 or remove_every == -1
    all_idx = list(range(L))
    if use_every == 1:
        return tuple(all_idx)
    if use_every > 0:
        use_idx = all_idx[1::use_every]  # don't use the first frame (rnd seq)
    elif remove_every > 0:
        remove_idx = all_idx[::remove_every]
        use_idx = list(set(all_idx) - set(remove_idx))
    else:
        raise ValueError('Either use_every or remove_every must be > 0')
    # make sure we use labels on the last frame
    if L - 1 not in use_idx:
        use_idx.append(L - 1)
    return tuple(use_idx)


def _crop_to_fov_filter(xyxy: Tuple[th.Tensor],
                        dataset_name: str = 'gen1',
                        downsampled_by_2: bool = False):
    """Fix the bbox that are partially or completely outside the frame.
    See https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/issues/19.
    """
    frame_height = DATASET2HEIGHT[dataset_name]
    frame_width = DATASET2WIDTH[dataset_name]
    if downsampled_by_2:
        frame_height //= 2
        frame_width //= 2

    x1, y1, x2, y2 = xyxy
    x1_cropped = torch.clamp(x1, min=0., max=frame_width - 1.)
    y1_cropped = torch.clamp(y1, min=0., max=frame_height - 1.)
    x2_cropped = torch.clamp(x2, min=0., max=frame_width - 1.)
    y2_cropped = torch.clamp(y2, min=0., max=frame_height - 1.)
    w_cropped = x2_cropped - x1_cropped
    h_cropped = y2_cropped - y1_cropped

    # remove bboxes that have 0 height or width, i.e. are outside the FOV
    keep = (w_cropped > 0) & (h_cropped > 0)

    return (x1_cropped, y1_cropped, x2_cropped, y2_cropped), keep


def _prophesee_bbox_filter(xyxy: Tuple[th.Tensor],
                           dataset_name: str = 'gen1',
                           downsampled_by_2: bool = False):
    """Filter bbox that are too small by its height, width, or diag length."""
    # follow preprocessing, use conservative bbox filter for gen4
    if dataset_name == 'gen4':
        return _conservative_bbox_filter(xyxy)

    min_box_diag = 60 if dataset_name == 'gen4' else 30
    # in the supplementary mat, they say that min_box_side is 20 for gen4.
    min_box_side = 20 if dataset_name == 'gen4' else 10
    if downsampled_by_2:
        min_box_diag //= 2
        min_box_side //= 2

    x1, y1, x2, y2 = xyxy
    width, height = (x2 - x1), (y2 - y1)
    diag_square = width**2 + height**2
    keep = (diag_square >= min_box_diag**2) & (width >= min_box_side) & (
        height >= min_box_side)

    return keep


def _conservative_bbox_filter(xyxy: Tuple[th.Tensor], **kwargs):
    """Filter bbox that are too small by its height, width, or diag length."""
    # this is used on Gen4, see `preprocess_dataset.py`
    min_box_side = 5
    x1, y1, x2, y2 = xyxy
    width, height = (x2 - x1), (y2 - y1)
    keep = (width >= min_box_side) & (height >= min_box_side)
    return keep


def _faulty_huge_bbox_filter(xyxy: Tuple[th.Tensor],
                             dataset_name: str = 'gen1',
                             downsampled_by_2: bool = False):
    """Filter bbox that are super wide without covering objects."""
    frame_width = DATASET2WIDTH[dataset_name]
    if downsampled_by_2:
        frame_width //= 2
    max_width = (9 * frame_width) // 10
    width = xyxy[2] - xyxy[0]
    keep = (width <= max_width)
    return keep


def filter_pred_boxes(boxes: th.Tensor,
                      dataset_name: str = 'gen1',
                      downsampled_by_2: bool = False):
    """Filter bbox as in data pre-processing."""
    # boxes: [N, 4 (x1, y1, x2, y2)]
    xyxy, last4 = get_bbox_coords(boxes, last4=True)

    # 1. crop to FOV
    xyxy, keep = _crop_to_fov_filter(xyxy, dataset_name, downsampled_by_2)

    # 2. filter small bbox
    # keep &= _prophesee_bbox_filter(xyxy, dataset_name, downsampled_by_2)
    keep &= _conservative_bbox_filter(xyxy)  # TODO: better be conservative?

    # 3. remove faulty huge bbox
    keep &= _faulty_huge_bbox_filter(xyxy, dataset_name, downsampled_by_2)

    # xyxy back to bbox
    boxes = construct_bbox(xyxy, last4)

    return boxes, keep


def filter_w_thresh(scores: th.Tensor, class_ids: th.Tensor,
                    thresh: Union[float, List[float]]):
    """Filter the scores with one or per-class thresholds."""
    if isinstance(thresh, float):
        return (scores > thresh)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    for i, t in enumerate(thresh):
        mask |= ((class_ids == i) & (scores > t))
    return mask


@temporal_wrapper
def pred2label(pred: List[th.Tensor],
               obj_thresh: Union[float, List[float]] = 0.9,
               cls_thresh: Union[float, List[float]] = 0.9,
               filter_bbox_fn: Callable = None,
               hw: Tuple[int, int] = (-1, -1)):
    """Convert the model prediction after post-processing to pseudo GT.

    1. Filter with objectness score threshold.
    2. Filter with class score threshold.
    3. (Optional) Filter as in `preprocess_dataset.py`.
    """
    # pred: `B`-len List[(N_i, 7)], [(x1, y1, x2, y2), obj_conf, cls_conf, cls_idx]
    lst_lens = [len(p) for p in pred]
    pred = torch.cat(pred, dim=0)  # [\Sum_i N_i, 7]
    obj_conf, cls_conf, cls_idx = pred[:, 4], pred[:, 5], pred[:, 6]
    sel_mask = (filter_w_thresh(obj_conf, cls_idx, obj_thresh)) & \
        (filter_w_thresh(cls_conf, cls_idx, cls_thresh))
    if filter_bbox_fn is not None:
        new_bbox, keep_mask = filter_bbox_fn(pred[:, :4])
        sel_mask &= keep_mask
        pred[:, :4] = new_bbox
    label = pred[sel_mask]  # [N, 7]
    # also process the lst_lens so that we can split back later
    new_lst_lens = []
    for i in range(len(lst_lens)):
        start_idx, end_idx = sum(lst_lens[:i]), sum(lst_lens[:i + 1])
        keep_num = sel_mask[start_idx:end_idx].float().sum().item()
        new_lst_lens.append(int(keep_num))
    # the loaded GT `ObjectLabels` are in xywh (corner) format, [M, 7]
    #   7: [t, (x, y, w, h), cls_idx, cls_conf]
    # so here we also convert to corner-format (x, y, w, h)
    xywh = xyxy2xywh(label[:, :4], format_='corner', last4=True)  # [N, 4]
    t = torch.zeros_like(xywh[:, 0:1])  # fake a timestep, [N, 1]
    obj_conf, cls_conf, cls_idx = label[:, 4:5], label[:, 5:6], label[:, 6:7]
    label = torch.cat([t, xywh, cls_idx, cls_conf, obj_conf], dim=1)  # [N, 8]
    # back to list of tensors
    assert sum(new_lst_lens) == label.shape[0]
    label = torch.split(label, new_lst_lens, dim=0)
    label = [ObjectLabels(lbl, hw) for lbl in label]
    # label: `B`-len List[ObjectLabels]
    return list(label)


@temporal_wrapper
def merge_label(gt_label: List[Union[th.Tensor, None]],
                pseudo_label: List[Union[th.Tensor, None]]):
    """Merge GT labels with pseudo labels.

    For frames with GT, keep the original GT.
    For frames without GT, use the pseudo labels.
    """
    # gt_label: `B`-len list of bbox or None
    # pseudo_label: `B`-len list of bbox or None
    assert len(gt_label) == len(pseudo_label)
    gt_mask = [lbl is not None for lbl in gt_label]
    for i in range(len(gt_label)):
        if gt_label[i] is None:
            gt_label[i] = pseudo_label[i]
    return gt_label, gt_mask


def _evaluate_label_one_class(gt_bbox: th.Tensor,
                              pseudo_bbox: th.Tensor,
                              all_thresh: Tuple[float] = (0.25, 0.50, 0.75)):
    """Evaluate bbox belonging to the same class."""
    if gt_bbox is None or len(gt_bbox) == 0:
        return None
    metrics = [0.] * 2 * len(all_thresh)
    if len(pseudo_bbox) == 0:
        return metrics
    # compute iou, bbox is [M/N, 4 (x, y, w, h)]
    ious = bboxes_iou(gt_bbox, pseudo_bbox, xyxy=False)  # [M, N]
    # compute precision (false pos) and recall (missing det)
    for thresh_idx, thresh in enumerate(all_thresh):
        mask = (ious > thresh)
        ar = mask.any(dim=1).float().mean()
        ap = mask.any(dim=0).float().mean()
        metrics[thresh_idx] = ar
        metrics[thresh_idx + len(all_thresh)] = ap
    return metrics


@temporal_wrapper
def evaluate_label(gt_label: List[Union[ObjectLabels, None]],
                   pseudo_label: List[ObjectLabels],
                   pred_mask: np.ndarray,
                   num_cls: int,
                   prefix: str = '',
                   all_thresh: Tuple[float] = (0.25, 0.50, 0.75)):
    """Evaluate the quality of filtered pseudo labels."""
    # pred_mask: False --> skipped frames, don't count as missing detections
    assert len(gt_label) == len(pseudo_label) == len(pred_mask)
    all_metrics = [[] for _ in range(num_cls)]
    num_gt_insts = [[] for _ in range(num_cls)]
    num_pred_insts = [[] for _ in range(num_cls)]
    for gt, pseudo, is_pred in zip(gt_label, pseudo_label, pred_mask):
        if gt is None or len(gt) == 0 or not is_pred:
            continue
        all_gt = gt.get_labels_as_tensors()[:, :5]  # [M, 5]
        all_pseudo = pseudo.get_labels_as_tensors()[:, :5]  # [N, 5]
        # each bbox is [num_bbox, 5 (cls_idx, x, y, w, h)], center format
        for cls_idx in range(num_cls):
            gt_bbox = all_gt[all_gt[:, 0] == cls_idx, 1:]
            pseudo_bbox = all_pseudo[all_pseudo[:, 0] == cls_idx, 1:]
            if len(gt_bbox) == 0:
                continue
            metrics = _evaluate_label_one_class(
                gt_bbox, pseudo_bbox, all_thresh=all_thresh)
            # metrics: [2 * num_thresh], ar_1, ..., ar_n, ap_1, ..., ap_n
            all_metrics[cls_idx].append(metrics)
            num_gt_insts[cls_idx].append(len(gt_bbox))
            num_pred_insts[cls_idx].append(len(pseudo_bbox))
    # compute average for logging
    log_dict = {}
    label_map = get_labelmap(num_cls=num_cls)
    for cls_idx, (metrics, gt_insts, pred_insts) in \
            enumerate(zip(all_metrics, num_gt_insts, num_pred_insts)):
        # don't log if no GT
        if len(metrics) == 0:
            assert len(gt_insts) == 0
            continue
        cls_name = label_map[cls_idx]
        log_dict[f'num_{cls_name}'] = len(metrics)
        # AR, AP per class
        metrics = torch.tensor(metrics).mean(dim=0).cpu().numpy()
        for thresh_idx, thresh in enumerate(all_thresh):
            thresh = int(thresh * 100)
            ar, ap = metrics[thresh_idx], metrics[thresh_idx + len(all_thresh)]
            log_dict[f'{prefix}teacher_AR@{thresh}_{cls_name}'] = ar
            log_dict[f'{prefix}teacher_AP@{thresh}_{cls_name}'] = ap
        # number of instances per class
        log_dict[f'{prefix}gt_num_{cls_name}'] = np.array(gt_insts).mean()
        log_dict[f'{prefix}pred_num_{cls_name}'] = np.array(pred_insts).mean()
    return log_dict


def _get_scores_ious_one_class(gt_label: ObjectLabels,
                               pseudo_label: ObjectLabels,
                               cls_idx: int = None):
    """Collect the IoUs and scores for this one class."""
    gt_bbox = gt_label.get_labels_as_tensors()[:, :5]  # [M, 5]
    pseudo_bbox = pseudo_label.get_labels_as_tensors()[:, :5]  # [N, 5]
    # each bbox is [num_bbox, 5 (cls_idx, x, y, w, h)], center format
    if cls_idx is not None:
        gt_bbox = gt_bbox[gt_bbox[:, 0] == cls_idx]
        pse_mask = (pseudo_bbox[:, 0] == cls_idx)
        pseudo_bbox = pseudo_bbox[pse_mask]
    if len(gt_bbox) == 0:
        return None, None, None
    # compute iou, bbox is [M/N, 4 (x, y, w, h)]
    ious = bboxes_iou(gt_bbox[:, 1:], pseudo_bbox[:, 1:], xyxy=False)  # [M, N]
    # for each predicted bbox, find the best matching GT bbox
    best_pred_ious = ious.max(dim=0)[0]  # [N]
    pred_cls_conf = pseudo_label.class_confidence  # [N]
    pred_obj_conf = pseudo_label.objectness  # [N]
    if cls_idx is not None:
        pred_cls_conf = pred_cls_conf[pse_mask]
        pred_obj_conf = pred_obj_conf[pse_mask]
    return best_pred_ious, pred_cls_conf, pred_obj_conf


@temporal_wrapper
def get_scores_ious(gt_label: List[Union[ObjectLabels, None]],
                    pseudo_label: List[ObjectLabels],
                    pred_mask: np.ndarray,
                    num_cls: int,
                    prefix: str = ''):
    """Collect the predicted bbox's IoU with GTs, and their cls/obj_scores."""
    # pred_mask: False --> skipped frames, don't count as missing detections
    assert len(gt_label) == len(pseudo_label) == len(pred_mask)
    all_ious = [[] for _ in range(num_cls + 1)]
    all_cls_scores = [[] for _ in range(num_cls + 1)]
    all_obj_scores = [[] for _ in range(num_cls + 1)]
    for gt, pseudo, is_pred in zip(gt_label, pseudo_label, pred_mask):
        if gt is None or len(gt) == 0 or not is_pred:
            continue
        for i, cls_idx in enumerate(list(range(num_cls)) + [None]):
            ious, cls_scores, obj_scores = _get_scores_ious_one_class(
                gt_label=gt, pseudo_label=pseudo, cls_idx=cls_idx)
            if ious is None:
                continue
            all_ious[i].append(ious)
            all_cls_scores[i].append(cls_scores)
            all_obj_scores[i].append(obj_scores)
    all_ious = [th_cat(ious).cpu().numpy().tolist() for ious in all_ious]
    all_cls_scores = [
        th_cat(scores).cpu().numpy().tolist() for scores in all_cls_scores
    ]
    all_obj_scores = [
        th_cat(scores).cpu().numpy().tolist() for scores in all_obj_scores
    ]
    log_dict = {}
    label_map = get_labelmap(num_cls=num_cls)
    for cls_idx, (ious, cls_scores, obj_scores) in \
            enumerate(zip(all_ious, all_cls_scores, all_obj_scores)):
        if cls_idx == num_cls:
            cls_name = 'all'
        else:
            cls_name = label_map[cls_idx]
        log_dict[f'{prefix}true_ious_{cls_name}'] = ious
        log_dict[f'{prefix}cls_scores_{cls_name}'] = cls_scores
        log_dict[f'{prefix}obj_scores_{cls_name}'] = obj_scores
    return log_dict


def weak2strong_label(weak_label: List[List[ObjectLabels]],
                      weak_aug: BatchAugmState, strong_aug: BatchAugmState):
    """Convert the weak aug labels (teacher) to strong aug labels (student).

    Args:
        weak_label: `L`-len list of `B`-len sub-lists each is `ObjectLabels`
        weak/strong_aug: {
            'h_flip': {'active': `B`-len list of bool},
            'zoom_out': {
                'active': [`B`-len list of bool],
                'x0': [`B`-len list of int],
                'y0': [`B`-len list of int],
                'factor': [`B`-len list of float],
            },
            'zoom_in': {
                'active': [`B`-len list of bool],
                'x0': [`B`-len list of int],
                'y0': [`B`-len list of int],
                'factor': [`B`-len list of float],
            },
            'rotation': {
                'active': [`B`-len list of bool],
                'angle_deg': [`B`-len list of float],
            },
        }
    """
    # the same aug is applied to all timesteps
    L = len(weak_label)
    # we first reverse the weak_aug
    # TODO: assume that we only do horizontal flip in weak_aug
    assert not any(weak_aug['rotation']['active'])
    assert not any(weak_aug['zoom_in']['active'])
    assert not any(weak_aug['zoom_out']['active'])
    h_flip_active = [weak_aug['h_flip']['active']] * L
    # True --> the data is flipped in weak_aug --> need to flip back here
    weak_label = LabelAugmentorGenX.flip_lr(weak_label, h_flip_active)

    # then apply the same strong_aug
    """ we should follow the same order as in the Augmentor
        if self.augm_state.apply_h_flip:
            data_dict = self._flip(data_dict, type_='h')
        if self.augm_state.rotation.active:
            data_dict = self._rotate(data_dict)
        if self.augm_state.zoom_in.active:
            data_dict = self._zoom_in_and_rescale(data_dict=data_dict)
        if self.augm_state.zoom_out.active:
            assert not self.augm_state.zoom_in.active
            data_dict = self._zoom_out_and_rescale(data_dict=data_dict)
    """
    h_flip_active = [strong_aug['h_flip']['active']] * L
    strong_label = LabelAugmentorGenX.flip_lr(weak_label, h_flip_active)

    rot_active = [strong_aug['rotation']['active']] * L
    rot_angle_deg = [strong_aug['rotation']['angle_deg']] * L
    strong_label = LabelAugmentorGenX.rotate(strong_label, rot_active,
                                             rot_angle_deg)

    zoom_in_active = [strong_aug['zoom_in']['active']] * L
    zoom_in_x0 = [strong_aug['zoom_in']['x0']] * L
    zoom_in_y0 = [strong_aug['zoom_in']['y0']] * L
    zoom_in_factor = [strong_aug['zoom_in']['factor']] * L
    strong_label = LabelAugmentorGenX.zoom_in(strong_label, zoom_in_active,
                                              zoom_in_x0, zoom_in_y0,
                                              zoom_in_factor)

    zoom_out_active = [strong_aug['zoom_out']['active']] * L
    zoom_out_x0 = [strong_aug['zoom_out']['x0']] * L
    zoom_out_y0 = [strong_aug['zoom_out']['y0']] * L
    zoom_out_factor = [strong_aug['zoom_out']['factor']] * L
    strong_label = LabelAugmentorGenX.zoom_out(strong_label, zoom_out_active,
                                               zoom_out_x0, zoom_out_y0,
                                               zoom_out_factor)

    return strong_label


@torch.no_grad()
def ema_model_update(model: th.nn.Module,
                     ema_model: th.nn.Module,
                     global_step: int,
                     alpha: float = 0.999):
    # Use the true average until the exponential average is more correct
    # Follow SoftTeacher and 3DIoUMatch
    alpha = min(1. - 1. / (global_step + 1.), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1. - alpha)


@torch.no_grad()
def model_update(student_model: th.nn.Module,
                 teacher_model: th.nn.Module,
                 global_step: int,
                 method: str = 'ema',
                 alpha: float = 0.999):
    method = method.lower()
    if method == 'ema':
        ema_model_update(
            model=student_model,
            ema_model=teacher_model,
            global_step=global_step,
            alpha=alpha)
    elif 'every-' in method:  # 'every-10000'
        num_step = int(method.split('-')[-1])
        if (global_step + 1) % num_step == 0:
            teacher_model.load_state_dict(student_model.state_dict())
            print(f'Update teacher model at step {global_step + 1}')
    else:
        raise NotImplementedError(f'Unknown model update method: {method}')
