import copy
from typing import Tuple

import numpy as np


def greedy_matching(cost_matrix: np.ndarray, idx_lst: np.ndarray,
                    thresh: float = 0.0) -> np.ndarray:
    cost_matrix = copy.deepcopy(cost_matrix)
    matched_indices = []
    assert len(idx_lst) == cost_matrix.shape[0]
    for i in idx_lst:
        if cost_matrix[i].max() < thresh:
            continue
        j = np.argmax(cost_matrix[i])
        cost_matrix[:, j] = -np.inf
        matched_indices.append([i, j])
    return np.array(matched_indices)  # (N, 2)


def iou_batch_xywh(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """
    Computes IOU between two bboxes in the form [x,y,w,h,(cls_id)]
      both bbox are in shape (N, 4/5) where N is the number of bboxes
    If class_id is provided, take it into account by ignoring the IOU between
      bboxes of different classes.
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0] - bb_test[..., 2] / 2.,
                     bb_gt[..., 0] - bb_gt[..., 2] / 2.)
    yy1 = np.maximum(bb_test[..., 1] - bb_test[..., 3] / 2.,
                     bb_gt[..., 1] - bb_gt[..., 3] / 2.)
    xx2 = np.minimum(bb_test[..., 0] + bb_test[..., 2] / 2.,
                     bb_gt[..., 0] + bb_gt[..., 2] / 2.)
    yy2 = np.minimum(bb_test[..., 1] + bb_test[..., 3] / 2.,
                     bb_gt[..., 1] + bb_gt[..., 3] / 2.)
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / (
        bb_test[..., 2] * bb_test[..., 3] + bb_gt[..., 2] * bb_gt[..., 3] - wh)

    # set IoU of between different class objects to 0
    if bb_test.shape[-1] == 5 and bb_gt.shape[-1] == 5:
        o[bb_gt[..., 4] != bb_test[..., 4]] = 0.

    return o


def xyxy2xywh(bbox: np.ndarray) -> np.ndarray:
    """
    Takes a bbox in the form [x1,y1,x2,y2] and returns a new bbox in the form
      [x,y,w,h] where x,y is the center and w,h are the width and height
    """
    x1, y1, x2, y2 = bbox
    bbox = [(x1 + x2) / 2., (y1 + y2) / 2., x2 - x1, y2 - y1]
    return np.array(bbox)


def xywh2xyxy(bbox: np.ndarray) -> np.ndarray:
    """
    Takes a bounding box in the form [x,y,w,h] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    x, y, w, h = bbox
    bbox = [x - w / 2., y - h / 2., x + w / 2., y + h / 2.]
    return np.array(bbox)


def clamp_bbox(bbox: np.ndarray,
               img_hw: Tuple[int, int],
               format_: str = 'xyxy') -> np.ndarray:
    """
    Clamp bbox to image boundaries.
    """
    # bbox: (4,) or (1, 4) or (4, 1)
    bbox_shape = bbox.shape
    bbox = bbox.squeeze()  # to (4,)
    H, W = img_hw
    assert format_ in ['xyxy', 'xywh']
    if format_ == 'xywh':
        bbox = xywh2xyxy(bbox)
    x1_, y1_, x2_, y2_ = bbox
    x1 = np.clip(x1_, 0., W - 1.)
    x2 = np.clip(x2_, 0., W - 1.)
    y1 = np.clip(y1_, 0., H - 1.)
    y2 = np.clip(y2_, 0., H - 1.)
    bbox = np.array([x1, y1, x2, y2])
    clamp_top, clamp_down = (y1 != y1_), (y2 != y2_)
    clamp_left, clamp_right = (x1 != x1_), (x2 != x2_)
    if format_ == 'xywh':
        bbox = xyxy2xywh(bbox)
    bbox = bbox.reshape(bbox_shape)
    return bbox, clamp_top, clamp_down, clamp_left, clamp_right
