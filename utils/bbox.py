from typing import Union, List, Tuple

import numpy as np

import torch
import torch as th

from data.genx_utils.labels import ObjectLabels


def np_th_stack(values: List[Union[np.ndarray, th.Tensor]], axis: int = 0):
    """Stack a list of numpy arrays or tensors."""
    if isinstance(values[0], np.ndarray):
        return np.stack(values, axis=axis)
    elif isinstance(values[0], th.Tensor):
        return torch.stack(values, dim=axis)
    else:
        raise ValueError(f'Unknown type {type(values[0])}')


def np_th_concat(values: List[Union[np.ndarray, th.Tensor]], axis: int = 0):
    """Concat a list of numpy arrays or tensors."""
    if isinstance(values[0], np.ndarray):
        return np.concatenate(values, axis=axis)
    elif isinstance(values[0], th.Tensor):
        return torch.cat(values, dim=axis)
    else:
        raise ValueError(f'Unknown type {type(values[0])}')


def get_bbox_coords(bbox: Union[np.ndarray, th.Tensor], last4: bool = None):
    """Get the 4 coords (xyxy/xywh) from a bbox array or tensor."""
    if isinstance(bbox, list):
        bbox = np_th_stack(bbox, axis=0)
    if last4 is None:  # infer from shape, buggy when bbox.shape == (4, 4)
        if bbox.shape[0] == 4:
            last4 = False
        elif bbox.shape[-1] == 4:
            last4 = True
        else:
            raise ValueError(f'Unknown shape {bbox.shape}')
    if last4:
        a, b, c, d = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
    else:
        a, b, c, d = bbox
    return (a, b, c, d), last4


def construct_bbox(abcd: Tuple[Union[np.ndarray, th.Tensor]], last4: bool):
    """Construct a bbox from 4 coords (xyxy/xywh)."""
    if last4:
        return np_th_stack(abcd, axis=-1)
    return np_th_stack(abcd, axis=0)


def xywh2xyxy(xywh: Union[np.ndarray, th.Tensor], format_: str = 'center', last4: bool = None):
    """Convert bounding box from xywh to xyxy format."""
    if isinstance(xywh, ObjectLabels):
        return xywh.get_xyxy()

    (x, y, w, h), last4 = get_bbox_coords(xywh, last4=last4)

    if format_ == 'center':
        x1, x2 = x - w / 2., x + w / 2.
        y1, y2 = y - h / 2., y + h / 2.
    elif format_ == 'corner':
        x1, x2 = x, x + w
        y1, y2 = y, y + h
    else:
        raise NotImplementedError(f'Unknown format {format_}')

    return construct_bbox((x1, y1, x2, y2), last4=last4)


def xyxy2xywh(xyxy: Union[np.ndarray, th.Tensor], format_: str = 'center', last4: bool = None):
    """Convert bounding box from xyxy to xywh format."""
    if isinstance(xyxy, ObjectLabels):
        return xyxy.get_xywh(format_=format_)

    (x1, y1, x2, y2), last4 = get_bbox_coords(xyxy, last4=last4)

    w, h = x2 - x1, y2 - y1
    if format_ == 'center':
        x, y = (x1 + x2) / 2., (y1 + y2) / 2.
    elif format_ == 'corner':
        x, y = x1, y1
    else:
        raise NotImplementedError(f'Unknown format {format_}')

    return construct_bbox((x, y, w, h), last4=last4)
