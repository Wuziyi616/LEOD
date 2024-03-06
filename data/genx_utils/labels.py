from __future__ import annotations

from typing import List, Tuple, Union, Optional

import copy
import math
import numpy as np
import torch as th
from einops import rearrange
from torch.nn.functional import pad

BBOX_DTYPE = np.dtype({
    'names': ['t', 'x', 'y', 'w', 'h', 'class_id', 'class_confidence', 'objectness'],
    'formats': ['<i8', '<f4', '<f4', '<f4', '<f4', '<u4', '<f4', '<f4'],
    'offsets': [0, 8, 12, 16, 20, 24, 28, 32], 'itemsize': 40
})


class ObjectLabelBase:
    """Class that represents N bbox labels in shape [N, num_fields (7)].

    Similar to torch.Tensor, has dtype, device, etc. properties, with bbox
        attributes e.g. x,y,w,h,class_id.
    **The bbox format is corner! I.e. x,y are top-left corner coords.**
    """

    _str2idx = {
        't': 0,
        'x': 1,
        'y': 2,
        'w': 3,
        'h': 4,
        'class_id': 5,
        'class_confidence': 6,
        'objectness': 7,
    }

    def __init__(self,
                 object_labels: Union[th.Tensor, np.ndarray],
                 input_size_hw: Tuple[int, int]):
        assert isinstance(object_labels, (th.Tensor, np.ndarray))
        assert 'float' in str(object_labels.dtype)
        assert len(object_labels.shape) == 2
        assert object_labels.shape[-1] == len(self._str2idx)
        assert isinstance(input_size_hw, tuple)
        assert len(input_size_hw) == 2

        self.object_labels = object_labels  # [N, 8]
        self._input_size_hw = input_size_hw
        self._is_numpy = isinstance(object_labels, np.ndarray)

    def clamp_to_frame_(self):
        ht, wd = self.input_size_hw
        x0 = th.clamp(self.x, min=0, max=wd - 1)
        y0 = th.clamp(self.y, min=0, max=ht - 1)
        x1 = th.clamp(self.x + self.w, min=0, max=wd - 1)
        y1 = th.clamp(self.y + self.h, min=0, max=ht - 1)
        w = x1 - x0
        h = y1 - y0
        assert th.all(w > 0)
        assert th.all(h > 0)
        self.x = x0  # corner instead of center!
        self.y = y0
        self.w = w
        self.h = h

    def remove_flat_labels_(self):
        keep = (self.w > 0) & (self.h > 0)
        self.object_labels = self.object_labels[keep]

    def _assert_not_numpy(self):
        assert not self._is_numpy, "Labels have been converted numpy. \
        Numpy is not supported for the intended operations."

    def to(self, *args, **kwargs):
        # This function executes torch.to on self tensors and returns self.
        self._assert_not_numpy()
        # This will be used by Pytorch Lightning to transfer to the relevant device
        self.object_labels = self.object_labels.to(*args, **kwargs)
        return self

    def numpy_(self) -> None:
        """In place conversion to numpy (detach + to cpu + to numpy)."""
        if self._is_numpy:
            return
        self._is_numpy = True
        self.object_labels = self.object_labels.detach().cpu().numpy()

    def torch_(self) -> None:
        """In place conversion to torch (from numpy)."""
        if not self._is_numpy:
            return
        self._is_numpy = False
        self.object_labels = th.from_numpy(self.object_labels)

    @property
    def input_size_hw(self) -> Tuple[int, int]:
        return self._input_size_hw

    @input_size_hw.setter
    def input_size_hw(self, height_width: Tuple[int, int]):
        assert isinstance(height_width, tuple)
        assert len(height_width) == 2
        assert height_width[0] > 0
        assert height_width[1] > 0
        self._input_size_hw = height_width

    @classmethod
    def keys(cls) -> List[str]:
        return list(cls._str2idx.keys())

    def get(self, request: str):
        assert request in self._str2idx
        return self.object_labels[:, self._str2idx[request]]

    @property
    def t(self):
        return self.object_labels[:, self._str2idx['t']]

    @property
    def x(self):
        return self.object_labels[:, self._str2idx['x']]

    @x.setter
    def x(self, value: Union[th.Tensor, np.ndarray]):
        self.object_labels[:, self._str2idx['x']] = value

    @property
    def y(self):
        return self.object_labels[:, self._str2idx['y']]

    @y.setter
    def y(self, value: Union[th.Tensor, np.ndarray]):
        self.object_labels[:, self._str2idx['y']] = value

    @property
    def w(self):
        return self.object_labels[:, self._str2idx['w']]

    @w.setter
    def w(self, value: Union[th.Tensor, np.ndarray]):
        self.object_labels[:, self._str2idx['w']] = value

    @property
    def h(self):
        return self.object_labels[:, self._str2idx['h']]

    @h.setter
    def h(self, value: Union[th.Tensor, np.ndarray]):
        self.object_labels[:, self._str2idx['h']] = value

    @property
    def class_id(self):
        return self.object_labels[:, self._str2idx['class_id']]

    @class_id.setter
    def class_id(self, value: Union[th.Tensor, np.ndarray]):
        self.object_labels[:, self._str2idx['class_id']] = value

    @property
    def class_confidence(self):
        return self.object_labels[:, self._str2idx['class_confidence']]

    @property
    def objectness(self):
        return self.object_labels[:, self._str2idx['objectness']]

    def is_gt_label(self) -> Union[th.Tensor, np.ndarray]:
        # GT labels have t > 0
        return ~(self.is_pseudo_label())

    def is_pseudo_label(self) -> Union[th.Tensor, np.ndarray]:
        # pseudo labels are those with t == 0
        return (self.t == 0)

    def is_ignore(self, ignore_label) -> Union[th.Tensor, np.ndarray]:
        return (self.class_id == ignore_label)

    @property
    def dtype(self):
        return self.object_labels.dtype

    @property
    def device(self):
        return self.object_labels.device


class ObjectLabelFactory(ObjectLabelBase):
    """A wrapper containing many frames and their bbox labels."""

    def __init__(self,
                 object_labels: th.Tensor,
                 objframe_idx_2_label_idx: th.Tensor,
                 input_size_hw: Tuple[int, int],
                 downsample_factor: Optional[float] = None):
        super().__init__(object_labels=object_labels, input_size_hw=input_size_hw)
        assert objframe_idx_2_label_idx.dtype == th.int64
        assert objframe_idx_2_label_idx.dim() == 1

        self.objframe_idx_2_label_idx = objframe_idx_2_label_idx
        self.downsample_factor = downsample_factor
        if self.downsample_factor is not None:
            assert self.downsample_factor > 1
        self.clamp_to_frame_()

    @staticmethod
    def from_structured_array(object_labels: np.ndarray,
                              objframe_idx_2_label_idx: np.ndarray,
                              input_size_hw: Tuple[int, int],
                              downsample_factor: Optional[float] = None) -> ObjectLabelFactory:
        np_labels = []
        for key in ObjectLabels._str2idx.keys():
            if key in object_labels.dtype.names:
                np_labels.append(object_labels[key].astype('float32'))
            else:
                assert key == 'objectness', f'invalid {key=}'
                np_labels.append(object_labels['class_confidence'].astype('float32'))
        np_labels = rearrange(np_labels, 'fields L -> L fields')  # [num_boxes, num_fields]
        torch_labels = th.from_numpy(np_labels)
        objframe_idx_2_label_idx = th.from_numpy(objframe_idx_2_label_idx.astype('int64'))  # [num_frames], the start/end frame_idx of labels
        # assert objframe_idx_2_label_idx.numel() == np.unique(object_labels['t']).size
        return ObjectLabelFactory(object_labels=torch_labels,
                                  objframe_idx_2_label_idx=objframe_idx_2_label_idx,
                                  input_size_hw=input_size_hw,
                                  downsample_factor=downsample_factor)

    def __len__(self):
        return len(self.objframe_idx_2_label_idx)  # number of frames

    def __getitem__(self, item: int) -> ObjectLabels:
        """Load the bbox labels for the item-th frame."""
        assert item >= 0
        length = len(self)
        assert length > 0
        assert item < length
        is_last_item = (item == length - 1)

        from_idx = self.objframe_idx_2_label_idx[item]
        to_idx = self.object_labels.shape[0] if is_last_item else self.objframe_idx_2_label_idx[item + 1]
        assert to_idx > from_idx
        object_labels = ObjectLabels(
            object_labels=self.object_labels[from_idx:to_idx].clone(),  # [num_boxes, num_fields]
            input_size_hw=self.input_size_hw)
        if self.downsample_factor is not None:
            object_labels.scale_(scaling_multiplier=1 / self.downsample_factor)
        return object_labels


class ObjectLabels(ObjectLabelBase):
    """More advanced bbox label class, with transformation support."""

    def __init__(self,
                 object_labels: th.Tensor,
                 input_size_hw: Tuple[int, int]):
        super().__init__(object_labels=object_labels, input_size_hw=input_size_hw)

    def __len__(self) -> int:
        return self.object_labels.shape[0]

    def __add__(self, other: ObjectLabels) -> ObjectLabels:
        """Add two ObjectLabelBase objects by concating."""
        assert isinstance(other, ObjectLabels), f'{type(other)=}'
        assert self.input_size_hw == other.input_size_hw, 'Wrong input size'
        assert self._str2idx == other._str2idx, 'Wrong label format'
        if self._is_numpy:
            obj_labels = np.concatenate([self.object_labels, other.object_labels], axis=0)
        else:
            obj_labels = th.cat([self.object_labels, other.object_labels], dim=0)
        return ObjectLabels(object_labels=obj_labels, input_size_hw=self.input_size_hw)

    def __eq__(self, other: ObjectLabels) -> bool:
        """Check if two ObjectLabelBase objects are equal."""
        if not isinstance(other, ObjectLabels) or \
                self.input_size_hw != other.input_size_hw or \
                self.object_labels.shape != other.object_labels.shape:
            return False
        # bbox is order-invariant
        for other_label in other.object_labels:
            is_eq = False
            for label in self.object_labels:
                if (label - other_label).abs().max() < 1e-3:
                    is_eq = True
                    break
            if not is_eq:
                return False
        return True

    def new_zeros(self) -> ObjectLabels:
        """Create an empty ObjectLabels."""
        zeros = th.zeros((0, len(self._str2idx))).type_as(self.object_labels)
        return ObjectLabels(
            object_labels=zeros, input_size_hw=self.input_size_hw)

    @staticmethod
    def from_structured_array(labels: np.ndarray,
                              input_size_hw: Tuple[int, int],
                              downsample_factor: float = None) -> ObjectLabels:
        np_labels = []
        for key in ObjectLabels._str2idx.keys():
            if key in labels.dtype.names:
                np_labels.append(labels[key].astype('float32'))
            else:
                assert key == 'objectness', f'invalid {key=}'
                np_labels.append(labels['class_confidence'].astype('float32'))
        np_labels = rearrange(np_labels, 'fields L -> L fields')  # [num_boxes, num_fields]
        torch_labels = th.from_numpy(np_labels)
        object_labels = ObjectLabels(object_labels=torch_labels, input_size_hw=input_size_hw)
        if downsample_factor is not None:
            object_labels.scale_(scaling_multiplier=1 / downsample_factor)
        return object_labels

    def to_structured_array(self) -> np.array:
        """Convert to BBOX_DTYPE np.array."""
        if not self._is_numpy:
            self.numpy_()
        labels = np.zeros((len(self),), dtype=BBOX_DTYPE)
        labels['t'] = np.asarray(self.t, dtype=BBOX_DTYPE['t'])
        labels['x'] = np.asarray(self.x, dtype=BBOX_DTYPE['x'])
        labels['y'] = np.asarray(self.y, dtype=BBOX_DTYPE['y'])
        labels['w'] = np.asarray(self.w, dtype=BBOX_DTYPE['w'])
        labels['h'] = np.asarray(self.h, dtype=BBOX_DTYPE['h'])
        labels['class_id'] = np.asarray(self.class_id, dtype=BBOX_DTYPE['class_id'])
        labels['class_confidence'] = np.asarray(self.class_confidence, dtype=BBOX_DTYPE['class_confidence'])
        labels['objectness'] = np.asarray(self.objectness, dtype=BBOX_DTYPE['objectness'])
        return copy.deepcopy(labels)

    def rotate_(self, angle_deg: float):
        if len(self) == 0:
            return
        # (x0,y0)---(x1,y0)   p00---p10
        #  |             |    |       |
        #  |             |    |       |
        # (x0,y1)---(x1,y1)   p01---p11
        p00 = th.stack((self.x, self.y), dim=1)
        p10 = th.stack((self.x + self.w, self.y), dim=1)
        p01 = th.stack((self.x, self.y + self.h), dim=1)
        p11 = th.stack((self.x + self.w, self.y + self.h), dim=1)
        # points: 4 x N x 2
        points = th.stack((p00, p10, p01, p11), dim=0)

        cx = self._input_size_hw[1] // 2
        cy = self._input_size_hw[0] // 2
        center = th.tensor([cx, cy], device=self.device)

        angle_rad = angle_deg / 180 * math.pi
        # counter-clockwise rotation
        rot_matrix = th.tensor([[math.cos(angle_rad), math.sin(angle_rad)],
                                [-math.sin(angle_rad), math.cos(angle_rad)]], device=self.device)

        points = points - center
        points = th.einsum('ij,pnj->pni', rot_matrix, points)
        points = points + center

        height, width = self.input_size_hw
        x0 = th.clamp(th.min(points[..., 0], dim=0)[0], min=0, max=width - 1)
        y0 = th.clamp(th.min(points[..., 1], dim=0)[0], min=0, max=height - 1)
        x1 = th.clamp(th.max(points[..., 0], dim=0)[0], min=0, max=width - 1)
        y1 = th.clamp(th.max(points[..., 1], dim=0)[0], min=0, max=height - 1)

        self.x = x0
        self.y = y0
        self.w = x1 - x0
        self.h = y1 - y0

        self.remove_flat_labels_()

        assert th.all(self.x >= 0)
        assert th.all(self.y >= 0)
        assert th.all(self.x + self.w <= self.input_size_hw[1] - 1)
        assert th.all(self.y + self.h <= self.input_size_hw[0] - 1)

    def zoom_in_and_rescale_(self, zoom_coordinates_x0y0: Tuple[int, int], zoom_in_factor: float):
        """
        1) Computes a new smaller canvas size: original canvas scaled by a factor of 1/zoom_in_factor (downscaling)
        2) Places the smaller canvas inside the original canvas at the top-left coordinates zoom_coordinates_x0y0
        3) Extract the smaller canvas and rescale it back to the original resolution
        """
        if len(self) == 0:
            return
        assert len(zoom_coordinates_x0y0) == 2
        assert zoom_in_factor >= 1
        if zoom_in_factor == 1:
            return
        z_x0, z_y0 = zoom_coordinates_x0y0
        h_orig, w_orig = self.input_size_hw
        assert 0 <= z_x0 <= w_orig - 1, f'{z_x0=} is larger than {w_orig-1=}'
        assert 0 <= z_y0 <= h_orig - 1, f'{z_y0=} is larger than {h_orig-1=}'
        zoom_window_h, zoom_window_w = tuple(x / zoom_in_factor for x in self.input_size_hw)
        z_x1 = min(z_x0 + zoom_window_w, w_orig - 1)
        assert z_x1 <= w_orig - 1, f'{z_x1=} is larger than {w_orig-1=}'
        z_y1 = min(z_y0 + zoom_window_h, h_orig - 1)
        assert z_y1 <= h_orig - 1, f'{z_y1=} is larger than {h_orig-1=}'

        x0 = th.clamp(self.x, min=z_x0, max=z_x1 - 1)
        y0 = th.clamp(self.y, min=z_y0, max=z_y1 - 1)

        x1 = th.clamp(self.x + self.w, min=z_x0, max=z_x1 - 1)
        y1 = th.clamp(self.y + self.h, min=z_y0, max=z_y1 - 1)

        self.x = x0 - z_x0
        self.y = y0 - z_y0
        self.w = x1 - x0
        self.h = y1 - y0
        self.input_size_hw = (zoom_window_h, zoom_window_w)

        self.remove_flat_labels_()

        self.scale_(scaling_multiplier=zoom_in_factor)

    def reverse_zoom_in_and_rescale_(self, zoom_coordinates_x0y0: Tuple[int, int], zoom_in_factor: float):
        """
        Reverse operation of zoom_in_and_rescale_.

        1) Rescales the original canvas to a smaller canvas by a factor of 1/zoom_in_factor (downscaling)
        2) Places the smaller canvas back at the top-left coordinates zoom_coordinates_x0y0 within the larger canvas 
        (the same size as the original canvas)
        """
        if len(self) == 0:
            return
        assert len(zoom_coordinates_x0y0) == 2
        assert zoom_in_factor >= 1
        if zoom_in_factor == 1:
            return
        z_x0, z_y0 = zoom_coordinates_x0y0
        h_orig, w_orig = self.input_size_hw

        # Rescale the larger canvas back to the size of the smaller canvas
        self.scale_(scaling_multiplier=1 / zoom_in_factor)

        # Adjust the bounding box coordinates to be relative to the original canvas
        self.x = self.x + z_x0
        self.y = self.y + z_y0

        self.input_size_hw = (h_orig, w_orig)

    def zoom_out_and_rescale_(self, zoom_coordinates_x0y0: Tuple[int, int], zoom_out_factor: float):
        """
        1) Scales the input by a factor of 1/zoom_out_factor (i.e. reduces the canvas size)
        2) Places the downscaled canvas into the original canvas at the top-left coordinates zoom_coordinates_x0y0
        """
        if len(self) == 0:
            return
        assert len(zoom_coordinates_x0y0) == 2
        assert zoom_out_factor >= 1
        if zoom_out_factor == 1:
            return

        h_orig, w_orig = self.input_size_hw
        self.scale_(scaling_multiplier=1 / zoom_out_factor)

        self.input_size_hw = (h_orig, w_orig)
        z_x0, z_y0 = zoom_coordinates_x0y0
        assert 0 <= z_x0 <= w_orig - 1, f'{z_x0=} is larger than {w_orig-1=}'
        assert 0 <= z_y0 <= h_orig - 1, f'{z_y0=} is larger than {h_orig-1=}'

        self.x = self.x + z_x0
        self.y = self.y + z_y0

    def reverse_zoom_out_and_rescale_(self, zoom_coordinates_x0y0: Tuple[int, int], zoom_out_factor: float):
        """
        Reverse operation of zoom_out_and_rescale_.

        1) Places the downscaled canvas back into the original position within the original canvas (offset by zoom_coordinates_x0y0)
        1) Rescales this canvas to its original size by scaling it back with the zoom_out_factor
        """
        if len(self) == 0:
            return
        assert len(zoom_coordinates_x0y0) == 2
        assert zoom_out_factor >= 1
        if zoom_out_factor == 1:
            return

        z_x0, z_y0 = zoom_coordinates_x0y0
        self.x = self.x - z_x0
        self.y = self.y - z_y0

        h_orig, w_orig = self.input_size_hw
        self.scale_(scaling_multiplier=zoom_out_factor)
        self.input_size_hw = (h_orig, w_orig)  # input size stays the same

        assert (self.x >= 0).all() and (self.x <= w_orig - 1).all()
        assert (self.y >= 0).all() and (self.y <= h_orig - 1).all()
        assert (self.x + self.w <= w_orig - 1).all()
        assert (self.y + self.h <= h_orig - 1).all()

    def scale_(self, scaling_multiplier: float):
        if len(self) == 0:
            return
        assert scaling_multiplier > 0
        if scaling_multiplier == 1:
            return
        img_ht, img_wd = self.input_size_hw
        new_img_ht = scaling_multiplier * img_ht
        new_img_wd = scaling_multiplier * img_wd
        self.input_size_hw = (new_img_ht, new_img_wd)
        x1 = th.clamp((self.x + self.w) * scaling_multiplier, max=new_img_wd - 1)
        y1 = th.clamp((self.y + self.h) * scaling_multiplier, max=new_img_ht - 1)
        self.x = self.x * scaling_multiplier
        self.y = self.y * scaling_multiplier

        self.w = x1 - self.x
        self.h = y1 - self.y

        self.remove_flat_labels_()

    def flip_lr_(self) -> None:
        if len(self) == 0:
            return
        self.x = self.input_size_hw[1] - 1 - self.x - self.w

    def reverse_flip_lr_(self) -> None:
        """Reverse operation of flip_lr_."""
        self.flip_lr_()

    def get_reverse(self):
        # reverse the order of self.object_labels
        object_labels = th.from_numpy(self.object_labels) if \
            self._is_numpy else self.object_labels
        return ObjectLabels(object_labels.flip(0), self.input_size_hw)

    def get_xywh(self, format_='center', add_class_id: bool = False):
        assert format_ in ['center', 'corner']
        x, y, w, h = self.x, self.y, self.w, self.h  # corner x,y
        if format_ == 'center':
            x, y = x + 0.5 * w, y + 0.5 * h  # center x,y
        bbox = [x, y, w, h]
        if add_class_id:
            bbox.append(self.class_id)
        if self._is_numpy:
            return np.stack(bbox, axis=-1)
        return th.stack(bbox, dim=-1)

    def get_xyxy(self, add_class_id: bool = False):
        x1, y1, w, h = self.x, self.y, self.w, self.h
        x2, y2 = x1 + w, y1 + h
        bbox = [x1, y1, x2, y2]
        if add_class_id:
            bbox.append(self.class_id)
        if self._is_numpy:
            return np.stack(bbox, axis=-1)
        return th.stack(bbox, dim=-1)

    def get_labels_as_tensors(self, format_: str = 'yolox') -> th.Tensor:
        """Returns bbox labels in shape [num_bbox, 5]."""
        # self._assert_not_numpy()
        if self._is_numpy:
            self.torch_()

        out = th.zeros((len(self), 7), dtype=th.float32, device=self.device)
        if len(self) == 0:
            return out
        if format_ == 'yolox':  # [cls_id, (xywh), obj_conf, cls_conf]
            out[:, 0] = self.class_id
            out[:, 1] = self.x + 0.5 * self.w  # corner to center
            out[:, 2] = self.y + 0.5 * self.h
            out[:, 3] = self.w
            out[:, 4] = self.h
            out[:, 5] = self.objectness
            out[:, 6] = self.class_confidence
            return out
        elif format_ == 'prophesee':  # xyxy, obj_conf, cls_conf, cls_id
            out[:, 0] = self.x
            out[:, 1] = self.y
            out[:, 2] = self.x + self.w
            out[:, 3] = self.y + self.h
            out[:, 4] = self.objectness
            out[:, 5] = self.class_confidence
            out[:, 6] = self.class_id
            return out
        else:
            raise NotImplementedError(f'Unknown format {format_}')

    @staticmethod
    def get_labels_as_batched_tensor(obj_label_list: List[ObjectLabels], format_: str = 'yolox') -> th.Tensor:
        """Returns a batch of bbox labels, [num_frames, num_bbox, 5].
        Each frame is padded to the max number of boxes.
        """
        num_object_frames = len(obj_label_list)
        assert num_object_frames > 0
        N = max([len(x) for x in obj_label_list])  # max num_labels per frame
        assert N > 0
        return ObjectLabels.pad_labels(obj_label_list, N=N, format_=format_)

    @staticmethod
    def pad_labels(obj_label_list: Union[List[ObjectLabels], List[th.Tensor]], N: int, format_: str = 'yolox') -> th.Tensor:
        """Pad the labels to length N and stack them for return."""
        if format_ == 'yolox':
            tensor_labels = []
            for labels in obj_label_list:  # ObjectLabels or torch.Tensor
                if isinstance(labels, ObjectLabels):
                    obj_labels_tensor = labels.get_labels_as_tensors(format_=format_)  # [num_bbox, 7]
                elif isinstance(labels, th.Tensor):
                    assert tuple(labels.shape) == (len(labels), 7)
                    obj_labels_tensor = labels  # [num_bbox, 7 (cls_id, xywh, obj_conf, cls_conf)]
                else:
                    raise NotImplementedError(f'Unknown type: {type(labels)}')
                num_to_pad = N - len(labels)
                padded_labels = pad(obj_labels_tensor, (0, 0, 0, num_to_pad), mode='constant', value=0)
                tensor_labels.append(padded_labels)
            tensor_labels = th.stack(tensors=tensor_labels, dim=0)
            return tensor_labels  # [num_frames, num_bbox, 7]
        else:
            raise NotImplementedError(f'Unknown format: {format_}')


class SparselyBatchedObjectLabels:
    """A wrapper for batching the data in dataloader."""

    def __init__(self, sparse_object_labels_batch: List[Optional[ObjectLabels]]):
        # Can contain None elements that indicate missing labels.
        for entry in sparse_object_labels_batch:
            assert isinstance(entry, ObjectLabels) or entry is None
        self.sparse_object_labels_batch = sparse_object_labels_batch
        self.set_empty_labels_to_none_()

    def __len__(self) -> int:
        return len(self.sparse_object_labels_batch)

    def __iter__(self):
        return iter(self.sparse_object_labels_batch)

    def __getitem__(self, item: int) -> Optional[ObjectLabels]:
        if item < 0 or item >= len(self):
            raise IndexError(f'Index ({item}) out of range (0, {len(self) - 1})')
        return self.sparse_object_labels_batch[item]

    def __add__(self, other: SparselyBatchedObjectLabels) -> SparselyBatchedObjectLabels:
        """List concatenation."""
        sparse_object_labels_batch = self.sparse_object_labels_batch + other.sparse_object_labels_batch
        return SparselyBatchedObjectLabels(sparse_object_labels_batch=sparse_object_labels_batch)

    def __eq__(self, other: SparselyBatchedObjectLabels) -> bool:
        if len(self) != len(other):
            return False
        for idx in range(len(self)):
            if self[idx] != other[idx]:
                return False
        return True

    def set_empty_labels_to_none_(self):
        for idx, obj_label in enumerate(self.sparse_object_labels_batch):
            if obj_label is not None and len(obj_label) == 0:
                self.sparse_object_labels_batch[idx] = None

    def set_non_gt_labels_to_none_(self):
        for idx, obj_label in enumerate(self.sparse_object_labels_batch):
            if obj_label is not None and obj_label.is_pseudo_label().all():
                self.sparse_object_labels_batch[idx] = None

    def is_empty(self) -> bool:
        """If no labels or all labels are None."""
        return len(self) == 0 or all([x is None for x in self.sparse_object_labels_batch])

    @property
    def input_size_hw(self) -> Optional[Union[Tuple[int, int], Tuple[float, float]]]:
        for obj_labels in self.sparse_object_labels_batch:
            if obj_labels is not None:
                return obj_labels.input_size_hw
        return None

    def zoom_in_and_rescale_(self, *args, **kwargs):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].zoom_in_and_rescale_(*args, **kwargs)
        # We may have deleted labels. If no labels are left, set the object to None
        self.set_empty_labels_to_none_()

    def reverse_zoom_in_and_rescale_(self, *args, **kwargs):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].reverse_zoom_in_and_rescale_(*args, **kwargs)
        # We may have deleted labels. If no labels are left, set the object to None
        self.set_empty_labels_to_none_()

    def zoom_out_and_rescale_(self, *args, **kwargs):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].zoom_out_and_rescale_(*args, **kwargs)

    def reverse_zoom_out_and_rescale_(self, *args, **kwargs):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].reverse_zoom_out_and_rescale_(*args, **kwargs)

    def rotate_(self, *args, **kwargs):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].rotate_(*args, **kwargs)

    def scale_(self, *args, **kwargs):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].scale_(*args, **kwargs)
        # We may have deleted labels. If no labels are left, set the object to None
        self.set_empty_labels_to_none_()

    def flip_lr_(self):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].flip_lr_()

    def reverse_flip_lr_(self):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].reverse_flip_lr_()

    def time_flip_(self):
        self.sparse_object_labels_batch.reverse()

    def to(self, *args, **kwargs):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].to(*args, **kwargs)
        return self

    def get_valid_labels_and_batch_indices(self, ignore: bool = False, ignore_label: int = None) -> \
            Tuple[List[ObjectLabels], List[int]]:
        """Return a list of valid bbox labels and their idx."""
        if ignore:
            assert ignore_label is not None, 'ignore_label must be provided'
        out, valid_indices = list(), list()
        for idx, label in enumerate(self.sparse_object_labels_batch):
            if label is not None:
                # don't take frames that only have bbox with `ignore_label`
                if ignore and label.is_ignore(ignore_label).all():
                    continue
                out.append(label)
                valid_indices.append(idx)
        return out, valid_indices

    def get_labels_padded(self, pad=None) -> Tuple[List[ObjectLabels], List[int]]:
        """Return a list of bbox labels or None if not provided."""
        return [label if label is not None else pad for label in self.sparse_object_labels_batch], \
            [idx for idx in range(len(self)) if self[idx] is not None]

    @staticmethod
    def transpose_list(list_of_sparsely_batched_object_labels: List[SparselyBatchedObjectLabels]) -> \
            List[SparselyBatchedObjectLabels]:
        """Similar to transpose this list of lists.

        Assume we have [Label([None, None, bbox, None]), Label([None, None, bbox, None]), Label([bbox, None, None, None])]
        This will return [Label([None, None, bbox]), Label([None, None, None]), Label([bbox, bbox, None]), Label([None, None, None])]

        This is used as collate_fn for the dataloader.
        So after batching, we will have a list (length == seq_len `L`) of `SparselyBatchedObjectLabels`,
            each contains a list (length == batch_size `B`) of `ObjectLabels` or None.
        """
        return [SparselyBatchedObjectLabels(list(labels_as_tuple)) for labels_as_tuple \
                in zip(*list_of_sparsely_batched_object_labels)]


if __name__ == '__main__':
    input_size = (240, 304)
    object_labels = ObjectLabels(th.tensor([[9.1000e+06, 1.9500e+02, 1.4000e+02, 5.2000e+01, 3.8000e+01, 0.0000e+00, 1.0000e+00, 1.0000e+00], [9.1000e+06, 1.9500e+02, 1.4000e+02, 5.2000e+01, 3.8000e+01, 0.0000e+00, 1.0000e+00, 1.0000e+00]]), input_size)
    zoom_coordinates_x0y0 = (42, 52)
    zoom_factor = 1.321398913860321
    bbox = SparselyBatchedObjectLabels([object_labels])

    # test reverse_zoom_out_and_rescale_
    bbox_copy = copy.deepcopy(bbox)
    bbox_copy.zoom_out_and_rescale_(zoom_coordinates_x0y0=zoom_coordinates_x0y0, zoom_out_factor=zoom_factor)
    bbox_copy.reverse_zoom_out_and_rescale_(zoom_coordinates_x0y0=zoom_coordinates_x0y0, zoom_out_factor=zoom_factor)
    assert th.allclose(bbox_copy[0].object_labels, bbox[0].object_labels)

    # test reverse_zoom_in_and_rescale_
    bbox_copy = copy.deepcopy(bbox)
    bbox_copy.zoom_in_and_rescale_(zoom_coordinates_x0y0=zoom_coordinates_x0y0, zoom_in_factor=zoom_factor)
    bbox_copy.reverse_zoom_in_and_rescale_(zoom_coordinates_x0y0=zoom_coordinates_x0y0, zoom_in_factor=zoom_factor)
    assert th.allclose(bbox_copy[0].object_labels, bbox[0].object_labels)

    # test reverse_flip_lr_
    bbox_copy = copy.deepcopy(bbox)
    bbox_copy.flip_lr_()
    bbox_copy.reverse_flip_lr_()
    assert th.equal(bbox_copy[0].object_labels, bbox[0].object_labels)
