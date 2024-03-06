import os
import copy
from pathlib import Path
from typing import Any, List, Optional

import h5py
import numpy as np
import torch
from torchdata.datapipes.map import MapDataPipe

from data.genx_utils.labels import ObjectLabelFactory, ObjectLabels
from data.utils.spatial import get_original_hw
from data.utils.types import DatasetType, DataType, LoaderDataDictGenX


def get_event_representation_dir(path: Path, ev_representation_name: str) -> Path:
    ev_repr_dir = path / 'event_representations_v2' / ev_representation_name
    assert ev_repr_dir.is_dir(), f'{ev_repr_dir}'
    return ev_repr_dir


def get_objframe_idx_2_repr_idx(path: Path, ev_representation_name: str) -> np.ndarray:
    ev_repr_dir = get_event_representation_dir(path=path, ev_representation_name=ev_representation_name)
    objframe_idx_2_repr_idx = np.load(str(ev_repr_dir / 'objframe_idx_2_repr_idx.npy'))
    return objframe_idx_2_repr_idx


class SequenceBase(MapDataPipe):
    """A wrapper of an event sequence.
    Given a frame_idx, can load its bbox labels, event repr, etc.

    Structure example of a sequence:
    .
    ├── event_representations_v2
    │ └── ev_representation_name
    │     ├── event_representations.h5  # keep the same
    │     ├── objframe_idx_2_repr_idx.npy  # labeled frame idx to ev_repr idx
    │     └── timestamps_us.npy  # useless
    └── labels_v2
        ├── labels.npz  # see below
        └── timestamps_us.npy  # useless
    labels.npz contains `labels` and `objframe_idx_2_label_idx`:
    - `labels`: a np.array with many fields (see `labels.py`), each is (N,).
                It contains all bbox of all labeled frames in one sequence.
                To know which labeled frame contained what bboxes, see below.
    - `objframe_idx_2_label_idx`: (num_labeled_frames,), o_idx2l_idx[i] is the
                start_idx of bbox of the i-th labeled frame. o_idx2l_idx[i+1]
                is the stop_idx of that frame. Use [start:stop] to get it.
    """

    def __init__(self,
                 path: Path,
                 ev_representation_name: str,
                 sequence_length: int,
                 dataset_type: DatasetType,
                 downsample_by_factor_2: bool,
                 only_load_end_labels: bool,
                 objframe_idx: List[int] = None,
                 data_ratio: float = -1.0,
                 tflip_offset: int = -1):
        assert sequence_length >= 1
        assert path.is_dir()
        assert dataset_type in {DatasetType.GEN1, DatasetType.GEN4}, \
            f'{dataset_type} not implemented'

        self.path = str(path)
        self.only_load_end_labels = only_load_end_labels

        # event representations
        ev_repr_dir = get_event_representation_dir(
            path=path, ev_representation_name=ev_representation_name)
        ds_str = '_ds2_nearest' if downsample_by_factor_2 else ''
        self.ev_repr_file = ev_repr_dir / f'event_representations{ds_str}.h5'
        while self.ev_repr_file.is_symlink():
            self.ev_repr_file = Path(os.readlink(str(self.ev_repr_file)))
        assert self.ev_repr_file.exists(), f'{str(self.ev_repr_file)=}'
        self.seq_len = sequence_length
        with h5py.File(str(self.ev_repr_file), 'r') as h5f:
            self.num_ev_repr = h5f['data'].shape[0]

        # labels
        height, width = get_original_hw(dataset_type)
        label_data = np.load(str(path / 'labels_v2' / 'labels.npz'))
        objframe_idx_2_label_idx = label_data['objframe_idx_2_label_idx']
        labels = label_data['labels']
        label_factory = ObjectLabelFactory.from_structured_array(
            object_labels=labels,
            objframe_idx_2_label_idx=objframe_idx_2_label_idx,
            input_size_hw=(height, width),
            downsample_factor=2 if downsample_by_factor_2 else None)
        self.label_factory = label_factory
        # `label_factory`: an iterable variable containing all bbox labels of
        #   this seq. `label_factory[i]` returns the bbox of the i-th frame,
        #   represented as a `ObjectLabels` instance (like a [N, 7] tensor).
        # It supports basic rigid transformations for data_aug.

        # get the ev_repr index of all labeled frames
        self.objframe_idx_2_repr_idx = get_objframe_idx_2_repr_idx(
            path=path, ev_representation_name=ev_representation_name)
        # if an idx is in this dict's keys, we know that frame has labels
        self.repr_idx_2_objframe_idx = dict(
            zip(self.objframe_idx_2_repr_idx,
                range(len(self.objframe_idx_2_repr_idx))))

        # sub-sample the labeled frames
        self.real_all_objframe_idx, self.all_objframe_idx, self.skip_label = \
            self._subsample_labels(data_ratio, objframe_idx)

        # Useful for weighted sampler that is based on label statistics:
        self._only_load_labels = False

        # for time_flip data augmentation
        self.time_flip = False
        self.time_flip_label_offset = tflip_offset

    def _subsample_labels(self, data_ratio: float, objframe_idx: List[int]):
        """Potentially sub-sample the labeled frames."""
        all_objframe_idx = sorted(list(self.repr_idx_2_objframe_idx.values()))
        skip_label = (0. < data_ratio < 1.) or (objframe_idx is not None)
        if not skip_label:
            return tuple(all_objframe_idx), tuple(all_objframe_idx), False

        real_all_objframe_idx = copy.deepcopy(all_objframe_idx)
        try:
            assert data_ratio <= 0.5, f'Invalid sparse {data_ratio=}'
        except AssertionError:
            # only when we want to generate pseudo labels for a skipped seq
            assert isinstance(objframe_idx, list) and len(objframe_idx) == 0
        if objframe_idx is None:  # construct objframe_idx by sub-sampling
            skip = round(1. / data_ratio)  # skipping frames uniformly
            all_objframe_idx = all_objframe_idx[::skip]
            # it's fine we just sample from the first label even though its
            #   corresponding ev_repr_idx may be very small (e.g. 1)
            # this is because we will adjust it in `SequenceForRandomAccess` by
            #   offseting with at least `self.seq_len`
            if len(all_objframe_idx) == 0:  # we need at least 1 labeled frame
                all_objframe_idx = [real_all_objframe_idx[-1]]
                print('Warning: only 1 labeled frame in this sequence.')
        else:
            try:
                assert len(objframe_idx) > 0, 'No subsample label idx provided'
            except AssertionError:
                assert data_ratio == -1
            all_objframe_idx = objframe_idx
        return tuple(real_all_objframe_idx), tuple(all_objframe_idx), True

    def _load_range_labels(self, start_idx, end_idx):
        """Load labels of frames in [start_idx, end_idx)."""
        # when time_flip, ev_repr[i] should load label[i-1]
        # for example, if in forward mode, ev_repr[1] is 1.0, 1.1, ..., 2.0s,
        #   and label[1] is the bbox at the moment of 2.0s.
        # however, if we load ev_repr[1] in the backward mode, the model
        #   actually sees events 2.0, 1.9, ..., 1.0s, so we should load the
        #   bbox at the moment of 1.0s, which is label[0].
        # TODO: this might hold on Gen4 as the RGB frame is not well-aligned
        # TODO: with the events. The RGB bbox are usually more ahead in time.
        if self.time_flip:
            start_idx = start_idx + self.time_flip_label_offset
            end_idx = end_idx + self.time_flip_label_offset
        labels, skipped_labels = list(), list()
        for repr_idx in range(start_idx, end_idx):
            label, valid = self._get_labels_from_repr_idx(repr_idx)
            if valid:  # has label
                labels.append(label)
                skipped_labels.append(None)
            else:
                if label is None:  # no label
                    labels.append(None)
                    skipped_labels.append(None)
                else:  # has label, but should be skipped
                    labels.append(None)
                    skipped_labels.append(label)
        return labels, skipped_labels

    def _get_labels_from_repr_idx(self, repr_idx: int) -> Optional[ObjectLabels]:
        """Return the bbox label of the frame at the end of this event repr. If not an exact match, return None."""
        idx = self.repr_idx_2_objframe_idx.get(repr_idx, None)
        if idx is None:  # no label for this frame
            return None, False
        if idx not in self.all_objframe_idx:  # has label, but skipped
            return self.label_factory[idx], False
        return self.label_factory[idx], True  # has label, and not skipped

    def _get_event_repr_torch(self, start_idx: int, end_idx: int) -> List[torch.Tensor]:
        """Load a list of event repr in torch.float32 tensor, each [C, H, W]."""
        assert end_idx > start_idx
        with h5py.File(str(self.ev_repr_file), 'r') as h5f:
            ev_repr = h5f['data'][start_idx:end_idx]
        ev_repr = torch.from_numpy(ev_repr)  # [num_repr (T), Bins (C), H, W]
        if ev_repr.dtype != torch.uint8:
            ev_repr = ev_repr.float()
        ev_repr = [x for x in ev_repr]
        return ev_repr  # a `T`-len list of [C, H, W] tensors

    def _rand_another(self, idx=None) -> Any:
        """Randomly load a batch using another index."""
        if idx is None:
            idx = np.random.randint(0, len(self))
        return self[idx]

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    @staticmethod
    def time_flip_data(data: LoaderDataDictGenX) -> LoaderDataDictGenX:
        """Flip the order of the sequence."""
        assert data[DataType.IS_REVERSED]
        # DataType.PATH: unchanged
        # DataType.EV_IDX: reversed
        data[DataType.EV_IDX].reverse()
        # DataType.EV_REPR: reversed
        data[DataType.EV_REPR] = [
            x.flip(0) for x in data[DataType.EV_REPR][::-1]
        ]
        # DataType.OBJLABELS_SEQ: reversed
        data[DataType.OBJLABELS_SEQ].time_flip_()
        # DataType.IS_FIRST_SAMPLE: unchanged
        # DataType.IS_LAST_SAMPLE: unchanged
        # DataType.IS_PADDED_MASK: reversed
        data[DataType.IS_PADDED_MASK].reverse()
        # DataType.SKIPPED_OBJLABELS_SEQ: reversed
        if DataType.SKIPPED_OBJLABELS_SEQ in data:
            data[DataType.SKIPPED_OBJLABELS_SEQ].time_flip_()
        return data

    def is_only_loading_labels(self) -> bool:
        return self._only_load_labels

    def only_load_labels(self):
        self._only_load_labels = True

    def load_everything(self):
        self._only_load_labels = False
