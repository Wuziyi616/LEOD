import numpy as np
from pathlib import Path
from typing import Any, List

from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.genx_utils.sequence_base import SequenceBase
from data.utils.types import DataType, DatasetType, LoaderDataDictGenX
from utils.timers import TimerDummy as Timer


class SequenceForRandomAccess(SequenceBase):
    """Load labeled frames and `L` event reprs before it.
    Each frame is only loaded once.
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
        do_offset = (objframe_idx is None)

        super().__init__(path=path,
                         ev_representation_name=ev_representation_name,
                         sequence_length=sequence_length,
                         dataset_type=dataset_type,
                         downsample_by_factor_2=downsample_by_factor_2,
                         only_load_end_labels=only_load_end_labels,
                         objframe_idx=objframe_idx,
                         data_ratio=data_ratio,
                         tflip_offset=tflip_offset)
        assert not self.only_load_end_labels

        # find the first frame that has enough history event repr
        self.start_idx_offset = None
        for objframe_idx, repr_idx in enumerate(self.objframe_idx_2_repr_idx):
            if repr_idx - self.seq_len + 1 >= 0:
                # We can fit the sequence length to the label
                self.start_idx_offset = objframe_idx
                break
        if self.start_idx_offset is None:
            self.length = 0  # this seq will be skipped
            return
        # may need to adjust the selected `self.all_objframe_idx`
        elif self.skip_label and do_offset and (self.start_idx_offset > 0):
            self.all_objframe_idx = tuple([
                idx + self.start_idx_offset for idx in self.all_objframe_idx
                if idx + self.start_idx_offset in self.real_all_objframe_idx
            ])
        self.same_last_idx = \
            (self.all_objframe_idx[-1] == self.real_all_objframe_idx[-1])

        # `len(self.label_factory)` is the number of frames with labels
        self.length = len(self.label_factory) - self.start_idx_offset
        assert len(self.label_factory) == len(self.objframe_idx_2_repr_idx)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> LoaderDataDictGenX:
        """Load a frame with its bbox labels, and `L` event reprs before it."""
        if self.time_flip:
            corrected_idx = index  # don't offset (see below for why)
            labels_repr_idx = self.objframe_idx_2_repr_idx[corrected_idx]  # get ev_repr idx of this frame
            # there is no ev_repr after the true last labeled frame
            # since we do offset here, we won't have ev_reprs of enough length
            if corrected_idx == self.real_all_objframe_idx[-1]:
                return self._rand_another(idx=corrected_idx)
            labels_repr_idx -= self.time_flip_label_offset  # compensate offset
            # if we follow the original indexing, the labeled frame will be the
            #   first frame of the sequence, which makes learning challenging
            # instead, we want to put it as late as possible
            end_idx = min(self.num_ev_repr, labels_repr_idx + self.seq_len)
        else:  # make the labeled frame the last frame of the sequence
            corrected_idx = index + self.start_idx_offset  # offset to get enough history information
            labels_repr_idx = self.objframe_idx_2_repr_idx[corrected_idx]  # get ev_repr idx of this frame
            end_idx = labels_repr_idx + 1
        start_idx = end_idx - self.seq_len
        assert_msg = f'{self.ev_repr_file=}, {self.start_idx_offset=}, {start_idx=}, {end_idx=}'
        assert start_idx >= 0, assert_msg

        labels, skipped_labels = self._load_range_labels(start_idx, end_idx)
        # a list of `ObjectLabels` ([n, 7] bbox) or `None`
        # the loaded seq should have at least one labels
        if all(lbl is None for lbl in labels):
            return self._rand_another()
        # wrap with `SparselyBatchedObjectLabels` for collating and batching
        sparse_labels = SparselyBatchedObjectLabels(
            sparse_object_labels_batch=labels)
        sparse_skipped_labels = SparselyBatchedObjectLabels(
            sparse_object_labels_batch=skipped_labels)
        out = {
            DataType.OBJLABELS_SEQ: sparse_labels,
            DataType.SKIPPED_OBJLABELS_SEQ: sparse_skipped_labels,
        }
        if self._only_load_labels:
            return out

        with Timer(timer_name='read ev reprs'):
            ev_repr = self._get_event_repr_torch(
                start_idx=start_idx, end_idx=end_idx)
        ev_idx = list(range(start_idx, end_idx))
        assert len(labels) == len(skipped_labels) == len(ev_repr) == len(ev_idx)

        out.update({
            DataType.PATH: self.path,  # str
            DataType.EV_IDX: ev_idx,  # [L] list of int
            DataType.EV_REPR: ev_repr,  # `L`-len list of [C, H, W], torch.Tensor
            DataType.IS_FIRST_SAMPLE: True,  # always reset the RNN
            DataType.IS_LAST_SAMPLE: False,  # dummy
            DataType.IS_REVERSED: self.time_flip,  # bool
            DataType.IS_PADDED_MASK: [False] * len(ev_repr),
        })
        if self.time_flip:
            out = self.time_flip_data(data=out)
        return out

    def _rand_another(self, idx=None) -> Any:
        """2 cases:
        - not skipping labels: only happens when `time_flip` is True
          simply choose another frame (except the last one)
        - skipping labels: happens when `time_flip` is True or False
          need to find a labeled frame, then manually offset the idx
        """
        if not self.skip_label:
            assert self.time_flip, 'only happens when `time_flip` is True'
            assert idx == self.real_all_objframe_idx[-1], \
                'only happens when trying to load the last labeled frame'
            # shouldn't choose the last idx
            idx = np.random.choice(len(self) - 1, 1)[0]
            return self[idx]

        # need to find frames with labels
        if self.time_flip and self.same_last_idx:
            # skip the last labeled frame as there is no ev_repr after it
            all_objframe_idx = self.all_objframe_idx[:-1]
        else:
            all_objframe_idx = self.all_objframe_idx
        idx = np.random.choice(all_objframe_idx, 1)[0]
        if not self.time_flip:
            # we will offset in `self.__getitem__()`, reverse it here
            idx -= self.start_idx_offset
        return self[idx]
