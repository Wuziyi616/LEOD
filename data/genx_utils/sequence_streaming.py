from pathlib import Path
from typing import List, Optional, Union, Tuple

import h5py
import numpy as np
import torch
from omegaconf import DictConfig
from torchdata.datapipes.iter import IterDataPipe

from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.genx_utils.sequence_base import SequenceBase, get_objframe_idx_2_repr_idx
from data.utils.augmentor import RandomSpatialAugmentorGenX
from data.utils.types import DataType, DatasetType, LoaderDataDictGenX
from utils.timers import TimerDummy as Timer
from data.utils.ssod_augmentor import SSODAugmentorGenX


def _scalar_as_1d_array(scalar: Union[int, float]):
    return np.atleast_1d(scalar)


def _get_ev_repr_range_indices(indices: np.ndarray, max_len: int) -> List[Tuple[int, int]]:
    """
    Computes a list of index ranges based on the input array of indices and a maximum length.
    The index ranges are computed such that the difference between consecutive indices
    should not exceed the maximum length (max_len).

    Parameters:
    -----------
    indices : np.ndarray
        A NumPy array of indices, where the indices are sorted in ascending order.
    max_len : int
        The maximum allowed length between consecutive indices.

    Returns:
    --------
    out : List[Tuple[int, int]]
        A list of tuples, where each tuple contains two integers representing the start and
        stop indices of the range.
    """
    meta_indices_stop = np.flatnonzero(np.diff(indices) > max_len)  # get the idx where the diff exceeds max_len

    meta_indices_start = np.concatenate((np.atleast_1d(0), meta_indices_stop + 1))
    meta_indices_stop = np.concatenate((meta_indices_stop, np.atleast_1d(len(indices) - 1)))

    out = list()
    for meta_idx_start, meta_idx_stop in zip(meta_indices_start, meta_indices_stop):
        idx_start = max(indices[meta_idx_start] - max_len + 1, 0)
        idx_stop = indices[meta_idx_stop] + 1
        out.append((idx_start, idx_stop))
    return out


class SequenceForIter(SequenceBase):
    """A wrapper of a seq of events, each data sample is a sub-seq of `length`.

    Compared to `SequenceForRandomAccess`, where all the loaded event seq are
      treated as initial seq (so reset model's RNN states), here, we might load
      consecutive seq, and let the model do recurrent learning.
    """

    def __init__(self,
                 path: Path,
                 ev_representation_name: str,
                 sequence_length: int,  # num_frames in each data sample
                 dataset_type: DatasetType,
                 downsample_by_factor_2: bool,
                 range_indices: Optional[Tuple[int, int]] = None,
                 objframe_idx: List[int] = None,
                 data_ratio: float = -1.0,
                 tflip_offset: int = -1,
                 start_from_zero: bool = False):
        if (0. < data_ratio < 1.):
            assert len(objframe_idx) > 0, \
                'Should specify `objframe_idx` for streaming data'

        super().__init__(path=path,
                         ev_representation_name=ev_representation_name,
                         sequence_length=sequence_length,
                         dataset_type=dataset_type,
                         downsample_by_factor_2=downsample_by_factor_2,
                         only_load_end_labels=False,
                         objframe_idx=objframe_idx,
                         data_ratio=data_ratio,
                         tflip_offset=tflip_offset)

        if len(self.objframe_idx_2_repr_idx) == 0 and not start_from_zero:
            self.length = 0  # this seq will be skipped
            return
        num_ev_repr = self.num_ev_repr
        if range_indices is None:  # start from `length` reprs prior to the first labeled frame
            if start_from_zero:
                repr_idx_start = 0
            else:
                repr_idx_start = max(self.objframe_idx_2_repr_idx[0] - sequence_length + 1, 0)
            repr_idx_stop = num_ev_repr
        else:
            repr_idx_start, repr_idx_stop = range_indices
        # Set start idx such that the first label is no further than the last timestamp of the first sample sub-sequence
        if start_from_zero:
            min_start_repr_idx = 0
        else:
            min_start_repr_idx = max(self.objframe_idx_2_repr_idx[0] - sequence_length + 1, 0)
        assert 0 <= min_start_repr_idx <= repr_idx_start < repr_idx_stop <= num_ev_repr, \
            f'{min_start_repr_idx=}, {repr_idx_start=}, {repr_idx_stop=}, {num_ev_repr=}, {path=}'

        self.start_indices = list(range(repr_idx_start, repr_idx_stop, sequence_length))
        self.stop_indices = self.start_indices[1:] + [repr_idx_stop]
        self.length = len(self.start_indices)

        self._padding_representation = None

        # for time_flip data augmentation
        time_flip_start_indices = list(range(repr_idx_stop - 1, repr_idx_start - 1, -sequence_length))
        time_flip_stop_indices = time_flip_start_indices[1:] + [repr_idx_start - 1]
        # e.g. repr_idx_stop == 21, repr_idx_start == 0, sequence_length == 10
        # then we have start: [20, 10, 0], stop: [10, 0, -1]
        # need to reverse as start_idx should be smaller than stop_idx
        self.time_flip_start_indices = [i + 1 for i in time_flip_stop_indices]
        self.time_flip_stop_indices = [i + 1 for i in time_flip_start_indices]
        # now start: [11, 1, 0], stop: [21, 11, 1]

    @staticmethod
    def get_sequences_with_guaranteed_labels(
            path: Path,
            ev_representation_name: str,
            sequence_length: int,
            dataset_type: DatasetType,
            downsample_by_factor_2: bool,
            tflip_offset: int = -1) -> List['SequenceForIter']:
        """Generate sequences such that we do always have labels **within**
          (not necessarily at the end!) each sample of the sequence.
        This is required for training such that we are guaranteed to always
          have labels in the training step.
        However, for validation we don't require this as we will run the
          model (with RNN) over the entire sequence.
        """
        objframe_idx_2_repr_idx = get_objframe_idx_2_repr_idx(
            path=path, ev_representation_name=ev_representation_name)
        if len(objframe_idx_2_repr_idx) == 0:
            return list()
        # max diff for repr idx is sequence length
        range_indices_list = _get_ev_repr_range_indices(
            indices=objframe_idx_2_repr_idx, max_len=sequence_length)
        sequence_list = list()
        for range_indices in range_indices_list:
            sequence_list.append(
                SequenceForIter(path=path,
                                ev_representation_name=ev_representation_name,
                                sequence_length=sequence_length,
                                dataset_type=dataset_type,
                                downsample_by_factor_2=downsample_by_factor_2,
                                range_indices=range_indices,
                                tflip_offset=tflip_offset)
            )
        return sequence_list

    @property
    def padding_representation(self) -> torch.Tensor:
        if self._padding_representation is None:
            ev_repr = self._get_event_repr_torch(start_idx=0, end_idx=1)[0]
            self._padding_representation = torch.zeros_like(ev_repr)
        return self._padding_representation

    def get_fully_padded_sample(self) -> LoaderDataDictGenX:
        ev_repr = [self.padding_representation] * self.seq_len
        sparse_labels = SparselyBatchedObjectLabels(
            sparse_object_labels_batch=[None] * self.seq_len)
        out = {
            DataType.PATH: '',
            DataType.EV_IDX: [-1] * self.seq_len,
            DataType.EV_REPR: ev_repr,
            DataType.OBJLABELS_SEQ: sparse_labels,
            DataType.SKIPPED_OBJLABELS_SEQ: sparse_labels,
            DataType.IS_FIRST_SAMPLE: False,
            DataType.IS_LAST_SAMPLE: False,
            DataType.IS_REVERSED: False,
            DataType.IS_PADDED_MASK: [True] * self.seq_len,
        }
        return out

    def __len__(self):
        return self.length

    def __getitem__(self, index: int) -> LoaderDataDictGenX:
        """Load a sub-seq of event data + labels."""
        # in time-flip mode, the event seq's temporal order is reversed
        if self.time_flip:
            start_idx = self.time_flip_start_indices[index]
            end_idx = self.time_flip_stop_indices[index]
        else:
            start_idx = self.start_indices[index]
            end_idx = self.stop_indices[index]

        # sequence info ###
        sample_len = end_idx - start_idx
        assert self.seq_len >= sample_len > 0, \
            f'{self.seq_len=}, {sample_len=}, {start_idx=}, {end_idx=}, ' \
            f'\n{self.start_indices=}\n{self.stop_indices=}'

        ev_idx = list(range(start_idx, end_idx))
        is_first_sample = (index == 0)
        is_last_sample = (index == self.length - 1)
        is_padded_mask = [False] * sample_len
        ###################

        # labels ###
        labels, skipped_labels = self._load_range_labels(start_idx, end_idx)
        # a list of `ObjectLabels` ([n, 7] bbox) or `None`

        # apply padding (if necessary) ###
        if sample_len < self.seq_len:
            padding_len = self.seq_len - sample_len
            if self.time_flip:  # pad in front, will be reversed later
                ev_idx = [-1] * padding_len + ev_idx
                labels = [None] * padding_len + labels
                skipped_labels = [None] * padding_len + skipped_labels
                is_padded_mask = [True] * padding_len + is_padded_mask
            else:  # pad in back
                ev_idx.extend([-1] * padding_len)
                labels.extend([None] * padding_len)
                skipped_labels.extend([None] * padding_len)
                is_padded_mask.extend([True] * padding_len)
        ##################################

        if self._only_load_labels:
            sparse_labels = SparselyBatchedObjectLabels(
                sparse_object_labels_batch=labels)
            sparse_skipped_labels = SparselyBatchedObjectLabels(
                sparse_object_labels_batch=skipped_labels)
            ev_repr = [self.padding_representation] * self.seq_len
            return {
                DataType.PATH: self.path,  # str
                DataType.EV_IDX: ev_idx,  # List[int]
                DataType.EV_REPR: ev_repr,  # empty
                DataType.IS_FIRST_SAMPLE: is_first_sample,  # bool
                DataType.IS_LAST_SAMPLE: is_last_sample,  # bool
                DataType.IS_REVERSED: self.time_flip,  # bool
                DataType.IS_PADDED_MASK: is_padded_mask,  # List[bool]
                DataType.OBJLABELS_SEQ: sparse_labels,  # packed ObjectLabels
                DataType.SKIPPED_OBJLABELS_SEQ: sparse_skipped_labels,  # same
            }
        ############

        # event representations ###
        with Timer(timer_name='read ev reprs'):
            ev_repr = self._get_event_repr_torch(
                start_idx=start_idx, end_idx=end_idx)
        if sample_len < self.seq_len:
            if self.time_flip:  # pad in front, will be reversed later
                ev_repr = [self.padding_representation] * padding_len + ev_repr
            else:  # pad in back
                ev_repr.extend([self.padding_representation] * padding_len)
        assert len(labels) == len(skipped_labels) == len(ev_repr) == len(ev_idx)
        # a list of [C, H, W], torch.Tensor
        ###########################

        # convert labels to sparse labels for datapipes and dataloader
        sparse_labels = SparselyBatchedObjectLabels(
            sparse_object_labels_batch=labels)
        sparse_skipped_labels = SparselyBatchedObjectLabels(
            sparse_object_labels_batch=skipped_labels)

        out = {
            DataType.PATH: self.path,  # str
            DataType.EV_IDX: ev_idx,  # [L] list of int
            DataType.EV_REPR: ev_repr,  # `L`-len list of [C, H, W], torch.Tensor
            DataType.OBJLABELS_SEQ: sparse_labels,  # `L` ObjectLabels packed
            DataType.SKIPPED_OBJLABELS_SEQ: sparse_skipped_labels,  # same
            DataType.IS_FIRST_SAMPLE: is_first_sample,  # bool
            DataType.IS_LAST_SAMPLE: is_last_sample,  # bool
            DataType.IS_REVERSED: self.time_flip,  # bool
            DataType.IS_PADDED_MASK: is_padded_mask,  # [L] list, True: padded
        }
        if self.time_flip:
            out = self.time_flip_data(data=out)
        return out


class RandAugmentIterDataPipe(IterDataPipe):
    """RandAugment pipeline for event data."""

    def __init__(self, source_dp: IterDataPipe, dataset_config: DictConfig):
        super().__init__()
        self.source_dp = source_dp
        # an instance of `SequenceForIter` converted to IterDataPipe
        # see https://github.com/pytorch/data/blob/main/torchdata/datapipes/map/util/converter.py
        # `source_dp.datapipe` is the original `SequenceForIter` instance

        resolution_hw = tuple(dataset_config.resolution_hw)
        assert len(resolution_hw) == 2
        ds_by_factor_2 = dataset_config.downsample_by_factor_2
        if ds_by_factor_2:
            resolution_hw = tuple(x // 2 for x in resolution_hw)

        augm_config = dataset_config.data_augmentation
        self.spatial_augmentor = self._create_augmentor(
            dataset_hw=resolution_hw, augm_config=augm_config)

    def _create_augmentor(self, dataset_hw: Tuple[int, int],
                          augm_config: DictConfig):
        """Create augmentor by using RandomSpatialAugmentorGenX."""
        return RandomSpatialAugmentorGenX(
            dataset_hw=dataset_hw,
            automatic_randomization=False,
            augm_config=augm_config.stream)

    def __iter__(self):
        # only randomize once and keep the same aug for the entire event seq
        self.spatial_augmentor.randomize_augmentation()
        # have to apply it when loading the event seq
        if self.spatial_augmentor.augm_state.apply_t_flip:
            self.source_dp.datapipe.time_flip = True
            self.spatial_augmentor.augm_state.apply_t_flip = False
        else:
            self.source_dp.datapipe.time_flip = False
        for x in self.source_dp:
            yield self.spatial_augmentor(x)


class SSODRandAugmentIterDataPipe(RandAugmentIterDataPipe):
    """Similar as RandAugmentIterDataPipe, but we use SSODAugmentorGenX for data aug."""

    def _create_augmentor(self, dataset_hw: Tuple[int, int],
                          augm_config: DictConfig):
        """Create augmentor by using SSODAugmentorGenX."""
        return SSODAugmentorGenX(
            dataset_hw=dataset_hw,
            automatic_randomization=False,
            augm_config=augm_config.stream)
