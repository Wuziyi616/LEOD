from enum import Enum, auto
from typing import List, Optional, Union, Tuple, Dict, Any

import torch
import torch as th

from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.utils.types import BackboneFeatures, LstmStates, DatasetSamplingMode


class Mode(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


mode_2_string = {
    Mode.TRAIN: 'train',
    Mode.VAL: 'val',
    Mode.TEST: 'test',
}

WORKER_ID_KEY = 'worker_id'
DATA_KEY = 'data'


class BackboneFeatureSelector:
    """A container for backbone features in torch.Tensor."""

    def __init__(self):
        self.features = None  # a dict of {stage_id: list of [b, C, h, w]}
        self.reset()

    def reset(self):
        self.features = dict()

    def add_backbone_features(self,
                              backbone_features: BackboneFeatures,
                              selected_indices: Optional[List[int]] = None) -> None:
        # backbone_features: dict{stage_id: feats, [B, C, h, w]}
        # selected_indices: batch_idx of valid samples in this batch
        if selected_indices is not None:
            if len(selected_indices) == 0:
                return
        for k, v in backbone_features.items():
            if k not in self.features:
                self.features[k] = [v[selected_indices]] if selected_indices is not None else [v]
            else:
                self.features[k].append(v[selected_indices] if selected_indices is not None else v)

    def get_batched_backbone_features(self) -> Optional[BackboneFeatures]:
        if len(self.features) == 0:
            return None
        # {stage_id: [\Sum_i b_i, C, h, w]}
        return {k: th.cat(v, dim=0) for k, v in self.features.items()}

    def is_empty(self):
        return self.features is None or len(self.features) == 0


class EventReprSelector:
    """A container for event repr in torch.Tensor."""

    def __init__(self):
        self.repr_list = None  # a list of [C, H, W]
        self.reset()

    def reset(self):
        self.repr_list = list()

    def __len__(self):
        return len(self.repr_list)

    def add_ev_repr(self,
                    ev_repr: th.Tensor,
                    selected_indices: Optional[List[int]] = None) -> None:
        """Append `ev_repr[selected_indices]` to `self.repr_list`."""
        # ev_repr: [B, C, H, W]
        # selected_indices: [n] (idx of valid samples)
        if selected_indices is not None:
            assert len(selected_indices) > 0
        self.repr_list.extend(x[0] for x in ev_repr[selected_indices].split(1))  # each as a [C, H, W]

    def get_ev_repr_as_list(self,
                            start_idx: int = 0,
                            end_idx: Optional[int] = None) -> Optional[List[th.Tensor]]:
        if len(self) == 0:
            return None
        if end_idx is None:
            end_idx = len(self)
        assert start_idx < end_idx, f'{start_idx=}, {end_idx=}'
        return self.repr_list[start_idx:end_idx]


class RNNStates:
    """A container for RNN states, support partial update/reset.

    We maintain a dict of {worker_id: states} for each data worker.
    """

    def __init__(self):
        self.states = {}  # each state is (lstm_h, lstm_c), [B, C, h, w]

    def _has_states(self):
        return len(self.states) > 0

    @classmethod
    def recursive_detach(cls, inp: Union[th.Tensor, List, Tuple, Dict]):
        """Detach all."""
        if isinstance(inp, th.Tensor):
            return inp.detach()
        if isinstance(inp, list):
            return [cls.recursive_detach(x) for x in inp]
        if isinstance(inp, tuple):
            return tuple(cls.recursive_detach(x) for x in inp)
        if isinstance(inp, dict):
            return {k: cls.recursive_detach(v) for k, v in inp.items()}
        raise NotImplementedError

    @classmethod
    def recursive_reset(cls,
                        inp: Union[th.Tensor, List, Tuple, Dict],
                        indices_or_bool_tensor: Optional[Union[List[int], torch.Tensor]] = None):
        """Reset the hidden states of all/partial inputs (RNN) to 0."""
        if isinstance(inp, th.Tensor):
            assert inp.requires_grad is False, 'Not assumed here but should be the case.'
            if indices_or_bool_tensor is None:
                inp[:] = 0
            else:
                assert len(indices_or_bool_tensor) > 0
                inp[indices_or_bool_tensor] = 0
            return inp
        if isinstance(inp, list):
            return [cls.recursive_reset(x, indices_or_bool_tensor=indices_or_bool_tensor) for x in inp]
        if isinstance(inp, tuple):
            return tuple(cls.recursive_reset(x, indices_or_bool_tensor=indices_or_bool_tensor) for x in inp)
        if isinstance(inp, dict):
            return {k: cls.recursive_reset(v, indices_or_bool_tensor=indices_or_bool_tensor) for k, v in inp.items()}
        raise NotImplementedError

    def save_states_and_detach(self, worker_id: int, states: LstmStates) -> None:
        """Store the RNN states to this worker_id (random/streaming data)."""
        self.states[worker_id] = self.recursive_detach(states)

    def get_states(self, worker_id: int) -> Optional[LstmStates]:
        if not self._has_states():
            return None
        if worker_id not in self.states:
            return None
        return self.states[worker_id]

    def reset(self, worker_id: int, indices_or_bool_tensor: Optional[Union[List[int], torch.Tensor]] = None):
        if not self._has_states():
            return
        if worker_id in self.states:
            self.states[worker_id] = self.recursive_reset(
                self.states[worker_id], indices_or_bool_tensor=indices_or_bool_tensor)


class SeqLens:
    """Record how many timesteps we have seen for each sequence."""

    def __init__(self):
        self.lens = {}  # {worker_id: [B] torch.tensor of int}

    def _has_lens(self):
        return len(self.lens) > 0

    def update_lens(self, worker_id: int, lens: th.Tensor) -> None:
        """Update the lens by adding."""
        if worker_id not in self.lens:
            self.lens[worker_id] = lens
        else:
            self.lens[worker_id] += lens

    def get_lens(self, worker_id: int) -> Optional[th.Tensor]:
        if not self._has_lens():
            return None
        if worker_id not in self.lens:
            return None
        return self.lens[worker_id]

    def reset(self, worker_id: int, indices_or_bool_tensor: Optional[Union[List[int], torch.Tensor]] = None):
        if worker_id not in self.lens:
            self.lens[worker_id] = th.zeros(len(indices_or_bool_tensor)).long()
            return
        if worker_id in self.lens:
            if indices_or_bool_tensor is None:
                self.lens[worker_id] = th.zeros_like(self.lens[worker_id])
            else:
                assert len(indices_or_bool_tensor) > 0
                self.lens[worker_id][indices_or_bool_tensor] = 0


def mixed_collate_fn(x1: Union[th.Tensor, List[th.Tensor]], x2: Union[th.Tensor, List[th.Tensor]]):
    if isinstance(x1, th.Tensor):
        assert isinstance(x2, th.Tensor)
        return th.cat((x1, x2))
    # dataloader returns a list `L` of `SparselyBatchedObjectLabels` objects,
    #   each contains `batch_size` (bs_rand or bs_stream) labels or None
    # after concat, we get a list `L`, each contains `batch_size` labels or None
    # so the size of each element is fixed as `batch_size` now!
    if isinstance(x1, SparselyBatchedObjectLabels):
        assert isinstance(x2, SparselyBatchedObjectLabels)
        return x1 + x2
    if isinstance(x1, list):
        assert isinstance(x2, list)
        assert len(x1) == len(x2)
        if isinstance(x1[0], str):
            return x1 + x2
        return [mixed_collate_fn(x1=el_1, x2=el_2) for el_1, el_2 in zip(x1, x2)]
    if isinstance(x1, dict):  # augm_state: Dict[Dict[str: List]], <= 2 level
        assert isinstance(x2, dict)
        x = {}
        for k in x1.keys():
            if isinstance(x1[k], dict):
                x[k] = mixed_collate_fn(x1[k], x2[k])
            elif isinstance(x1[k], list):
                x[k] = x1[k] + x2[k]
            else:
                raise NotImplementedError(f'{type(x1[k])=}, {type(x2[k])=}')
        return x
    raise NotImplementedError(f'{type(x1)=}, {type(x2)=}')


def merge_mixed_batches(batch: Dict[str, Any]):
    if DATA_KEY in batch:
        return batch
    rnd_data = batch[DatasetSamplingMode.RANDOM][DATA_KEY]
    stream_batch = batch[DatasetSamplingMode.STREAM]
    # We only care about the worker id of the streaming dataloader because the states will be anyway reset for the
    # random dataloader batch.
    out = {WORKER_ID_KEY: stream_batch[WORKER_ID_KEY]}
    stream_data = stream_batch[DATA_KEY]
    assert rnd_data.keys() == stream_data.keys(), f'{rnd_data.keys()=}, {stream_data.keys()=}'
    data_out = dict()  # event reprs, bbox labels, etc.
    for key in rnd_data.keys():
        data_out[key] = mixed_collate_fn(stream_data[key], rnd_data[key])
    out.update({DATA_KEY: data_out})
    return out
