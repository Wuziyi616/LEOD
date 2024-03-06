import copy
from typing import Any, Callable, Dict, Optional, Type, Tuple, Union

import torch

from data.genx_utils.collate_from_pytorch import collate, default_collate_fn_map
from data.genx_utils.labels import ObjectLabels, SparselyBatchedObjectLabels
from data.utils.augmentor import AugmentationState
from modules.utils.detection import WORKER_ID_KEY, DATA_KEY


def collate_object_labels(batch, *, collate_fn_map: Optional[
        Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return batch


def collate_sparsely_batched_object_labels(batch, *, collate_fn_map: Optional[
        Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    return SparselyBatchedObjectLabels.transpose_list(batch)


def collate_augm_state(batch, *, collate_fn_map: Optional[
        Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
    """Collates a batch of AugmentationState objects into a dictionary where
    each field of AugmentationState maps to a list of values.

    Returns: {
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
    return AugmentationState.collate_augm_state(batch)


custom_collate_fn_map = copy.deepcopy(default_collate_fn_map)
custom_collate_fn_map[ObjectLabels] = collate_object_labels
custom_collate_fn_map[SparselyBatchedObjectLabels] = collate_sparsely_batched_object_labels
custom_collate_fn_map[AugmentationState] = collate_augm_state


def custom_collate(batch: Any):
    return collate(batch, collate_fn_map=custom_collate_fn_map)


def custom_collate_rnd(batch: Any):
    samples = batch
    # NOTE: We do not really need the worker id for map style datasets (rnd) but we still provide the id for consistency
    worker_info = torch.utils.data.get_worker_info()
    local_worker_id = 0 if worker_info is None else worker_info.id
    return {
        DATA_KEY: custom_collate(samples),
        WORKER_ID_KEY: local_worker_id,
    }


def custom_collate_streaming(batch: Any):
    """We assume that we receive a batch collected by a worker of our streaming datapipe
    """
    samples = batch[0]
    worker_id = batch[1]
    assert isinstance(worker_id, int)
    return {
        DATA_KEY: custom_collate(samples),
        WORKER_ID_KEY: worker_id,
    }
