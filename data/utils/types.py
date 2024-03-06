from enum import auto, Enum

try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum
from typing import Dict, List, Optional, Tuple, Union, Any

import torch as th

from data.genx_utils.labels import ObjectLabels, SparselyBatchedObjectLabels
# from data.utils.augmentor import AugmentationState  # avoid circular import


class DataType(Enum):
    PATH = auto()  # 'path/to/dst/test/17-06-97_12-14-33_244500000_304500000'
    EV_IDX = auto()  # index of the loaded ev_repr in the entire sequence
    EV_REPR = auto()
    FLOW = auto()
    IMAGE = auto()
    OBJLABELS = auto()
    OBJLABELS_SEQ = auto()
    SKIPPED_OBJLABELS_SEQ = auto()  # GT labels that are skipped in SSOD
    IS_PADDED_MASK = auto()
    IS_FIRST_SAMPLE = auto()
    IS_LAST_SAMPLE = auto()
    IS_REVERSED = auto()  # whether the sequence is in reverse order
    TOKEN_MASK = auto()
    PRED_MASK = auto()  # if the teacher do prediction at this frame
    GT_MASK = auto()  # which labels are from GT, others are pseudo labels
    PRED_PROBS = auto()  # obj/cls_probs predicted by the teacher model
    AUGM_STATE = auto()  # augmentation state


class DatasetType(Enum):
    GEN1 = auto()
    GEN4 = auto()


class DatasetMode(Enum):
    TRAIN = auto()
    VALIDATION = auto()
    TESTING = auto()


class DatasetSamplingMode(StrEnum):
    RANDOM = 'random'
    STREAM = 'stream'
    MIXED = 'mixed'


class ObjDetOutput(Enum):
    LABELS_PROPH = auto()
    PRED_PROPH = auto()
    EV_REPR = auto()
    SKIP_VIZ = auto()


LoaderDataDictGenX = Dict[DataType, Union[List[th.Tensor], ObjectLabels,
                                          SparselyBatchedObjectLabels,
                                          # AugmentationState,
                                          List[bool], ]]

LstmState = Optional[Tuple[th.Tensor, th.Tensor]]
LstmStates = List[LstmState]

FeatureMap = th.Tensor
BackboneFeatures = Dict[int, th.Tensor]
BatchAugmState = Dict[str, Dict[str, List[Any]]]
