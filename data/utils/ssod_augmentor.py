import copy
from typing import Tuple

from omegaconf import DictConfig

from data.genx_utils.labels import ObjectLabels
from data.utils.types import LoaderDataDictGenX
from .augmentor import RandomSpatialAugmentorGenX


def _all_is_type(lst, t):
    """Check if all elements in a list are of a certain type."""
    return all(isinstance(x, t) for x in lst)


def _all_same_len(lst):
    """Check if all elements in a list have the same length."""
    return len(set(len(x) for x in lst)) == 1


class SSODAugmentorGenX:
    """Data augmentor for SSOD training, with weak and strong augmentation."""

    def __init__(self, dataset_hw: Tuple[int, int],
                 automatic_randomization: bool, augm_config: DictConfig):

        self.dataset_hw = dataset_hw
        self.automatic_randomization = automatic_randomization

        # strong aug: as default
        #   horizontal flip + zoom augmentation (zoom in or zoom out)
        self.strong_augm_config = copy.deepcopy(augm_config)
        self.strong_aug = RandomSpatialAugmentorGenX(
            self.dataset_hw, self.automatic_randomization,
            self.strong_augm_config)

        # weak aug: only do horizontal flip
        self.weak_augm_config = copy.deepcopy(augm_config)
        self.weak_augm_config.prob_hflip = 0.5
        self.weak_augm_config.rotate.prob = 0  # no rotation
        self.weak_augm_config.zoom.prob = 0  # no zoom-in/out
        self.weak_aug = RandomSpatialAugmentorGenX(
            self.dataset_hw, self.automatic_randomization,
            self.weak_augm_config)

    def __call__(self, data_dict: LoaderDataDictGenX) -> LoaderDataDictGenX:
        """Apply weak and strong augmentation to get two versions of output."""
        data_dict_copy = copy.deepcopy(data_dict)  # avoid in-place data aug

        data_dict_weak = self.weak_aug(data_dict)
        data_dict_strong = self.strong_aug(data_dict_copy)

        # combine data_dict for weak and strong augmentation
        data_dict_weak = {f'weak_{k}': v for k, v in data_dict_weak.items()}
        data_dict = {**data_dict_weak, **data_dict_strong}
        return data_dict

    def randomize_augmentation(self):
        """Calls `randomize_augmentation()` for both strong and weak aug."""
        self.strong_aug.randomize_augmentation()
        self.weak_aug.randomize_augmentation()


class LabelAugmentorGenX:
    """Augmentor helper class. Only for ObjectLabels."""

    @staticmethod
    def zoom_out(data: ObjectLabels, active: bool, x0: int, y0: int,
                 factor: float) -> ObjectLabels:
        """Apply zoom out to one or a list of data."""
        if isinstance(data, list):
            assert _all_is_type([active, x0, y0, factor], list)
            assert _all_same_len([data, active, x0, y0, factor])
            return [
                LabelAugmentorGenX.zoom_out(d, a, x, y, f)
                for d, a, x, y, f in zip(data, active, x0, y0, factor)
            ]

        assert isinstance(data, ObjectLabels)

        if not active or factor == 1:
            return data
        data.zoom_out_and_rescale_(
            zoom_coordinates_x0y0=(x0, y0), zoom_out_factor=factor)
        return data

    @staticmethod
    def zoom_in(data: ObjectLabels, active: bool, x0: int, y0: int,
                factor: float) -> ObjectLabels:
        """Apply zoom in to one or a list of data."""
        if isinstance(data, list):
            assert _all_is_type([active, x0, y0, factor], list)
            assert _all_same_len([data, active, x0, y0, factor])
            return [
                LabelAugmentorGenX.zoom_in(d, a, x, y, f)
                for d, a, x, y, f in zip(data, active, x0, y0, factor)
            ]

        assert isinstance(data, ObjectLabels)

        if not active or factor == 1:
            return data
        data.zoom_in_and_rescale_(
            zoom_coordinates_x0y0=(x0, y0), zoom_in_factor=factor)
        return data

    @staticmethod
    def rotate(data: ObjectLabels, active: bool,
               angle_deg: float) -> ObjectLabels:
        """Apply rotation to one or a list of data."""
        if isinstance(data, list):
            assert _all_is_type([active, angle_deg], list)
            assert _all_same_len([data, active, angle_deg])
            return [
                LabelAugmentorGenX.rotate(d, a, angle)
                for d, a, angle in zip(data, active, angle_deg)
            ]

        assert isinstance(data, ObjectLabels)

        if not active or angle_deg == 0:
            return data
        data.rotate_(angle_deg=angle_deg)
        return data

    @staticmethod
    def flip_lr(data: ObjectLabels, active: bool) -> ObjectLabels:
        """Apply horizontal flip to one or a list of data."""
        if isinstance(data, list):
            assert isinstance(active, list)
            assert len(data) == len(active)
            return [
                LabelAugmentorGenX.flip_lr(d, a) for d, a in zip(data, active)
            ]

        assert isinstance(data, ObjectLabels)

        if not active:
            return data
        data.flip_lr_()
        return data
