from collections import namedtuple, defaultdict
from collections.abc import Iterable
import os
import os.path as osp
from pathlib import Path
from typing import List, Tuple

import numpy as np
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm

from data.genx_utils.labels import SparselyBatchedObjectLabels
from data.genx_utils.sequence_rnd import SequenceForRandomAccess
from data.utils.augmentor import RandomSpatialAugmentorGenX
from data.utils.types import DatasetMode, LoaderDataDictGenX, DatasetType, DataType
from data.utils.ssod_augmentor import SSODAugmentorGenX
from utils.preprocessing import subsample_sequence

from nerv.utils import load_obj, dump_obj


class SequenceDataset(Dataset):
    """A wrapper for RandomAccessSequence. We perform data aug here."""

    def __init__(self,
                 path: Path,
                 dataset_mode: DatasetMode,
                 dataset_config: DictConfig,
                 label_list: List[int] = None):
        assert path.is_dir()

        # extract settings from config
        sequence_length = dataset_config.sequence_length
        assert isinstance(sequence_length, int)
        assert sequence_length > 0
        self.output_seq_len = sequence_length

        ev_representation_name = dataset_config.ev_repr_name
        downsample_by_factor_2 = dataset_config.downsample_by_factor_2
        only_load_end_labels = dataset_config.only_load_end_labels
        if dataset_config.name == 'gen1':
            dataset_type = DatasetType.GEN1
        elif dataset_config.name == 'gen4':
            dataset_type = DatasetType.GEN4
        else:
            raise NotImplementedError

        augm_config = dataset_config.data_augmentation
        tflip_offset = augm_config.tflip_offset

        # can load any labeled frame, and `L` event reprs before it
        sparse_ratio = dataset_config.ratio
        # print(f'Use sparse sequence label with ratio: {sparse_ratio:.3f}')
        self.sequence = SequenceForRandomAccess(
            path=path,
            ev_representation_name=ev_representation_name,
            sequence_length=sequence_length,
            dataset_type=dataset_type,
            downsample_by_factor_2=downsample_by_factor_2,
            only_load_end_labels=only_load_end_labels,
            objframe_idx=label_list,
            data_ratio=sparse_ratio,
            tflip_offset=tflip_offset)

        self.always_tflip = dataset_config.reverse_event_order
        self.spatial_augmentor = None
        if dataset_mode == DatasetMode.TRAIN:
            resolution_hw = tuple(dataset_config.resolution_hw)
            assert len(resolution_hw) == 2
            ds_by_factor_2 = dataset_config.downsample_by_factor_2
            if ds_by_factor_2:
                resolution_hw = tuple(x // 2 for x in resolution_hw)
            self.spatial_augmentor = self._create_augmentor(
                resolution_hw, augm_config)

    def _create_augmentor(self, dataset_hw: Tuple[int, int],
                          augm_config: DictConfig):
        """Create augmentor by using RandomSpatialAugmentorGenX."""
        return RandomSpatialAugmentorGenX(
            dataset_hw=dataset_hw,
            automatic_randomization=False,
            augm_config=augm_config.random)

    def only_load_labels(self):
        self.sequence.only_load_labels()

    def load_everything(self):
        self.sequence.load_everything()

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index: int) -> LoaderDataDictGenX:
        apply_aug = False
        if self.spatial_augmentor is not None and \
                not self.sequence.is_only_loading_labels():
            # manually call it every time, in order to apply time_flip
            self.spatial_augmentor.randomize_augmentation()
            apply_aug = True
            if self.spatial_augmentor.augm_state.apply_t_flip:
                if len(self.sequence.all_objframe_idx) == 1 and \
                        self.sequence.same_last_idx:
                    # cannot do time flip in this case, because in order to
                    # load the last labeled frame, we need to ev_repr after it
                    # which is not available in pre-processed event data
                    if self.always_tflip:
                        raise ValueError  # will do rand_another
                    self.sequence.time_flip = False
                else:
                    self.sequence.time_flip = True
                self.spatial_augmentor.augm_state.apply_t_flip = False
            else:
                self.sequence.time_flip = False

        if self.always_tflip:
            assert self.sequence.time_flip, 'Not applying time flip'
        item = self.sequence[index]
        if apply_aug:
            item = self.spatial_augmentor(item)

        return item


class SSODSequenceDataset(SequenceDataset):
    """Similar as SequenceDataset, but we also use the SSODAugmentorGenX for data aug."""

    def _create_augmentor(self, dataset_hw: Tuple[int, int], augm_config: DictConfig):
        """Create augmentor by using SSODAugmentorGenX."""
        return SSODAugmentorGenX(
            dataset_hw=dataset_hw,
            automatic_randomization=False,
            augm_config=augm_config.random)


class CustomConcatDataset(ConcatDataset):
    datasets: List[SequenceDataset]

    def __init__(self, datasets: Iterable[SequenceDataset]):
        super().__init__(datasets=datasets)

    def __getitem__(self, idx):
        try:
            return super().__getitem__(idx)
        except ValueError:
            return self._rand_another()

    def _rand_another(self):
        """Need to find frames with labels."""
        idx = np.random.randint(len(self))
        return self[idx]

    def only_load_labels(self):
        for idx, dataset in enumerate(self.datasets):
            self.datasets[idx].only_load_labels()

    def load_everything(self):
        for idx, dataset in enumerate(self.datasets):
            self.datasets[idx].load_everything()


def build_random_access_dataset(
        dataset_mode: DatasetMode,
        dataset_config: DictConfig) -> CustomConcatDataset:
    """Build a dataset similar to torch.utils.data.Dataset.
    Each training seq contains a label.
    """
    dataset_path = Path(dataset_config.path)
    assert dataset_path.is_dir(), f'{str(dataset_path)}'

    mode2str = {DatasetMode.TRAIN: 'train',
                DatasetMode.VALIDATION: 'val',
                DatasetMode.TESTING: 'test'}

    assert dataset_mode == DatasetMode.TRAIN, 'Only use random_seq in training'
    split_path = dataset_path / mode2str[dataset_mode]
    while split_path.is_symlink():
        split_path = Path(os.readlink(str(split_path)))
    assert split_path.is_dir()
    # sub-sample event sequences, but keep the labeling frequency (SSOD)
    seq_dirs = subsample_sequence(split_path, dataset_config.train_ratio)

    # sub-sample labeling frequency (WSOD)
    # load sub-sampled label list from file (actually a Dict[path, List[int]])
    sparse_ratio = dataset_config.ratio
    sub_sample = (0. < sparse_ratio < 1.)
    if sub_sample:
        cur_dir = osp.dirname(osp.realpath(__file__))
        label_list_fn = osp.join(cur_dir, 'splits', dataset_config.name,
                                 f'ssod_{sparse_ratio:.3f}-off0.pkl')
        label_lists = load_obj(label_list_fn) if \
            osp.exists(label_list_fn) else defaultdict(lambda: None)
        print(f'Loaded label list from: {label_list_fn}')
    label_list = None  # dummy

    seq_datasets = list()
    desc = f'creating rnd access {mode2str[dataset_mode]} datasets'
    for entry in tqdm(seq_dirs, desc=desc):
        # 'gen1/val/18-03-29_12-59-36_427500000_487500000/'
        if sub_sample:
            label_list = label_lists[osp.basename(str(entry))]
        if dataset_config.ssod:  # semi-supervised learning
            seq_datasets.append(SSODSequenceDataset(
                path=entry,
                dataset_mode=dataset_mode,
                dataset_config=dataset_config,
                label_list=label_list))
        else:
            seq_datasets.append(SequenceDataset(
                path=entry,
                dataset_mode=dataset_mode,
                dataset_config=dataset_config,
                label_list=label_list))

    # save the sub-sampled label lists to file
    if sub_sample and label_list is None:
        label_lists = {
            osp.basename(seq_dst.sequence.path):
            seq_dst.sequence.all_objframe_idx
            for seq_dst in seq_datasets
        }
        os.makedirs(osp.dirname(label_list_fn), exist_ok=True)
        dump_obj(label_lists, label_list_fn)
        print(f'Saving {100. * sparse_ratio:.1f}% sub-sampled label lists')

    return CustomConcatDataset(seq_datasets)


def get_weighted_random_sampler(dataset: CustomConcatDataset) -> WeightedRandomSampler:
    class2count = dict()
    ClassAndCount = namedtuple('ClassAndCount', ['class_ids', 'counts'])
    classandcount_list = list()
    print('--- START generating weighted random sampler ---')
    dataset.only_load_labels()
    for idx, data in enumerate(tqdm(dataset, desc='iterate through dataset')):
        labels: SparselyBatchedObjectLabels = data[DataType.OBJLABELS_SEQ]
        label_list, valid_batch_indices = labels.get_valid_labels_and_batch_indices()
        class_ids_seq = list()
        for label in label_list:
            class_ids_numpy = np.asarray(label.class_id.numpy(), dtype='int32')
            class_ids_seq.append(class_ids_numpy)
        class_ids_seq, counts_seq = np.unique(np.concatenate(class_ids_seq), return_counts=True)
        for class_id, count in zip(class_ids_seq, counts_seq):
            class2count[class_id] = class2count.get(class_id, 0) + count
        classandcount_list.append(ClassAndCount(class_ids=class_ids_seq, counts=counts_seq))
    dataset.load_everything()

    class2weight = {}
    for class_id, count in class2count.items():
        count = max(count, 1)
        class2weight[class_id] = 1 / count

    weights = []
    for classandcount in classandcount_list:
        weight = 0
        for class_id, count in zip(classandcount.class_ids, classandcount.counts):
            # Not only weight depending on class but also depending on number of occurrences.
            # This will bias towards sampling "frames" with more bounding boxes.
            weight += class2weight[class_id] * count
        weights.append(weight)

    print('--- DONE generating weighted random sampler ---')
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
