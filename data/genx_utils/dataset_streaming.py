from collections import defaultdict
from functools import partialmethod
import os
import os.path as osp
import copy
from pathlib import Path
from typing import List, Union

from omegaconf import DictConfig
from torchdata.datapipes.map import MapDataPipe
from tqdm import tqdm

from data.genx_utils.sequence_streaming import SequenceForIter, \
    RandAugmentIterDataPipe, SSODRandAugmentIterDataPipe
from data.utils.stream_concat_datapipe import ConcatStreamingDataPipe
from data.utils.stream_sharded_datapipe import ShardedStreamingDataPipe
from data.utils.types import DatasetMode, DatasetType
from utils.preprocessing import subsample_sequence

from nerv.utils import load_obj


def build_streaming_dataset(
    dataset_mode: DatasetMode, dataset_config: DictConfig,
    batch_size: int, num_workers: int, pseudo_labeling: bool = False
) -> Union[ConcatStreamingDataPipe, ShardedStreamingDataPipe]:
    """Build a dataset similar to torch.utils.data.Dataset.
    Each training seq contains a label.
    Note: test dataset is always created with full sequences.
    """
    dataset_path = Path(dataset_config.path)
    assert dataset_path.is_dir(), f'{str(dataset_path)}'

    mode2str = {
        DatasetMode.TRAIN: 'train',
        DatasetMode.VALIDATION: 'val',
        DatasetMode.TESTING: 'test',
    }

    split_path = dataset_path / mode2str[dataset_mode]
    while split_path.is_symlink():
        split_path = Path(os.readlink(str(split_path)))
    assert split_path.is_dir()
    num_full_sequences, num_splits, num_split_sequences = 0, 0, 0
    guarantee_labels = (dataset_mode == DatasetMode.TRAIN)
    # guarantee at least one label for each loaded sub-seq during training

    sparse_ratio, label_lists = -1., defaultdict(lambda: None)
    if dataset_mode == DatasetMode.TRAIN:
        # generate pseudo labels for all frames
        if pseudo_labeling:
            guarantee_labels = False  # need to load all frames
            print('Build streaming dataset for pseudo labeling.')
        # sub-sample labeling frequency (WSOD)
        if 0. < dataset_config.ratio < 1.:
            guarantee_labels = False  # enable loading seq without labels
            sparse_ratio = dataset_config.ratio
            print(f'Use sparse sequence label with ratio: {sparse_ratio:.3f}')
            # load sub-sampled label index list for each sequence
            # should exist, as we already construct them in `dataset_rnd.py`
            cur_dir = osp.dirname(osp.realpath(__file__))
            label_list_fn = osp.join(cur_dir, 'splits', dataset_config.name,
                                     f'ssod_{sparse_ratio:.3f}-off0.pkl')
            label_lists = load_obj(label_list_fn)  # Dict[str, List[int]]
            print(f'Loaded label list from: {label_list_fn}')
        # sub-sample event sequences, but keep the labeling frequency (SSOD)
        seq_dirs = subsample_sequence(split_path, dataset_config.train_ratio)
        # when generating pseudo labels in the SSOD setting
        # we also need to load those skipped event sequences
        # we will make them skip all labels, as if the whole seq is skipped
        if pseudo_labeling and 0. < dataset_config.train_ratio < 1.:
            all_dirs = subsample_sequence(split_path, -1)
            cnt = 0
            for entry in all_dirs:
                if entry not in seq_dirs:
                    label_lists[osp.basename(str(entry))] = []
                    cnt += 1
            assert cnt == len(all_dirs) - len(seq_dirs)
            seq_dirs = all_dirs
    elif dataset_mode == DatasetMode.VALIDATION:
        seq_dirs = subsample_sequence(split_path, dataset_config.val_ratio)
    elif dataset_mode == DatasetMode.TESTING:
        seq_dirs = subsample_sequence(split_path, dataset_config.test_ratio)
    else:
        raise NotImplementedError(f'Unknown dataset mode: {dataset_mode}')

    # load and convert each event seq to iterable datapipes
    # can load data samples, each as a `L` event repr + labels + etc.
    datapipes = list()
    desc = f'creating streaming {mode2str[dataset_mode]} datasets'
    for entry in tqdm(seq_dirs, desc=desc):
        # 'gen1/val/18-03-29_12-59-36_427500000_487500000/'
        label_list = label_lists[osp.basename(str(entry))]
        new_datapipes = get_sequences(
            path=entry,
            dataset_config=dataset_config,
            guarantee_labels=guarantee_labels,
            label_list=label_list,
            sparse_ratio=sparse_ratio)
        if len(new_datapipes) == 0:  # corner case when no pse-labels
            print('Skip empty sequence:', str(entry))
            continue
        elif len(new_datapipes) == 1:  # val/test
            num_full_sequences += 1
        else:  # train
            num_splits += 1
            num_split_sequences += len(new_datapipes)
        datapipes.extend(new_datapipes)
    print(f'{num_full_sequences=}\n{num_splits=}\n{num_split_sequences=}')

    # TTA
    if hasattr(dataset_config, 'tta') and \
            dataset_config.tta.enable and dataset_config.tta.tflip:
        new_datapipes = copy.deepcopy(datapipes)
        for dp in new_datapipes:
            dp.time_flip = True
        datapipes.extend(new_datapipes)
        assert not dataset_config.reverse_event_order
        print('\nEnable time-flip TTA.\n')
    # sometimes we want to reverse the temporal order of event sequences
    if dataset_config.reverse_event_order:
        for dp in datapipes:
            dp.time_flip = True
        print('\nReverse the temporal order of event sequences.\n')
    # sometimes we only want to load labels
    if dataset_config.only_load_labels:
        for dp in datapipes:
            dp.only_load_labels()
        print('\nOnly load labels for event sequences.\n')

    if dataset_mode == DatasetMode.TRAIN and not pseudo_labeling:
        return build_streaming_train_dataset(
            datapipes=datapipes, dataset_config=dataset_config,
            batch_size=batch_size, num_workers=num_workers)
    elif dataset_mode == DatasetMode.TRAIN and pseudo_labeling:
        return build_streaming_evaluation_dataset(
            datapipes=datapipes, batch_size=batch_size)
    elif dataset_mode in (DatasetMode.VALIDATION, DatasetMode.TESTING):
        return build_streaming_evaluation_dataset(
            datapipes=datapipes, batch_size=batch_size)
    else:
        raise NotImplementedError


def get_sequences(
    path: Path, dataset_config: DictConfig, guarantee_labels: bool,
    label_list: List[int] = None, sparse_ratio: float = -1.0
) -> List[SequenceForIter]:
    """Load a list of sub-seq (train) or one full event seq (val/test)."""
    assert path.is_dir()

    # extract settings from config
    sequence_length = dataset_config.sequence_length
    ev_representation_name = dataset_config.ev_repr_name
    downsample_by_factor_2 = dataset_config.downsample_by_factor_2
    tflip_offset = dataset_config.data_augmentation.tflip_offset
    start_from_zero = dataset_config.data_augmentation.stream.start_from_zero
    if dataset_config.name == 'gen1':
        dataset_type = DatasetType.GEN1
    elif dataset_config.name == 'gen4':
        dataset_type = DatasetType.GEN4
    else:
        raise NotImplementedError

    # in training, we need event seq that have labels
    # returns a filtered list of event seq, where there is at least 1 labeled
    #   frame **within** each seq
    if guarantee_labels:
        assert sparse_ratio == -1., \
            'cannot guarantee label when loading sparse labels in stream mode'
        return SequenceForIter.get_sequences_with_guaranteed_labels(
            path=path,
            ev_representation_name=ev_representation_name,
            sequence_length=sequence_length,
            dataset_type=dataset_type,
            downsample_by_factor_2=downsample_by_factor_2,
            tflip_offset=tflip_offset)

    # in val/test/sparse_training, directly load the entire event seq
    return [SequenceForIter(
        path=path,
        ev_representation_name=ev_representation_name,
        sequence_length=sequence_length,
        dataset_type=dataset_type,
        downsample_by_factor_2=downsample_by_factor_2,
        objframe_idx=label_list,
        data_ratio=sparse_ratio,
        tflip_offset=tflip_offset,
        start_from_zero=start_from_zero)]


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


def build_streaming_train_dataset(
    datapipes: List[MapDataPipe], dataset_config: DictConfig,
    batch_size: int, num_workers: int
) -> ConcatStreamingDataPipe:
    assert len(datapipes) > 0
    if dataset_config.ssod:  # semi-supervised learning
        augmentation_datapipe_type = partialclass(
            SSODRandAugmentIterDataPipe, dataset_config=dataset_config)
    else:
        augmentation_datapipe_type = partialclass(
            RandAugmentIterDataPipe, dataset_config=dataset_config)
    streaming_dataset = ConcatStreamingDataPipe(
        datapipe_list=datapipes,
        batch_size=batch_size,
        num_workers=num_workers,
        augmentation_pipeline=augmentation_datapipe_type,
        print_seed_debug=False)
    return streaming_dataset


def build_streaming_evaluation_dataset(
    datapipes: List[MapDataPipe], batch_size: int
) -> ShardedStreamingDataPipe:
    assert len(datapipes) > 0
    fill_value = datapipes[0].get_fully_padded_sample()
    streaming_dataset = ShardedStreamingDataPipe(
        datapipe_list=datapipes,
        batch_size=batch_size,
        fill_value=fill_value)
    return streaming_dataset
