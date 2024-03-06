import os
from typing import Tuple

import math
from omegaconf import DictConfig, open_dict

from data.utils.spatial import get_dataloading_hw


def dynamically_modify_train_config(config: DictConfig):
    with open_dict(config):
        slurm_job_id = os.environ.get("SLURM_JOB_ID")
        if slurm_job_id and slurm_job_id != '':
            config.slurm_job_id = int(slurm_job_id)

        dst_cfg = config.dataset
        dst_name = dst_cfg.name
        assert dst_name in {'gen1', 'gen4'}, f'{dst_name=} not supported'
        num_classes = 2 if dst_name == 'gen1' else 3
        dst_cfg.num_classes = num_classes
        dataset_hw = get_dataloading_hw(dataset_config=dst_cfg)
        dst_cfg.ev_repr_hw = dataset_hw
        dataset_path = dst_cfg.path
        # TTA
        if not config.is_train:
            dst_cfg.tta = config.tta
        # when training on pseudo generated dataset
        if config.is_train and 'x0.' in dataset_path and '_ss' in dataset_path:
            if config.weight:
                print('Use pre-trained weight to init self-training')
                config.suffix = f'-pretrain{config.suffix}'
            if 0 < dst_cfg.ratio < 1:
                print('Please use mdl_cfg.use_label_every to sub-sample label')
                raise ValueError('Sub-sample labels on pseudo dataset')
        if config.is_train and 'x0.' in dataset_path and '_seq' in dataset_path:
            if config.weight:
                print('Use pre-trained weight to init self-training')
                config.suffix = f'-pretrain{config.suffix}'
            if 0 < dst_cfg.train_ratio < 1:
                print('Should train on all events as they are all pse-labeled')
                raise ValueError('Sub-sample events on pseudo dataset')

        mdl_cfg = config.model
        mdl_name = mdl_cfg.name
        if hasattr(mdl_cfg, 'backbone'):
            backbone_cfg = mdl_cfg.backbone
            backbone_name = backbone_cfg.name
            if backbone_name == 'MaxViTRNN':
                partition_split_32 = backbone_cfg.partition_split_32
                assert partition_split_32 in (1, 2, 4)  # gen1: 1, gen4: 2

                multiple_of = 32 * partition_split_32
                mdl_hw = _get_modified_hw_multiple_of(hw=dataset_hw, multiple_of=multiple_of)
                print(f'Set {backbone_name} backbone (height, width) to {mdl_hw}')
                backbone_cfg.in_res_hw = mdl_hw
                # ev_repr: 240x304 for gen1, 360x640 for gen4
                # pad_size: 256x320 for gen1, 384x640 for gen4

                attention_cfg = backbone_cfg.stage.attention
                partition_size = tuple(x // (32 * partition_split_32) for x in mdl_hw)
                assert (mdl_hw[0] // 32) % partition_size[0] == 0, f'{mdl_hw[0]=}, {partition_size[0]=}'
                assert (mdl_hw[1] // 32) % partition_size[1] == 0, f'{mdl_hw[1]=}, {partition_size[1]=}'
                print(f'Set partition sizes: {partition_size}')
                attention_cfg.partition_size = partition_size

                vit_dim = backbone_cfg.embed_dim
                vit_size = _get_vit_size(vit_dim)
                backbone_cfg.vit_size = vit_size
                # 32, 48, 64 --> tiny, small, base
            else:
                print(f'{backbone_name=} not available')
                raise NotImplementedError
            mdl_cfg.head.num_classes = num_classes
            print(f'Set {num_classes=} for detection head')
        else:
            print(f'{mdl_name=} not available')
            raise NotImplementedError

        # conversion between Gen1 and Gen4
        # gen1: ('car', 'ped'); gen4: ('ped', 'cyc', 'car')
        # we make gen4's cyc setting the same as ped if missed
        if hasattr(mdl_cfg, 'pseudo_label'):
            obj_thresh = mdl_cfg.pseudo_label.obj_thresh
            cls_thresh = mdl_cfg.pseudo_label.cls_thresh
            if dst_name == 'gen1':
                if isinstance(obj_thresh, (list, tuple)):
                    assert len(obj_thresh) == 2
                if isinstance(cls_thresh, (list, tuple)):
                    assert len(cls_thresh) == 2
            elif dst_name == 'gen4':
                if not isinstance(obj_thresh, float) and len(obj_thresh) == 2:
                    obj_thresh = type(obj_thresh)([obj_thresh[1], obj_thresh[1], obj_thresh[0]])
                    mdl_cfg.pseudo_label.obj_thresh = obj_thresh
                if not isinstance(cls_thresh, float) and len(cls_thresh) == 2:
                    cls_thresh = type(cls_thresh)([cls_thresh[1], cls_thresh[1], cls_thresh[0]])
                    mdl_cfg.pseudo_label.cls_thresh = cls_thresh
            else:
                raise NotImplementedError(f'{dst_name=} not supported')
        # also do this to the detection head
        if hasattr(mdl_cfg, 'head') and hasattr(mdl_cfg.head, 'ignore_bbox_thresh') and mdl_cfg.head.ignore_bbox_thresh:
            thresh = mdl_cfg.head.ignore_bbox_thresh
            if dst_name == 'gen1':
                assert len(thresh) == 2
            elif dst_name == 'gen4':
                if len(thresh) == 2:
                    mdl_cfg.head.ignore_bbox_thresh = type(thresh)([thresh[1], thresh[1], thresh[0]])
            else:
                raise NotImplementedError(f'{dst_name=} not supported')


def _get_modified_hw_multiple_of(hw: Tuple[int, int], multiple_of: int) -> Tuple[int, ...]:
    assert isinstance(hw, tuple), f'{type(hw)=}, {hw=}'
    assert len(hw) == 2
    assert isinstance(multiple_of, int)
    assert multiple_of >= 1
    if multiple_of == 1:
        return hw
    new_hw = tuple(math.ceil(x / multiple_of) * multiple_of for x in hw)
    return new_hw


def _get_vit_size(vit_dim: int) -> str:
    if vit_dim == 64:
        size = 'base'
    elif vit_dim == 48:
        size = 'small'
    elif vit_dim == 32:
        size = 'tiny'
    else:
        raise NotImplementedError(f'Unknown ViT dim {vit_dim=}')
    return size
