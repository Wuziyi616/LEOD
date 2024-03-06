import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import copy
from tqdm import tqdm
import numpy as np
import hdf5plugin  # resolve a weird h5py error

import torch
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')

import hydra
from omegaconf import DictConfig, OmegaConf

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_data_module
from modules.utils.ssod import evaluate_label, filter_w_thresh
from data.genx_utils.labels import ObjectLabels
from data.utils.types import DataType
from utils.evaluation.prophesee.evaluator import get_labelmap
from modules.pseudo_labeler import EventSeqData

from nerv.utils import AverageMeter, glob_all

empty_box = torch.zeros((0, 8)).float()
empty_box = ObjectLabels(empty_box, input_size_hw=(360, 640))


def filter_by_track(obj_labels, pse_cfg, labelmap):
    """Filter with tracklet length."""
    L = pse_cfg.min_track_len
    if L <= 0:
        return obj_labels, None, None
    labels, frame_idx = [], []
    for idx, obj_label in enumerate(obj_labels):
        if obj_label is None:
            continue
        labels.append(obj_label)
        frame_idx.append(idx)
    # forward tracking
    remove_idx, _ = EventSeqData._track(labels, frame_idx, min_track_len=L)
    # remove by setting class_id to 1024 (cannot use -1 as it is uint32)
    bbox_idx = 0
    num_bbox = {cls_name: 0 for cls_name in labelmap}
    num_remove_bbox = {cls_name: 0 for cls_name in labelmap}
    for idx, obj_label in enumerate(obj_labels):
        if obj_label is None:
            continue
        obj_labels[idx].torch_()
        new_class_id = copy.deepcopy(obj_label.class_id)
        for i in range(len(obj_label)):
            cls_id = int(new_class_id[i].item())
            num_bbox[labelmap[cls_id]] += 1
            if bbox_idx in remove_idx:
                new_class_id[i] = pse_cfg.ignore_label
                num_remove_bbox[labelmap[cls_id]] += 1
            bbox_idx += 1
        obj_labels[idx].class_id = new_class_id
    return obj_labels, num_bbox, num_remove_bbox


def filter_bbox(pred, obj_thresh=0.9, cls_thresh=0.9, ignore_label=1024):
    """Filter with objectness or class confidence scores."""
    # pred: (N, 7) th.tensor, [(x1, y1, x2, y2), obj_conf, cls_conf, cls_idx]
    obj_conf, cls_conf, cls_idx = \
        pred.objectness, pred.class_confidence, pred.class_id
    sel_mask = (filter_w_thresh(obj_conf, cls_idx, obj_thresh)) & \
        (filter_w_thresh(cls_conf, cls_idx, cls_thresh)) & \
        (cls_idx != ignore_label)
    pred.object_labels = pred.object_labels[sel_mask]
    return pred


@torch.inference_mode()
def eval_one_seq(full_cfg, pse_batch, batch, skip_gt=False):
    """Run the model on one event sequence and visualize it."""
    dst_cfg, mdl_cfg = full_cfg.dataset, full_cfg.model
    pse_data, data = pse_batch['data'], batch['data']
    # sanity check
    assert os.path.basename(pse_data[DataType.PATH][0]) == \
        os.path.basename(data[DataType.PATH][0])
    # get labels
    pse_obj_labels = pse_data[DataType.OBJLABELS_SEQ]
    pse_obj_labels = [lbl[0] for lbl in pse_obj_labels]
    labelmap = get_labelmap(dst_name=dst_cfg.name)
    pse_obj_labels, num_bbox, num_remove_bbox = filter_by_track(
        pse_obj_labels, full_cfg.model.pseudo_label, labelmap=labelmap)
    # loaded GT and skipped GT
    loaded_obj_labels = data[DataType.OBJLABELS_SEQ]
    loaded_obj_labels = [lbl[0] for lbl in loaded_obj_labels]
    skipped_obj_labels = data[DataType.SKIPPED_OBJLABELS_SEQ]
    skipped_obj_labels = [lbl[0] for lbl in skipped_obj_labels]
    # all are `L`-len list of ObjectLabels or None
    pse_labels, skipped_labels = [], []
    for pse_lbl, loaded_lbl, skipped_lbl in \
            zip(pse_obj_labels, loaded_obj_labels, skipped_obj_labels):
        if loaded_lbl is not None and not skip_gt:
            assert loaded_lbl == pse_lbl, 'GT labels mismatch'
        elif pse_lbl is not None:
            assert pse_lbl.is_pseudo_label().all(), 'Contain GT labels'
        if skipped_lbl is not None:
            skipped_labels.append(skipped_lbl.to('cuda'))
            if pse_lbl is None:
                pse_lbl = empty_box.to('cuda')
            else:
                pse_lbl = pse_lbl.to('cuda')
                # filter with score_thresh
                pse_lbl = filter_bbox(
                    pse_lbl,
                    obj_thresh=mdl_cfg.pseudo_label.obj_thresh,
                    cls_thresh=mdl_cfg.pseudo_label.cls_thresh,
                    ignore_label=mdl_cfg.pseudo_label.ignore_label)
            pse_labels.append(pse_lbl)

    # evaluate
    # in case some sequences have no skipped GT
    if len(skipped_labels) == 0:
        return {}

    # Precision & Recall
    pred_mask = np.ones(len(skipped_labels), dtype=bool)
    metrics = evaluate_label(
        skipped_labels,
        pse_labels,
        pred_mask,
        num_cls=dst_cfg.num_classes,
        prefix='ssod/')
    if num_remove_bbox is not None:
        for name in labelmap:
            metrics[f'track/num_bbox_{name}'] = num_bbox[name]
            metrics[f'track/num_remove_bbox_{name}'] = num_remove_bbox[name]
    return metrics


def eval_one_dataset(config: DictConfig):
    # ---------------------
    # Data
    # ---------------------
    dst_name = config.dataset.name
    config.batch_size.eval = 1
    config.hardware.num_workers.eval //= 2  # half-half for ori_dst & pse_dst
    config.dataset.sequence_length = 320 if dst_name == 'gen1' else 128
    config.dataset.only_load_labels = True
    config.dataset.data_augmentation.stream.start_from_zero = True
    # we first get the pesudo dataset generated by `predict.py`
    # load everything by setting ratio to -1
    sparse_ratio, subseq_ratio, pse_path = \
        config.dataset.ratio, config.dataset.train_ratio, config.dataset.path
    if sparse_ratio == -1:
        assert 0. < subseq_ratio < 1.
        config.dataset.train_ratio = -1
        print('Evaluating WSOD dataset')
    else:
        assert 0. < sparse_ratio < 1.
        config.dataset.ratio = -1
        print('Evaluating SSOD dataset')
    skip_gt = ('all_pse' in pse_path)
    if skip_gt:
        print('Pseudo dataset that does not take GT labels')
    pse_data_module = fetch_data_module(config=config)
    pse_data_module.setup(stage='predict')
    pse_loader = pse_data_module.predict_dataloader()
    # then we get the original (sub-sampled) dataset
    if sparse_ratio == -1:
        config.dataset.train_ratio = subseq_ratio
    else:
        config.dataset.ratio = sparse_ratio
    config.dataset.path = f'./datasets/{dst_name}'
    data_module = fetch_data_module(config=config)
    data_module.setup(stage='predict')
    loader = data_module.predict_dataloader()
    print('\nobj_thresh:', config.model.pseudo_label.obj_thresh)
    print('cls_thresh:', config.model.pseudo_label.cls_thresh, '\n')

    # ---------------------
    # Evaluate pseudo labels VS GT labels
    # ---------------------
    metrics_dict = {}
    for (pse_batch, batch) in tqdm(
            zip(pse_loader, loader), desc='Event sequence'):
        metrics = eval_one_seq(config, pse_batch, batch, skip_gt=skip_gt)
        for k, v in metrics.items():
            if k.startswith('num_'):
                continue
            if k not in metrics_dict:
                metrics_dict[k] = AverageMeter()
            if k.startswith('track/'):
                metrics_dict[k].update(v, n=1)
                continue
            cls_name = k.split('_')[-1]  # xxx_car
            metrics_dict[k].update(v, n=metrics[f'num_{cls_name}'])
    print('------------ Evaluation ------------')
    for k, v in metrics_dict.items():
        print(f'\t{k}: {v.avg:.4f}')
    print('------------------------------------')
    print('Dataset path:', pse_path)


@hydra.main(config_path='config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    # if given a valid dataset path, we eval it
    dst_path = config.dataset.path
    if os.path.exists(os.path.join(dst_path, 'train')):
        eval_one_dataset(config)
        exit(-1)
    # otherwise, we eval all folders under it
    all_dst_paths = glob_all(dst_path, only_dir=True)
    for dst_path in all_dst_paths:
        if not os.path.exists(os.path.join(dst_path, 'train')):
            continue
        one_config = copy.deepcopy(config)
        one_config.dataset.path = dst_path
        eval_one_dataset(one_config)


if __name__ == '__main__':
    main()
