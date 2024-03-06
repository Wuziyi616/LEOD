import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import gc
from tqdm import tqdm
import hdf5plugin  # resolve a weird h5py error

import numpy as np
import torch
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelSummary

from config.modifier import dynamically_modify_train_config
from data.genx_utils.labels import ObjectLabels
from data.utils.misc import read_npz_labels, read_ev_repr, get_ev_dir, get_ev_h5_fn
from modules.utils.fetch import fetch_data_module, fetch_model_module

from nerv.utils import load_obj, dump_obj, glob_all


def read_old_and_new_data(new_dir):
    """Read data from original and newly generated dataset."""
    new_dir = new_dir[:-1] if new_dir[-1] == '/' else new_dir
    new_ev_dir = get_ev_dir(new_dir)
    seq_name = os.path.basename(new_dir)  # 17-03-30_12-53-58_1037500000_10975
    dst_name = 'gen1' if 'gen1' in new_dir else 'gen4'
    old_dir = os.path.join('datasets', dst_name, 'train', seq_name)
    old_ev_dir = get_ev_dir(old_dir)
    old_ev_fn = get_ev_h5_fn(old_ev_dir)
    ev_repr = read_ev_repr(old_ev_fn)
    new_objframe_idx_2_repr_idx = np.load(
        os.path.join(new_ev_dir, 'objframe_idx_2_repr_idx.npy'))
    new_labels, new_objframe_idx_2_label_idx = read_npz_labels(new_dir)
    old_objframe_idx_2_repr_idx = np.load(
        os.path.join(old_ev_dir, 'objframe_idx_2_repr_idx.npy'))
    old_labels, old_objframe_idx_2_label_idx = read_npz_labels(old_dir)
    return ev_repr, new_objframe_idx_2_repr_idx, new_labels, \
        new_objframe_idx_2_label_idx, old_objframe_idx_2_repr_idx, \
        old_labels, old_objframe_idx_2_label_idx


def get_label(labels, objframe_idx_2_label_idx, objframe_idx, hw, ds_by2):
    start_idx = objframe_idx_2_label_idx[objframe_idx]
    end_idx = objframe_idx_2_label_idx[objframe_idx + 1] if \
        objframe_idx < len(objframe_idx_2_label_idx) - 1 else labels.shape[0]
    labels = ObjectLabels.from_structured_array(
        labels[start_idx:end_idx], hw, downsample_factor=2 if ds_by2 else None)
    labels.clamp_to_frame_()
    labels.numpy_()
    return labels


def verify_data(new_dir, ratio=-1, ds_by2=False):
    """Verify whether the newly generated data is correct."""
    ev_repr, new_objframe_idx_2_repr_idx, new_labels, \
        new_objframe_idx_2_label_idx, old_objframe_idx_2_repr_idx, \
        old_labels, old_objframe_idx_2_label_idx = \
        read_old_and_new_data(new_dir)
    hw = tuple(ev_repr.shape[-2:])
    if ds_by2:
        hw = tuple(s * 2 for s in hw)
    # skip labels
    dst_name = 'gen1' if 'gen1' in new_dir else 'gen4'
    if (0. < ratio < 1.):
        label_list_fn = os.path.join('data/genx_utils/splits', dst_name,
                                     f'ssod_{ratio:.3f}-off0.pkl')
        label_list = load_obj(label_list_fn)[os.path.basename(new_dir)]
    else:
        label_list = list(range(len(old_objframe_idx_2_repr_idx)))
    # frame idx should be correct
    print(f'{new_objframe_idx_2_repr_idx[-1]=}, {ev_repr.shape[0]=}')
    assert new_objframe_idx_2_repr_idx[-1] <= ev_repr.shape[0]
    assert (new_objframe_idx_2_repr_idx >= 0).all()
    assert all(idx == new_objframe_idx_2_repr_idx[:i + 1].max()
               for i, idx in enumerate(new_objframe_idx_2_repr_idx))  # sorted
    # match GT labels
    assert all(old_objframe_idx_2_repr_idx[i] in new_objframe_idx_2_repr_idx
               for i in label_list)
    for old_frame_idx, repr_idx in enumerate(old_objframe_idx_2_repr_idx):
        # find the new_frame_idx with the same repr_idx
        new_frame_idx = np.where(new_objframe_idx_2_repr_idx == repr_idx)[0]
        if len(new_frame_idx) == 0:
            assert old_frame_idx not in label_list, 'GT not retained'
            continue
        new_frame_idx = new_frame_idx[0]
        new_label = get_label(
            new_labels, new_objframe_idx_2_label_idx,
            new_frame_idx, hw=hw, ds_by2=ds_by2)
        assert (new_label.objectness >= 0).all() and \
            (new_label.objectness <= 1).all()
        assert (new_label.class_confidence >= 0).all() and \
            (new_label.class_confidence <= 1).all()
        if old_frame_idx not in label_list:
            assert new_label.is_pseudo_label().all(), 'should not contain GT'
            continue
        # if it's GT we don't skip, check the bbox is unchanged
        old_label = get_label(
            old_labels, old_objframe_idx_2_label_idx,
            old_frame_idx, hw=hw, ds_by2=ds_by2)
        for k in ObjectLabels.keys():
            assert np.abs(old_label.get(k) - new_label.get(k)).max() < 1e-6


@hydra.main(config_path='config', config_name='predict', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    # print(OmegaConf.to_yaml(config))
    _ = OmegaConf.to_yaml(config)
    print('---------------------------')

    if config.save_dir[-1] == '/':
        config.save_dir = config.save_dir[:-1]
    save_dir = os.path.dirname(config.save_dir)
    ss_ratio, seq_ratio = config.dataset.ratio, config.dataset.train_ratio
    # whether we run on sub-sampled dataset
    subsample = (0. < ss_ratio < 1.)
    subseq = (0. < seq_ratio < 1.)
    # we may only want to do tracking post-processing
    tracking = config.dataset.only_load_labels
    if tracking:
        assert not config.tta.enable, 'No need to do TTA in tracking-only case'
        config.dataset.ratio = -1  # don't skip any generated (pseudo) labels
        config.dataset.train_ratio = -1
        dst_path = config.dataset.path
        if 'x0.' in dst_path and '_ss' in dst_path:
            assert subsample, 'Should track on SSOD pseudo datasets'
        elif 'x0.' in dst_path and '_seq' in dst_path:
            assert subseq, 'Should track on WSOD pseudo datasets'
        else:
            print('\nWarning: tracking on full dataset\n')
        assert 'track' in config.save_dir or 'trk' in config.save_dir
        assert dst_path[:-1] in config.save_dir
        assert config.model.pseudo_label.min_track_len > 1
        if config.model.pseudo_label.inpaint:
            assert 'inpaint' in config.save_dir or 'ipt' in config.save_dir
    elif subsample:
        assert not subseq, 'Cannot do SSOD and WSOD at the same time'
        assert str(ss_ratio) in config.save_dir, f'Bad {config.save_dir=}'
        assert str(ss_ratio) in config.checkpoint, f'Bad {config.checkpoint=}'
    elif subseq:
        assert str(seq_ratio) in config.save_dir, f'Bad {config.save_dir=}'
        assert str(seq_ratio) in config.checkpoint, f'Bad {config.checkpoint=}'
    else:
        print('Generating labels using full dataset trained model')

    # ---------------------
    # GPU options
    # ---------------------
    gpus = config.hardware.gpus
    assert isinstance(gpus, int), 'no more than 1 GPU supported'
    gpus = [gpus]

    # ---------------------
    # Data
    # ---------------------
    if (not tracking) and 'T4' in torch.cuda.get_device_name() and \
            config.tta.enable and config.tta.hflip:
        if config.dataset.name == 'gen1':
            config.batch_size.eval = 12  # to avoid OOM on T4 GPU (16 GB)
        else:
            config.batch_size.eval = 6
    data_module = fetch_data_module(config=config)

    # ---------------------
    # Logging
    # ---------------------
    logger = CSVLogger(save_dir='./validation_logs')

    # ---------------------
    # Model
    # ---------------------
    module = fetch_model_module(config=config).eval()
    if not tracking:
        module.load_weight(config.checkpoint)

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = [ModelSummary(max_depth=2)]

    # ---------------------
    # Validation
    # ---------------------

    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        default_root_dir=None,
        devices=gpus,
        logger=logger,
        log_every_n_steps=100,
        precision=config.training.precision,
        move_metrics_to_cpu=False,
    )
    with torch.inference_mode():
        trainer.predict(
            model=module, datamodule=data_module, return_predictions=False)

        # save results
        for ev_data in tqdm(module.ev_path_2_ev_data.values(), desc='Saving'):
            assert ev_data.eoe, 'some data are not evaluated in full sequence'
            ev_data.save(save_dir=module.save_dir, dst_name=module.dst_name)
    print(f'Generate labels for {module.ev_cnt} event sequences.')

    # run evaluation (only when we have skipped GT)
    if (not tracking) and (subsample or subseq):
        # save stats for plotting and analysis
        results = {k: np.array(v) for k, v in module.results.items()}
        f = os.path.join(save_dir, 'model_results.pkl')
        dump_obj(results, f)

    # handle corner case when one seq doesn't have any pse-labels
    # then it won't be loaded --> won't be saved --> causing a wrong num_seq
    if tracking:
        ori_seq_dirs = os.listdir(os.path.join(dst_path, 'train'))
        cur_seq_dirs = os.listdir(os.path.join(save_dir, 'train'))
        for seq_dir in ori_seq_dirs:
            if seq_dir not in cur_seq_dirs:
                seq_dir = os.path.join(dst_path, 'train', seq_dir)
                cmd = f"cp -r {seq_dir} {os.path.join(save_dir, 'train')}"
                print(cmd)
                os.system(cmd)

    print(f'Dataset generation with {config.checkpoint=} finished.')

    # check the correctness of the generated labels, e.g., if GT is retained
    # only when we have GT labels, i.e., the WSOD case
    check_seqs = []
    if subsample and config.get('use_gt', True):
        all_seqs = glob_all(config.save_dir, only_dir=True)
        check_seqs = all_seqs[::len(all_seqs) // 10]  # check 10% of the seqs
    for seq in tqdm(check_seqs, desc='Checking events'):
        try:
            verify_data(seq, ratio=ss_ratio,
                        ds_by2=config.dataset.downsample_by_factor_2)
        except Exception as e:
            print(f'Error in {seq}: {e}')
            raise e

    if (not tracking) and (not subsample) and (not subseq):
        exit(-1)

    # ---------------------
    del module, data_module, trainer
    torch.cuda.empty_cache()
    gc.collect()
    # evaluate the quality of the generated labels
    command = 'python val_dst.py model=pseudo_labeler '
    if subsample:
        dst_cfg = f'{config.dataset.name}x{ss_ratio}_ss'
    else:
        dst_cfg = f'{config.dataset.name}x{seq_ratio}_seq'
    dataset = f'dataset={dst_cfg} dataset.path={save_dir} '
    misc = f'+experiment/{config.dataset.name}="small.yaml" ' + \
        'checkpoint=1 ' + \
        'model.pseudo_label.obj_thresh=0.01 ' + \
        'model.pseudo_label.cls_thresh=0.01'  # small threshs so won't filter
    command = command + dataset + misc
    print(f'Running:\n\t{command}')
    _ = os.system(command)


if __name__ == '__main__':
    main()
