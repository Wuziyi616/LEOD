import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch

torch.multiprocessing.set_sharing_strategy('file_system')
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True

import pwd
import hdf5plugin  # resolve a weird h5py error
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
from pytorch_lightning.strategies import DDPStrategy

from callbacks.custom import get_ckpt_callback, get_viz_callback
from callbacks.gradflow import GradFlowLogCallback
from config.modifier import dynamically_modify_train_config
from data.utils.types import DatasetSamplingMode
from loggers.utils import get_wandb_logger
from modules.utils.fetch import fetch_data_module, fetch_model_module

from nerv.training import find_old_slurm_id
from nerv.utils import sort_file_by_time, glob_all


def get_exp_name(config: DictConfig):
    """Compose the name used in wandb run's name and ckp path."""
    # dataset
    dst_name = OmegaConf.from_cli()['dataset']
    assert config.dataset.name in dst_name
    # model
    model_name = OmegaConf.from_cli()['model']
    assert config.model.name in model_name
    vit_size = config.model.backbone.vit_size
    model_name = f'{model_name}_{vit_size}'
    # training
    gpu_config = config.hardware.gpus
    gpus = OmegaConf.to_container(gpu_config) if \
        OmegaConf.is_config(gpu_config) else gpu_config
    gpus = gpus if isinstance(gpus, list) else [gpus]
    num_gpus = len(gpus)
    bs = config.batch_size.train * num_gpus
    steps = config.training.max_steps // 1000
    train_name = f'bs{bs}_iter{steps}k'
    lr = config.training.learning_rate
    if 'gen1' in dst_name and lr != 0.0002:
        train_name = f'{train_name}_lr{lr:.0e}'.replace('e-0', 'e-')
    elif 'gen4' in dst_name and lr != 0.000346:
        train_name = f'{train_name}_lr{lr:.0e}'.replace('e-0', 'e-')
    # misc
    suffix = config.suffix if hasattr(config, 'suffix') else ''
    # name
    exp_name = f'{model_name}-{dst_name}-{train_name}{suffix}'
    return exp_name


def detect_ckpt(ckpt_path: str):
    """Automatically detect checkpoints in the ckpt_path.
    Useful in SLURM preemption systems.
    """
    last_ckpt = None

    # automatically detect checkpoints
    if os.path.exists(ckpt_path):
        ckp_files = glob_all(ckpt_path)
        ckp_files = [ckp for ckp in ckp_files if ckp.endswith('.ckpt')]
        if ckp_files:
            ckp_files = sort_file_by_time(ckp_files)  # 0-th is oldest
            last_ckpt = ckp_files[-1]
            try:
                _ = torch.load(last_ckpt, map_location='cpu')
            # in case the last ckp is corrupted
            except:
                os.remove(last_ckpt)
                last_ckpt = None
                if len(ckp_files) > 1:
                    last_ckpt = ckp_files[-2]
            print(f'INFO: automatically detect checkpoint {last_ckpt}')

    return last_ckpt


@hydra.main(config_path='config', config_name='train', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    print(OmegaConf.to_yaml(config))
    print('---------------------------')

    # ---------------------
    # Reproducibility
    # ---------------------
    dataset_train_sampling = config.dataset.train.sampling
    assert dataset_train_sampling in iter(DatasetSamplingMode)
    disable_seed_everything = dataset_train_sampling in (DatasetSamplingMode.STREAM, DatasetSamplingMode.MIXED)
    if disable_seed_everything:
        print('Disabling PL seed everything because of unresolved issues with shuffling during training on streaming '
              'datasets')
    seed = config.reproduce.seed_everything
    if seed is not None and not disable_seed_everything:
        assert isinstance(seed, int)
        print(f'USING pl.seed_everything WITH {seed=}')
        pl.seed_everything(seed=seed, workers=True)

    # ---------------------
    # DDP
    # ---------------------
    gpu_config = config.hardware.gpus
    gpus = OmegaConf.to_container(gpu_config) if OmegaConf.is_config(gpu_config) else gpu_config
    gpus = gpus if isinstance(gpus, list) else [gpus]
    distributed_backend = config.hardware.dist_backend
    assert distributed_backend in ('nccl', 'gloo'), f'{distributed_backend=}'
    strategy = DDPStrategy(process_group_backend=distributed_backend,
                           find_unused_parameters=False,
                           gradient_as_bucket_view=True) if len(gpus) > 1 else None

    # ---------------------
    # Data
    # ---------------------
    data_module = fetch_data_module(config=config)

    # ---------------------
    # Logging and Checkpoints
    # ---------------------
    # cluster-specific
    SLURM_JOB_ID = os.environ.get('SLURM_JOB_ID')
    CHECKPOINT = './checkpoint/'
    exp_name = get_exp_name(config)
    ckpt_dir = os.path.join(CHECKPOINT, exp_name, 'models')
    os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
    wandb_name = f'{exp_name}-{SLURM_JOB_ID}'

    # on clusters, quota is limited
    # soft link temp space for checkpointing
    if SLURM_JOB_ID and os.path.isdir('/checkpoint/'):
        usr = pwd.getpwuid(os.getuid())[0]
        new_dir = f'/checkpoint/{usr}/{SLURM_JOB_ID}/'
        # `ckpt_dir` might exist, which means we are resuming training
        # retrieve the old slurm id so that we can resume the wandb run!
        if os.path.exists(ckpt_dir):
            # find slurm_id
            old_slurm_id = slurm_id = find_old_slurm_id(ckpt_dir)
            if slurm_id is None:
                slurm_id = SLURM_JOB_ID
            wandb_name = wandb_id = f'{exp_name}-{slurm_id}'
            # move everything to the new dir as the old dir might be purged
            if str(old_slurm_id) != str(SLURM_JOB_ID):
                for f in sort_file_by_time(glob_all(ckpt_dir)):  # from oldest
                    if 'SLURM_JOB_FINISHED' in f:
                        os.system(f'rm -f {f}')
                    else:
                        os.system(f'mv {f} {new_dir}')
            # remove it (the soft link)
            os.system(f'rm -rf {ckpt_dir}')
        assert not os.path.exists(ckpt_dir)
        os.system(f'ln -s {new_dir} {ckpt_dir}')
        os.system(f"touch {os.path.join(ckpt_dir, 'DELAYPURGE')}")
        wandb_id = wandb_name
    else:
        os.makedirs(ckpt_dir, exist_ok=True)
        wandb_id = None

    config.wandb.wandb_name = wandb_name
    config.wandb.wandb_id = wandb_id
    config.wandb.wandb_runpath = ckpt_dir
    config.wandb.group_name = config.dataset.name

    # we use native wandb logger as we don't need to log checkpoints
    logger = get_wandb_logger(config)
    # automatically detect checkpoints
    ckpt_path = detect_ckpt(ckpt_dir)  # None or a previous checkpoint's path
    if not ckpt_path and config.checkpoint:
        ckpt_path = config.checkpoint  # pre-specify a checkpoint's path
        print(f'INFO: use pre-specified checkpoint {ckpt_path}')

    # ---------------------
    # Model
    # ---------------------
    module = fetch_model_module(config=config)
    if not ckpt_path and config.weight:
        module.load_weight(config.weight)  # only load weight
        print(f'INFO: only load weight from {config.weight}')

    # ---------------------
    # Callbacks and Misc
    # ---------------------
    callbacks = list()
    callbacks.append(get_ckpt_callback(config, ckpt_dir=ckpt_dir))
    callbacks.append(GradFlowLogCallback(config.logging.train.log_model_every_n_steps * 100))
    if config.training.lr_scheduler.use:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    if config.logging.train.high_dim.enable or config.logging.validation.high_dim.enable:
        viz_callback = get_viz_callback(config=config)
        callbacks.append(viz_callback)
    callbacks.append(ModelSummary(max_depth=2))

    logger.watch(model=module, log='all', log_freq=config.logging.train.log_model_every_n_steps, log_graph=True)

    # ---------------------
    # Training
    # ---------------------

    # currently, we both eval every 20k iters
    val_check_interval = config.validation.val_check_interval
    check_val_every_n_epoch = config.validation.check_val_every_n_epoch
    assert val_check_interval is None or check_val_every_n_epoch is None

    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=callbacks,
        enable_checkpointing=True,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        default_root_dir=None,
        devices=len(gpus),
        gradient_clip_val=config.training.gradient_clip_val,
        gradient_clip_algorithm='value',
        limit_train_batches=config.training.limit_train_batches,
        limit_val_batches=config.validation.limit_val_batches,
        logger=logger,
        log_every_n_steps=config.logging.train.log_every_n_steps,
        plugins=None,
        precision=config.training.precision,
        max_epochs=config.training.max_epochs,
        max_steps=config.training.max_steps,
        strategy=strategy,
        sync_batchnorm=False if strategy is None else True,
        benchmark=config.reproduce.benchmark,
        deterministic=config.reproduce.deterministic_flag,
    )
    trainer.fit(model=module, ckpt_path=ckpt_path, datamodule=data_module)

    # copy the last ckpt to outer dir in case of auto-purge
    if not SLURM_JOB_ID:
        exit(-1)
    last_ckpt = detect_ckpt(ckpt_dir)
    last_name = os.path.basename(last_ckpt)
    ckpt = torch.load(last_ckpt, map_location='cpu')
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    torch.save(ckpt, os.path.join(CHECKPOINT, exp_name, last_name))


if __name__ == '__main__':
    main()
