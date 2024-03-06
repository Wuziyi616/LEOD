import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import hdf5plugin  # resolve a weird h5py error
import numpy as np
import matplotlib.pyplot as plt

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
from modules.utils.fetch import fetch_data_module, fetch_model_module


@hydra.main(config_path='config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print('------ Configuration ------')
    # print(OmegaConf.to_yaml(config))
    _ = OmegaConf.to_yaml(config)
    print('---------------------------')

    # ---------------------
    # GPU options
    # ---------------------
    gpus = config.hardware.gpus
    assert isinstance(gpus, int), 'no more than 1 GPU supported'
    gpus = [gpus]

    # ---------------------
    # Data
    # ---------------------
    if 'T4' in torch.cuda.get_device_name() and \
            config.tta.enable and config.tta.hflip:
        if config.dataset.name == 'gen1':
            config.batch_size.eval = 12  # to avoid OOM on T4 GPU
        else:
            config.batch_size.eval = 6
    if config.reverse:
        config.dataset.reverse_event_order = True
        print('Testing on event sequences with reversed temporal order.')
    data_module = fetch_data_module(config=config)

    # ---------------------
    # Logging
    # ---------------------
    logger = CSVLogger(save_dir='./validation_logs')

    # ---------------------
    # Model
    # ---------------------
    module = fetch_model_module(config=config).eval()
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
        trainer.test(model=module, datamodule=data_module)
    print(f'Evaluating {config.checkpoint=} finished.')
    print(f'Conf_thresh: {config.model.postprocess.confidence_threshold}')


if __name__ == '__main__':
    main()
