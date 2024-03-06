import pytorch_lightning as pl
from omegaconf import DictConfig

from modules.data.genx import DataModule as genx_data_module
from modules.detection import Module as rnn_det_module
from modules.utils.tta import TTAModule
from modules.pseudo_labeler import PseudoLabeler


def fetch_model_module(config: DictConfig) -> pl.LightningModule:
    """Build model."""
    model_str = config.model.name
    if model_str == 'rnndet':
        if config.get('tta', {}).get('enable', False):
            return TTAModule(config)
        return rnn_det_module(config)
    elif model_str == 'pseudo_labeler':
        return PseudoLabeler(config)
    raise NotImplementedError


def fetch_data_module(config: DictConfig) -> pl.LightningDataModule:
    """Build dataloaders."""
    batch_size_train = config.batch_size.train
    batch_size_eval = config.batch_size.eval
    num_workers_generic = config.hardware.get('num_workers', None)
    num_workers_train = config.hardware.num_workers.get('train', num_workers_generic)
    num_workers_eval = config.hardware.num_workers.get('eval', num_workers_generic)
    dataset_str = config.dataset.name
    if dataset_str in {'gen1', 'gen4'}:
        return genx_data_module(config.dataset,
                                num_workers_train=num_workers_train,
                                num_workers_eval=num_workers_eval,
                                batch_size_train=batch_size_train,
                                batch_size_eval=batch_size_eval)
    raise NotImplementedError
