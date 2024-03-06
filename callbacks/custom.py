from datetime import timedelta
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint

from callbacks.detection import DetectionVizCallback


def get_ckpt_callback(config: DictConfig, ckpt_dir: str = None) -> ModelCheckpoint:
    prefix = 'val'
    metric = 'AP'
    mode = 'max'
    ckpt_callback_monitor = prefix + '/' + metric
    filename_monitor_str = prefix + '_' + metric

    ckpt_filename = 'epoch_{epoch:03d}-step_{step}-' + filename_monitor_str + '_{' + ckpt_callback_monitor + ':.4f}'
    every_n_min = config.logging.ckpt_every_min
    cktp_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor=ckpt_callback_monitor,
        filename=ckpt_filename,
        auto_insert_metric_name=False,  # because backslash would create a directory
        save_top_k=2,  # in case the best one is broken
        mode=mode,
        train_time_interval=timedelta(minutes=every_n_min),
        save_last=True,
        verbose=True)
    cktp_callback.CHECKPOINT_NAME_LAST = 'last_epoch_{epoch:03d}-step_{step}'
    return cktp_callback


def get_viz_callback(config: DictConfig) -> Callback:
    if hasattr(config.model, 'pseudo_label'):
        prefixs = ['', 'pseudo_']
    else:
        prefixs = ['']
    return DetectionVizCallback(config=config, prefixs=prefixs)
