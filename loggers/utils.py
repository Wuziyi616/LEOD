from omegaconf import DictConfig
from pytorch_lightning.loggers.wandb import WandbLogger


def get_wandb_logger(full_config: DictConfig) -> WandbLogger:
    """Build the native PyTorch Lightning WandB logger."""
    wandb_config = full_config.wandb

    logger = WandbLogger(
        project=wandb_config.project_name,
        name=wandb_config.wandb_name,
        id=wandb_config.wandb_id,
        # specify both to make sure it's saved in the right place
        save_dir=wandb_config.wandb_runpath,
        dir=wandb_config.wandb_runpath,
        # group=wandb_config.group_name,  # not very useful
        # log_model=True,  # don't log model weights
        # save_last_only_final=False,
        # save_code=True,
        # config_args=full_config_dict,
    )

    return logger
