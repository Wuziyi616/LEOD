from typing import Dict, Optional, Tuple, Union

import torch as th
from omegaconf import DictConfig

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from ...recurrent_backbone import build_recurrent_backbone
from .build import build_yolox_fpn, build_yolox_head
from utils.timers import TimerDummy as CudaTimer
# from utils.timers import CudaTimer
from data.utils.types import BackboneFeatures, LstmStates


class YoloXDetector(th.nn.Module):
    """RNN-based MaxViT backbone + YOLOX detection head."""

    def __init__(self, model_cfg: DictConfig, ssod: bool = False):
        super().__init__()
        backbone_cfg = model_cfg.backbone
        fpn_cfg = model_cfg.fpn
        head_cfg = model_cfg.head

        self.backbone = build_recurrent_backbone(backbone_cfg)  # maxvit_rnn

        in_channels = self.backbone.get_stage_dims(fpn_cfg.in_stages)
        self.fpn = build_yolox_fpn(fpn_cfg, in_channels=in_channels)

        strides = self.backbone.get_strides(fpn_cfg.in_stages)
        self.yolox_head = build_yolox_head(head_cfg, in_channels=in_channels, strides=strides, ssod=ssod)

    def forward_backbone(self,
                         x: th.Tensor,
                         previous_states: Optional[LstmStates] = None,
                         token_mask: Optional[th.Tensor] = None) -> \
            Tuple[BackboneFeatures, LstmStates]:
        """Extract multi-stage features from the backbone.

        Input:
            x: (B, C, H, W), image
            previous_states: List[(lstm_h, lstm_c)], RNN states from prev timestep
            token_mask: (B, H, W) or None, pixel padding mask

        Returns:
            backbone_features: Dict{stage_id: feats, [B, C, h, w]}, multi-stage
            states: List[(lstm_h, lstm_c), same shape], RNN state of each stage
        """
        with CudaTimer(device=x.device, timer_name="Backbone"):
            backbone_features, states = self.backbone(x, previous_states, token_mask)
        return backbone_features, states

    def forward_detect(self,
                       backbone_features: BackboneFeatures,
                       targets: Optional[th.Tensor] = None,
                       soft_targets: Optional[th.Tensor] = None) -> \
            Tuple[th.Tensor, Union[Dict[str, th.Tensor], None]]:
        """Predict object bbox from multi-stage features.

        Returns:
            outputs: (B, N, 4 + 1 + num_cls), [(x, y, w, h), obj_conf, cls]
            losses: Dict{loss_name: loss, torch.scalar tensor}
        """
        device = next(iter(backbone_features.values())).device
        with CudaTimer(device=device, timer_name="FPN"):
            fpn_features = self.fpn(backbone_features)  # Tuple(feats, [B, C, h, w])
        if self.training:
            assert targets is not None
            with CudaTimer(device=device, timer_name="HEAD + Loss"):
                outputs, losses = self.yolox_head(fpn_features, targets, soft_targets)
            return outputs, losses
        with CudaTimer(device=device, timer_name="HEAD"):
            outputs, losses = self.yolox_head(fpn_features)
        assert losses is None
        return outputs, losses

    def forward(self,
                x: th.Tensor,
                previous_states: Optional[LstmStates] = None,
                retrieve_detections: bool = True,
                targets: Optional[th.Tensor] = None) -> \
            Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None], LstmStates]:
        backbone_features, states = self.forward_backbone(x, previous_states)
        outputs, losses = None, None
        if not retrieve_detections:
            assert targets is None
            return outputs, losses, states
        outputs, losses = self.forward_detect(backbone_features=backbone_features, targets=targets)
        return outputs, losses, states
