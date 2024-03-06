"""
Original Yolox Head code with slight modifications
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from models.detection.yolox.utils import bboxes_iou

from .losses import IOUloss, FocalLoss
from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    """YOLOX detection head."""

    def __init__(
            self,
            num_classes=80,
            strides=(8, 16, 32),
            in_channels=(256, 512, 1024),
            act="silu",
            depthwise=False,
            compile_cfg: Optional[Dict] = None,
            obj_focal_loss=False,
            bbox_loss_weighting='',  # 'obj' or 'cls' or 'objxcls' or ''
            ignore_bg_k=-1,  # ignore bg loss on the highest k% pred_scores
            reg_weight=5.0,  # bbox localization
            obj_weight=1.0,  # objectness, whether there is an object
            cls_weight=1.0,  # classification * predicted bbox IoU
            ignore_bbox_thresh=None,  # bbox with cls/obj lower than this will be ignored
            ignore_label=1024,  # skip loss on bbox with this class_id
    ):
        super().__init__()

        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        self.output_strides = None
        self.output_grids = None

        # Automatic width scaling according to original YoloX channel dims.
        # in[-1]/out = 4/1
        # out = in[-1]/4 = 256 * width
        # -> width = in[-1]/1024
        largest_base_dim_yolox = 1024
        largest_base_dim_from_input = in_channels[-1]
        width = largest_base_dim_from_input/largest_base_dim_yolox

        hidden_dim = int(256*width)
        self.hidden_dim = hidden_dim

        # build multi-scale CLS and REG prediction layers
        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=in_channels[i],
                    out_channels=hidden_dim,
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=self.num_classes,  # no background class here
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=hidden_dim,
                    out_channels=1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")
        self.obj_loss_fn = focal_loss if obj_focal_loss else bcewithlog_loss
        self.cls_loss_fn = bcewithlog_loss
        self.iou_loss = IOUloss(reduction="mean")

        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

        # loss weight
        self.ignore_bg_k = ignore_bg_k
        if ignore_bg_k > 0:
            print(f'Ignore BG loss on top {ignore_bg_k} of anchors')
        self.bbox_loss_weighting = bbox_loss_weighting
        if bbox_loss_weighting:
            print(f'Weighting bbox loss using {bbox_loss_weighting}')
        self.reg_weight = reg_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.is_gt_label = True
        self.ignore_bbox_thresh = ignore_bbox_thresh
        self.ignore_label = ignore_label  # skip loss on bbox of this class

        # According to Focal Loss paper:
        self.initialize_biases(prior_prob=0.01)

        ###### Compile if requested ######
        if compile_cfg is not None:
            compile_mdl = compile_cfg['enable']
            if compile_mdl and th_compile is not None:
                self.forward = th_compile(self.forward, **compile_cfg['args'])
            elif compile_mdl:
                print('Could not compile YOLOXHead because torch.compile is not available')
        ##################################

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(1, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, pred_probs=None):
        assert pred_probs is None
        # xin: multi-level feature maps from backbone + FPN
        # labels: [B, N, 5/6]; padded to a fixed `N` dim per image
        #   5: GT labels, [cls_id, (x, y, w, h)], (x, y) is bbox center
        #   6: pesudo soft labels, [cls_id, (x, y, w, h), obj_conf]
        train_outputs = []
        inference_outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)  # [B, num_cls, h, w]

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)  # [B, 4, h, w]
            obj_output = self.obj_preds[k](reg_feat)  # [B, 1, h, w]

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                # output: [B, 4 + 1 + num_cls, h, w]
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                # output: [B, h*w, 4 + 1 + num_cls]
                # convert `reg_output` to absolute coords, i.e. image-scale
                # grid: [1, h*w, 2]
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )  # [1, h*w]
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, 1, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )  # [B, h*w, 4]
                    origin_preds.append(reg_output.clone())
                train_outputs.append(output)  # [B, h*w, 4 + 1 + num_cls]
            inference_output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )  # [B, 4 + 1 + num_cls, h, w]
            inference_outputs.append(inference_output)

        # --------------------------------------------------------
        # Modification: return decoded output also during training
        # --------------------------------------------------------
        losses = None
        if self.training:
            losses = self.get_losses(
                x_shifts,  # List[(1, h_i*w_i)]
                y_shifts,  # List[(1, h_i*w_i)]
                expanded_strides,  # List[(1, h_i*w_i)]
                labels,  # [B, N, 5/6], cls_id+(x,y,w,h)[+obj_conf], pad to N
                torch.cat(train_outputs, 1),  # [B, n_anchors, 4 + 1 + num_cls]
                origin_preds,  # List[(B, h_i*w_i, 4)]
                dtype=xin[0].dtype,
            )
            assert len(losses) == 6
            losses = {
                "loss": losses[0],
                "iou_loss": losses[1],
                "conf_loss": losses[2],  # object-ness
                "cls_loss": losses[3],  # predicted class
                "l1_loss": losses[4],
                "num_fg": losses[5],
            }
        self.hw = [x.shape[-2:] for x in inference_outputs]
        # outputs: [batch, n_anchors, offsets (4) + obj_score (1) + num_cls]
        outputs = torch.cat(
            [x.flatten(start_dim=2) for x in inference_outputs], dim=2
        ).permute(0, 2, 1)
        if self.decode_in_inference:  # always True
            # decode to [B, num_anchors, (x, y, w, h) + obj_probs + cls_probs]
            #   in absolute coordinates, i.e. image-scale
            return self.decode_outputs(outputs), losses
        else:
            return outputs, losses

    def get_output_and_grid(self, output, k, stride, dtype):
        # output: [B, 4 + 1 + num_cls, h, w]
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, 1, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, hsize * wsize, -1
        )  # [B, h*w, 4 + 1 + num_cls]
        grid = grid.view(1, -1, 2)  # [1, h*w, 2]
        output[..., :2] = (output[..., :2] + grid) * stride  # absolute center
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride  # abs h,w
        return output, grid

    def decode_outputs(self, outputs):
        # outputs: [B, num_anchors, offsets (4) + obj_score (1) + num_cls]
        if self.output_grids is None:
            assert self.output_strides is None
            dtype = outputs.dtype
            device = outputs.device
            grids = []
            strides = []
            for (hsize, wsize), stride in zip(self.hw, self.strides):
                yv, xv = torch.meshgrid([torch.arange(hsize, device=device, dtype=dtype),
                                         torch.arange(wsize, device=device, dtype=dtype)])
                grid = torch.stack((xv, yv), 2).view(1, -1, 2)  # [1, h*w, 2]
                grids.append(grid)
                shape = grid.shape[:2]
                strides.append(torch.full((*shape, 1), stride, device=device, dtype=dtype))
            self.output_grids = torch.cat(grids, dim=1)  # [1, num_anchors, 2]
            self.output_strides = torch.cat(strides, dim=1)  # [1, num_anchors, 1]
        outputs = torch.cat([
            (outputs[..., 0:2] + self.output_grids) * self.output_strides,
            torch.exp(outputs[..., 2:4]) * self.output_strides,
            outputs[..., 4:]
        ], dim=-1)  # [B, num_anchors, (x, y, w, h) + obj_probs + cls_probs]
        return outputs

    @staticmethod
    @torch.no_grad()
    def _get_highest_score_mask(scores, k, exclude_mask=None):
        """Get k% pixels with the highest scores."""
        if k <= 0:
            return None
        scores = scores.squeeze()  # [num_anchors]
        # might need to exclude some pixels
        if exclude_mask is not None and exclude_mask.any():
            assert exclude_mask.dtype == torch.bool
            # only compute n from valid pixels
            n = int((~exclude_mask).float().sum().item() * k)
            # set excluded pixels' scores to be very small
            exclude_mask = exclude_mask.squeeze().type_as(scores)
            scores = scores * (1. - exclude_mask) + exclude_mask * (-1e6)
        else:
            n = int(scores.shape[0] * k)
        topk_mask = torch.zeros_like(scores).bool()
        if n == 0:
            return topk_mask
        _, topk_idx = scores.topk(n, dim=0, largest=True, sorted=False)
        topk_mask[topk_idx] = True
        return topk_mask

    @torch.no_grad()
    def _get_bbox_loss_weight(self, matched_gt_inds, obj_conf, cls_conf):
        """Use predicted scores to weigh the bbox loss."""
        if not self.bbox_loss_weighting:
            return None
        if '-' in self.bbox_loss_weighting:  # 'cls-w**2'
            val, expr = self.bbox_loss_weighting.split('-', 1)
        else:
            val, expr = self.bbox_loss_weighting, 'w'
        if val == 'obj':
            w = obj_conf[matched_gt_inds]
        elif val == 'cls':
            w = cls_conf[matched_gt_inds]
        elif val == 'objxcls':
            w = obj_conf[matched_gt_inds] * cls_conf[matched_gt_inds]
        else:
            raise NotImplementedError(f'Unknow {self.bbox_loss_weighting=}')
        # apply transform on `w`, e.g. 'w**2' then we will have `w` squared
        w = eval(expr)
        # normalize to mean=1
        # TODO: do this on all batch of weights, instead of per-image
        # w = w / w.mean()
        return w

    @torch.no_grad()
    def _ignore_bbox(self, labels):
        """Set cls_idx to `self.ignore_label` for low conf bbox."""
        if not self.ignore_bbox_thresh:
            return labels
        # labels: [B, N, 7]; padded to a fixed `N` dim per image
        #   7: GT labels, [cls_id, (xywh), obj_conf, cls_conf], (x,y) is center
        cls_idx = labels[:, :, 0]  # [B, N]
        obj_conf, cls_conf = labels[:, :, 5], labels[:, :, 6]  # [B, N]
        ignore_mask = torch.zeros_like(cls_idx).bool()  # [B, N]
        for idx, thresh in enumerate(self.ignore_bbox_thresh):
            mask = ((obj_conf < thresh) | (cls_conf < thresh))
            ignore_mask = (ignore_mask | ((cls_idx == idx) & mask))
        # also skip padded bbox
        non_pad_mask = (labels.sum(dim=2) > 0)  # [B, N]
        ignore_mask = (ignore_mask & non_pad_mask)
        # set cls_idx to `self.ignore_label`
        labels[:, :, 0] = torch.where(
            ignore_mask, torch.full_like(cls_idx, self.ignore_label), cls_idx)
        return labels

    def get_losses(
        self,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        labels = self._ignore_bbox(labels)

        if (labels[:, :, 0] == self.ignore_label).any():
            return self.get_losses_w_ignore(
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                outputs,
                origin_preds,
                dtype,
            )

        # labels: [B, N, 7]; padded to a fixed `N` dim per image
        #   7: GT labels, [cls_id, (xywh), obj_conf, cls_conf], (x,y) is center
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors]
        expanded_strides = torch.cat(expanded_strides, 1)  # [1, n_anchors]
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)  # [B, n_anchors, 4]

        gt_cls_idxs = []
        cls_targets = []
        reg_targets = []
        l1_targets = []
        bbox_ws = []
        obj_targets = []
        fg_masks = []
        ignore_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                gt_matched_classes = outputs.new_zeros(0)
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                bbox_w = outputs.new_zeros(0)
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]  # [n, 4]
                gt_classes = labels[batch_idx, :num_gt, 0]  # [n]
                gt_obj_conf = labels[batch_idx, :num_gt, 5]  # [n]
                gt_cls_conf = labels[batch_idx, :num_gt, 6]  # [n]
                bboxes_preds_per_image = bbox_preds[batch_idx]  # [n_anchors,4]

                try:
                    (
                        gt_matched_classes,  # [n_pos_matched]
                        fg_mask,  # [n_anchors], anchors that are positive
                        pred_ious_this_matching,  # [n_pos_matched]
                        matched_gt_inds,  # [n_pos_matched]
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds[batch_idx],
                        obj_preds[batch_idx],
                    )
                except RuntimeError as e:
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise

                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds[batch_idx],
                        obj_preds[batch_idx],
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                bbox_w = self._get_bbox_loss_weight(
                    matched_gt_inds, gt_obj_conf, gt_cls_conf)

                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            gt_cls_idxs.append(gt_matched_classes.long())
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            bbox_ws.append(bbox_w)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            ignore_mask = self._get_highest_score_mask(
                obj_preds[batch_idx], self.ignore_bg_k, exclude_mask=fg_mask)
            ignore_masks.append(ignore_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        gt_cls_idxs = torch.cat(gt_cls_idxs, 0)
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        if self.bbox_loss_weighting:
            bbox_ws = torch.cat(bbox_ws, 0)
            bbox_ws = bbox_ws / bbox_ws.mean()  # normalize to mean=1
            cls_bbox_ws = bbox_ws[:, None]
        else:
            bbox_ws = cls_bbox_ws = 1.
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)  # [batch*n_anchors]
        if self.ignore_bg_k > 0:
            valid_masks = ~(torch.cat(ignore_masks, 0))
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets, weights=bbox_ws)
        if self.ignore_bg_k > 0:
            loss_obj = (
                self.obj_loss_fn(obj_preds.view(-1, 1)[valid_masks], obj_targets[valid_masks])
            ).sum() / num_fg
        else:
            loss_obj = (
                self.obj_loss_fn(obj_preds.view(-1, 1), obj_targets)
            ).sum() / num_fg
        loss_cls = (
            self.cls_loss_fn(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            ) * cls_bbox_ws
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets) * bbox_ws
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        loss_iou = self.reg_weight * loss_iou
        loss_obj = self.obj_weight * loss_obj
        loss_cls = self.cls_weight * loss_cls
        loss = loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        num_gt,
        gt_bboxes_per_image,  # [n, 4]
        gt_classes,  # [n]
        bboxes_preds_per_image,  # [n_anchors, 4]
        expanded_strides,  # [1, n_anchors]
        x_shifts,  # [1, n_anchors]
        y_shifts,  # [1, n_anchors]
        cls_preds,  # [n_anchors, n_cls]
        obj_preds,  # [n_anchors, 1]
        mode="gpu",
    ):

        if mode == "cpu":
            print("-----------Using CPU for the Current Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, geometry_relation = self.get_geometry_constraint(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )
        # fg_mask: [n_anchors] boolean, if the anchor is positive for any gt
        # geometry_relation: [n, n_pos]

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]  # [n_pos, 4]
        cls_preds_ = cls_preds[fg_mask]  # [n_pos, n_cls]
        obj_preds_ = obj_preds[fg_mask]  # [n_pos, 1]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        # pair_wise_ious: [n, n_pos], pairwise bbox IoUs

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
        )  # [n, n_cls]
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)  # [n, n_pos]

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()  # [n_pos, n_cls]
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)  # [n, n_pos]
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        # all are [n_pos_matched]
        # `fg_mask` is modifier in-place in `simota_matching`
        # it also only contains `n_pos_matched` True values

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_geometry_constraint(
        self, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
    ):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        # gt_bboxes_per_image: [n, 4]
        # others: [1, n_anchors]
        x_centers_per_image = (x_shifts + 0.5) * expanded_strides  # [1, n_anchors]
        y_centers_per_image = (y_shifts + 0.5) * expanded_strides  # [1, n_anchors]

        # in fixed center
        center_radius = 1.5
        center_dist = expanded_strides * center_radius  # [1, n_anchors]
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist  # [n, n_anchors]
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l  # [n, n_anchors]
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0  # [n, n_anchors]
        anchor_filter = is_in_centers.sum(dim=0) > 0  # [n_anchors], is positive for any gt
        geometry_relation = is_in_centers[:, anchor_filter]  # [n, n_pos]

        return anchor_filter, geometry_relation

    def simota_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # cost: [n_gt, n_pos]
        # pair_wise_ious: [n_gt, n_pos]
        # gt_classes: [n_gt]
        # fg_mask: [n_anchors] boolean, if the anchor is positive for any gt
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        n_candidate_k = min(10, pair_wise_ious.size(1))
        topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            # may trigger `RuntimeError: selected index k out of range`
            # this happens when some bbox are invalid, e.g. center out-of-frame
            # there will be no anchors matched to it, i.e. fg_mask.sum() == 0
            # this is a feature, not a bug -- we should filter bbox labels!
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)  # [n_pos]
        # deal with the case that one anchor matches multiple ground-truths
        if anchor_matching_gt.max() > 1:
            multiple_match_mask = anchor_matching_gt > 1
            _, cost_argmin = torch.min(cost[:, multiple_match_mask], dim=0)
            matching_matrix[:, multiple_match_mask] *= 0
            matching_matrix[cost_argmin, multiple_match_mask] = 1
        fg_mask_inboxes = anchor_matching_gt > 0  # [n_pos] boolean, fg_mask_inboxes.sum() == n_pos_matched
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes  # only matched anchors are positive

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)  # [n_pos_matched], each is the idx of gt_bbox
        gt_matched_classes = gt_classes[matched_gt_inds]  # [n_pos_matched]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]  # [n_pos_matched]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def get_losses_w_ignore(
        self,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        # labels: [B, N, 5/6]; padded to a fixed `N` dim per image
        #   5: GT labels, [cls_id, (x, y, w, h)], (x, y) is bbox center
        #   6: pesudo soft labels, [cls_id, (x, y, w, h), obj_conf]
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors, n_cls]

        # calculate targets
        # skip bbox with class `self.ignore_label`
        non_zero_masks = (labels.sum(dim=2) > 0)
        valid_masks = (labels[:, :, 0] != self.ignore_label)
        nlabel = (non_zero_masks & valid_masks).sum(dim=1)  # number of objects
        nlabel_w_ignore = non_zero_masks.sum(dim=1)  # valid + ignore objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors]
        expanded_strides = torch.cat(expanded_strides, 1)  # [1, n_anchors]
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)  # [B, n_anchors, 4]

        gt_cls_idxs = []
        cls_targets = []
        reg_targets = []
        l1_targets = []
        bbox_ws = []
        obj_targets = []
        fg_masks = []
        ignore_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = nlabel[batch_idx].item()
            num_gts += num_gt
            if num_gt == 0:
                gt_matched_classes = outputs.new_zeros(0)
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                bbox_w = outputs.new_zeros(0)
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                if labels[batch_idx].sum() == 0:  # no label, all as background
                    ignore_mask = outputs.new_zeros(total_num_anchors).bool()
                else:  # has bbox, but are all with ignore_label
                    num_ignore = (labels[batch_idx, :, 0] == self.ignore_label).sum().item()
                    ignore_bboxes = labels[batch_idx, :num_ignore, 1:5]
                    ignore_mask, _ = self.get_geometry_constraint(
                        ignore_bboxes, expanded_strides, x_shifts, y_shifts)
                    # no loss on anchors covered by ignore_bboxes, i.e. `ignore_mask`
                    # for other regions, treat them as background
            else:
                num_gt_w_ignore = nlabel_w_ignore[batch_idx].item()
                valid_mask = valid_masks[batch_idx, :num_gt_w_ignore]  # [n]
                gt_bboxes_per_image = labels[batch_idx, :num_gt_w_ignore, 1:5]  # [n, 4]
                gt_classes = labels[batch_idx, :num_gt_w_ignore, 0]  # [n]
                gt_obj_conf = labels[batch_idx, :num_gt_w_ignore, 5]  # [n]
                gt_cls_conf = labels[batch_idx, :num_gt_w_ignore, 6]  # [n]
                bboxes_preds_per_image = bbox_preds[batch_idx]  # [n_anchors,4]

                try:
                    (
                        gt_matched_classes,  # [n_pos_matched]
                        fg_mask,  # [n_anchors], anchors that are positive
                        pred_ious_this_matching,  # [n_pos_matched]
                        matched_gt_inds,  # [n_pos_matched]
                        num_fg_img,
                        ignore_mask,  # [n_anchors], anchors that are ignored
                    ) = self.get_assignments_w_ignore(  # noqa
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds[batch_idx],
                        obj_preds[batch_idx],
                        valid_mask,
                    )
                except RuntimeError as e:
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise

                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                        ignore_mask,
                    ) = self.get_assignments_w_ignore(  # noqa
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds[batch_idx],
                        obj_preds[batch_idx],
                        valid_mask,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[valid_mask][matched_gt_inds]
                bbox_w = self._get_bbox_loss_weight(
                    matched_gt_inds, gt_obj_conf[valid_mask], gt_cls_conf[valid_mask])

                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[valid_mask][matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            gt_cls_idxs.append(gt_matched_classes.long())
            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            bbox_ws.append(bbox_w)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            ignore_masks.append(ignore_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        gt_cls_idxs = torch.cat(gt_cls_idxs, 0)
        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        if self.bbox_loss_weighting:
            bbox_ws = torch.cat(bbox_ws, 0)
            bbox_ws = bbox_ws / bbox_ws.mean()  # normalize to mean=1
            cls_bbox_ws = bbox_ws[:, None]
        else:
            bbox_ws = cls_bbox_ws = 1.
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)  # [batch*n_anchors]
        ignore_masks = torch.cat(ignore_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets, weights=bbox_ws)
        # ignore some anchors when computing objectness (fg-bg) loss
        valid_masks = (~ignore_masks)
        loss_obj = (
            self.obj_loss_fn(
                obj_preds.view(-1, 1)[valid_masks], obj_targets[valid_masks])
        ).sum() / num_fg
        loss_cls = (
            self.cls_loss_fn(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            ) * cls_bbox_ws
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets) * bbox_ws
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        loss_iou = self.reg_weight * loss_iou
        loss_obj = self.obj_weight * loss_obj
        loss_cls = self.cls_weight * loss_cls
        loss = loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    @torch.no_grad()
    def get_assignments_w_ignore(
        self,
        num_gt,
        gt_bboxes_per_image,  # [n, 4]
        gt_classes,  # [n]
        bboxes_preds_per_image,  # [n_anchors, 4]
        expanded_strides,  # [1, n_anchors]
        x_shifts,  # [1, n_anchors]
        y_shifts,  # [1, n_anchors]
        cls_preds,  # [n_anchors, n_cls]
        obj_preds,  # [n_anchors, 1]
        valid_mask,  # [n], True --> valid gt_bbox
        mode="gpu",
    ):
        if valid_mask.all():
            gt_matched_classes, fg_mask, pred_ious_this_matching, \
                matched_gt_inds, num_fg = self.get_assignments(
                    num_gt,
                    gt_bboxes_per_image,
                    gt_classes,
                    bboxes_preds_per_image,
                    expanded_strides,
                    x_shifts,
                    y_shifts,
                    cls_preds,
                    obj_preds,
                    mode,
                )
            # ignore_mask is just all False
            ignore_mask = torch.zeros_like(fg_mask)
            return gt_matched_classes, fg_mask, pred_ious_this_matching, \
                matched_gt_inds, num_fg, ignore_mask

        if mode == "cpu":
            print("-----------Using CPU for the Current Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            valid_mask = valid_mask.cpu()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, geometry_relation, ignore_mask = self.get_geometry_constraint_w_ignore(
            valid_mask,
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
        )
        # fg_mask: [n_anchors] boolean, if the anchor is positive for any gt
        #          ignored anchors are already set to False
        # geometry_relation: [n_valid, n_pos]
        # ignore_mask: [n_anchors] boolean, ignore anchor in loss computation

        # ignore bbox with ignore_label in bbox-related label assignment
        gt_classes = gt_classes[valid_mask]  # [n_valid]
        gt_bboxes_per_image = gt_bboxes_per_image[valid_mask]  # [n_valid, 4]

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]  # [n_pos, 4]
        cls_preds_ = cls_preds[fg_mask]  # [n_pos, n_cls]
        obj_preds_ = obj_preds[fg_mask]  # [n_pos, 1]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        # pair_wise_ious: [n_valid, n_pos], pairwise bbox IoUs

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
        )  # [n_valid, n_cls]
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)  # [n_valid, n_pos]

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().sigmoid_() * obj_preds_.float().sigmoid_()
            ).sqrt()  # [n_pos, n_cls]
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.unsqueeze(0).repeat(num_gt, 1, 1),
                gt_cls_per_image.unsqueeze(1).repeat(1, num_in_boxes_anchor, 1),
                reduction="none"
            ).sum(-1)  # [n_valid, n_pos]
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + float(1e6) * (~geometry_relation)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.simota_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss
        # all are [n_pos_matched]

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,  # [n_pos_matched]
            fg_mask,  # [n_anchors], bool, anchor matched to any gt
            pred_ious_this_matching,  # [n_pos_matched], IoU of the match
            matched_gt_inds,  # [n_pos_matched], index of the matched gt
            num_fg,  # int, number of fg anchors
            ignore_mask,  # [n_anchors], bool, anchor should be ignored
        )

    def get_geometry_constraint_w_ignore(
        self, valid_mask, gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts,
    ):
        """
        Calculate whether the center of an object is located in a fixed range of
        an anchor. This is used to avert inappropriate matching. It can also reduce
        the number of candidate anchors so that the GPU memory is saved.
        """
        # valid_mask: [n], True --> valid bbox
        # gt_bboxes_per_image: [n, 4]
        # others: [1, n_anchors]

        assert not valid_mask.all(), 'Should contain at least one ignore bbox'
        assert valid_mask.any(), 'Should contain at least one valid bbox'
        # the easiest way: compute constraints with and wo ignore_label
        # then take the intersection of the two as fg_mask and fg_anchor
        # anchor_filter_, _ = self.get_geometry_constraint(gt_bboxes_per_image, expanded_strides, x_shifts, y_shifts)
        # gt_bboxes_valid = gt_bboxes_per_image[valid_mask]
        # anchor_filter_valid_, geometry_relation_ = self.get_geometry_constraint(gt_bboxes_valid, expanded_strides, x_shifts, y_shifts)
        # ignore_mask_ = (anchor_filter_ & (~anchor_filter_valid_))
        # return anchor_filter_valid_, geometry_relation_, ignore_mask_

        # a better way is to compute constraints only once
        # and then look at is_in_centers: if only is_in_centers because of bbox
        #   with ignore_label, then its region should be ignored
        x_centers_per_image = (x_shifts + 0.5) * expanded_strides  # [1, n_anchors]
        y_centers_per_image = (y_shifts + 0.5) * expanded_strides  # [1, n_anchors]

        # in fixed center
        center_radius = 1.5
        center_dist = expanded_strides * center_radius  # [1, n_anchors]
        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0:1]) - center_dist  # [n, n_anchors]
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0:1]) + center_dist
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1:2]) - center_dist
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1:2]) + center_dist

        c_l = x_centers_per_image - gt_bboxes_per_image_l  # [n, n_anchors]
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)  # [n, n_anchors, 4]
        is_in_centers = center_deltas.min(dim=-1).values > 0.0  # [n, n_anchors]
        anchor_filter = is_in_centers.sum(dim=0) > 0  # [n_anchors], is positive for any gt

        anchor_filter_valid = (is_in_centers[valid_mask]).sum(dim=0) > 0
        ignore_mask = (anchor_filter & (~anchor_filter_valid))
        anchor_filter[ignore_mask] = False
        num_valid = valid_mask.sum().item()
        valid_mask = valid_mask[:, None] & anchor_filter[None, :]  # [n_valid, n_anchors]
        geometry_relation = is_in_centers[valid_mask].view(num_valid, -1)  # [n_valid, n_pos]
        # assert (anchor_filter == anchor_filter_valid_).all() and (geometry_relation_ == geometry_relation).all() and (ignore_mask_ == ignore_mask).all()

        return anchor_filter, geometry_relation, ignore_mask
