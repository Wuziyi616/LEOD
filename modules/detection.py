from typing import Any, Optional, Tuple, Dict
from warnings import warn

from tqdm import trange
import numpy as np
import pytorch_lightning as pl
import torch
import torch as th
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT

from data.genx_utils.labels import ObjectLabels
from data.utils.types import DataType, LstmStates, ObjDetOutput, DatasetSamplingMode
from models.detection.yolox.utils.boxes import postprocess
from models.detection.yolox_extension.models.detector import YoloXDetector
from utils.evaluation.prophesee.evaluator import PropheseeEvaluator
from utils.evaluation.prophesee.io.box_loading import to_prophesee
from utils.padding import InputPadderFromShape
from .utils.detection import BackboneFeatureSelector, EventReprSelector, Mode, \
    RNNStates, mode_2_string, merge_mixed_batches, WORKER_ID_KEY, DATA_KEY
from .utils.ssod import get_subsample_label_idx


class Module(pl.LightningModule):
    """Base model for event detection with:
    - A recurrent backbone extracting features from event repr.
        Here we use a recurrent ViT.
    - A detection head predicting bounding boxes from the features.
        Here we use a YOLOX detector.
    """

    def __init__(self, full_config: DictConfig, ssod: bool = False):
        super().__init__()

        self.full_config = full_config

        self.mdl_config = full_config.model
        self.num_classes = self.mdl_config.head.num_classes
        in_res_hw = tuple(self.mdl_config.backbone.in_res_hw)
        self.input_padder = InputPadderFromShape(desired_hw=in_res_hw)

        self.mdl = YoloXDetector(self.mdl_config, ssod=ssod)
        if self.mdl_config.backbone.compile.enable:
            self.mdl.backbone = torch.compile(self.mdl.backbone)

        self.dst_config = full_config.dataset
        self.label_subsample_idx = get_subsample_label_idx(
            L=self.dst_config.sequence_length,  # L
            use_every=self.mdl_config.get('use_label_every', 1))

        self.mode_2_rnn_states: Dict[Mode, RNNStates] = {
            Mode.TRAIN: RNNStates(),
            Mode.VAL: RNNStates(),
            Mode.TEST: RNNStates(),
        }

    def setup(self, stage: Optional[str] = None) -> None:
        dst_cfg = self.full_config.dataset
        dataset_name = dst_cfg.name
        self.mode_2_hw: Dict[Mode, Tuple[int, int]] = {}
        self.mode_2_batch_size: Dict[Mode, int] = {}
        self.mode_2_psee_evaluator: Dict[Mode, PropheseeEvaluator] = {}
        self.mode_2_sampling_mode: Dict[Mode, DatasetSamplingMode] = {}

        self.started_training = True

        dataset_train_sampling = dst_cfg.train.sampling
        dataset_eval_sampling = dst_cfg.eval.sampling
        assert dataset_train_sampling in iter(DatasetSamplingMode)
        assert dataset_eval_sampling in \
            (DatasetSamplingMode.STREAM, DatasetSamplingMode.RANDOM)
        if stage == 'fit':  # train + val
            self.train_config = self.full_config.training
            self.train_metrics_config = self.full_config.logging.train.metrics

            if self.train_metrics_config.compute:
                self.train_eval_every = \
                    self.train_metrics_config.detection_metrics_every_n_steps
                self.mode_2_psee_evaluator[Mode.TRAIN] = PropheseeEvaluator(
                    dataset=dataset_name,
                    downsample_by_2=dst_cfg.downsample_by_factor_2)
            self.mode_2_psee_evaluator[Mode.VAL] = PropheseeEvaluator(
                dataset=dataset_name,
                downsample_by_2=dst_cfg.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.TRAIN] = dataset_train_sampling
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling

            vis_config = self.full_config.logging.train.high_dim
            self.train_vis_every = vis_config.every_n_steps if \
                vis_config.enable else int(1e9)

            for mode in (Mode.TRAIN, Mode.VAL):
                self.mode_2_hw[mode] = None
                self.mode_2_batch_size[mode] = None
            self.started_training = False
        elif stage == 'validate':
            mode = Mode.VAL
            self.mode_2_psee_evaluator[mode] = PropheseeEvaluator(
                dataset=dataset_name,
                downsample_by_2=dst_cfg.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        elif stage in ['test', 'predict']:
            mode = Mode.TEST
            self.mode_2_psee_evaluator[mode] = PropheseeEvaluator(
                dataset=dataset_name,
                downsample_by_2=dst_cfg.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.TEST] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        else:
            raise NotImplementedError(f"Stage {stage} not implemented.")

    def forward(self,
                event_tensor: th.Tensor,
                previous_states: Optional[LstmStates] = None,
                retrieve_detections: bool = True,
                targets=None) -> \
            Tuple[th.Tensor, Dict[str, th.Tensor], LstmStates]:
        return self.mdl(x=event_tensor,
                        previous_states=previous_states,
                        retrieve_detections=retrieve_detections,
                        targets=targets)

    def get_worker_id_from_batch(self, batch: Any) -> int:
        return batch[WORKER_ID_KEY]

    def get_data_from_batch(self, batch: Any):
        data = batch[DATA_KEY]
        # pad ev_repr to desired HxW here
        ev_repr = torch.stack(data[DataType.EV_REPR]).to(dtype=self.dtype)
        # [L, B, C, H, W], event reprs
        padded_ev_repr = self.input_padder.pad_tensor_ev_repr(ev_repr)
        data[DataType.EV_REPR] = [ev for ev in padded_ev_repr]  # back to list
        if not self.training:
            return data
        # sub-sample labels to speed up training
        # per-frame pseudo labels contain redundant information and doesn't
        #   help model performance, but slows down training
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        for tidx in range(len(sparse_obj_labels)):
            if tidx in self.label_subsample_idx:
                continue
            # set labels as None, unless it's GT label
            sparse_obj_labels[tidx].set_non_gt_labels_to_none_()
        data[DataType.OBJLABELS_SEQ] = sparse_obj_labels
        return data

    def training_step(self, batch: Any,
                      batch_idx: int, log: bool = True) -> STEP_OUTPUT:
        batch = merge_mixed_batches(batch)
        data = self.get_data_from_batch(batch)  # event reprs, bbox labels, ...
        worker_id = self.get_worker_id_from_batch(batch)  # int
        # all data in this batch is loaded from one worker

        mode = Mode.TRAIN
        self.started_training = True
        step = self.trainer.global_step
        ev_tensor_sequence = data[DataType.EV_REPR]
        # a `L`-len list, each [B, C, H, W], event reprs
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        # a `L`-len list of `SparselyBatchedObjectLabels`
        #   each contains a `B`-len list of `ObjectLabels` or None
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        # [B], bool. Will reset RNN state if True, i.e. the first sample of seq
        token_mask_sequence = data.get(DataType.TOKEN_MASK, None)
        # should be None as we don't pad inputs in the dataloaders

        self.mode_2_rnn_states[mode].reset(
            worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        L = len(ev_tensor_sequence)
        assert L > 0
        B = len(sparse_obj_labels[0])
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = B
        else:
            assert self.mode_2_batch_size[mode] == B

        prev_states = \
            self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        # store backbone features at labeled frames to apply detection head
        backbone_feature_selector = BackboneFeatureSelector()
        # visualization purpose
        ev_repr_selector = EventReprSelector()
        obj_labels = list()
        for tidx in range(L):
            ev_tensors = ev_tensor_sequence[tidx]  # [B, C, H, W]
            if token_mask_sequence is not None:
                token_masks = self.input_padder.pad_token_mask(
                    token_mask=token_mask_sequence[tidx])
            else:
                token_masks = None

            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            backbone_features, states = self.mdl.forward_backbone(
                x=ev_tensors,
                previous_states=prev_states,
                token_mask=token_masks)
            # `backbone_features`: dict{stage_id: feats, [B, C, h, w]}
            # `states`: list[(lstm_h, lstm_c), same shape]
            prev_states = states

            current_labels, valid_batch_indices = \
                sparse_obj_labels[tidx].get_valid_labels_and_batch_indices(
                    ignore=self.mdl_config.get('ignore_image', False),
                    ignore_label=self.mdl_config.head.get('ignore_label', 1024))
            # `current_labels`: list[ObjectLabels], length is num_valid_frames
            # `valid_batch_indices`: list[int], idx of valid frames in a batch

            # Store backbone features that correspond to the available labels.
            # also store event reprs for labeled frames
            if len(current_labels) > 0:
                backbone_feature_selector.add_backbone_features(
                    backbone_features=backbone_features,
                    selected_indices=valid_batch_indices)
                obj_labels.extend(current_labels)
                ev_repr_selector.add_ev_repr(
                    ev_repr=ev_tensors, selected_indices=valid_batch_indices)

        # update RNN states for the next batch
        self.mode_2_rnn_states[mode].save_states_and_detach(
            worker_id=worker_id, states=prev_states)
        assert len(obj_labels) > 0

        # Batch the backbone features and labels to parallelize the detection.
        # backbone features are now dict: {stage_id: [B, C, h, w]}
        selected_backbone_features = \
            backbone_feature_selector.get_batched_backbone_features()
        # labels_yolox: [B, N, 7]; padded to a fixed `N` dim
        #   7: [cls_id, (xywh), obj_conf, cls_conf]
        labels_yolox = ObjectLabels.get_labels_as_batched_tensor(
            obj_label_list=obj_labels, format_='yolox')
        labels_yolox = labels_yolox.to(dtype=self.dtype)

        # predictions: (B, N, 4 + 1 + num_cls), [(x, y, w, h), obj_conf, cls]
        predictions, losses = self.mdl.forward_detect(
            backbone_features=selected_backbone_features, targets=labels_yolox)
        assert losses is not None
        assert 'loss' in losses
        output = {'loss': losses['loss']}

        if self.mode_2_sampling_mode[mode] in \
                (DatasetSamplingMode.MIXED, DatasetSamplingMode.RANDOM):
            # We only want to evaluate the last batch_size samples here.
            # Otherwise we will mostly evaluate the init phase of the sequence.
            predictions = predictions[-B:]
            obj_labels = obj_labels[-B:]

        # prepare for visualizing detection results
        if step % self.train_vis_every == 0:
            # pred_processed: `B`-len List[(N_i, 7)], [xyxy, obj, cls, cls_id]
            pred_processed = postprocess(
                prediction=predictions,
                num_classes=self.mdl_config.head.num_classes,
                conf_thre=self.mdl_config.postprocess.confidence_threshold,
                nms_thre=self.mdl_config.postprocess.nms_threshold)

            # make both results [N, 7], each: [t, (xywh), cls_idx, cls]
            loaded_labels_proph, yolox_preds_proph = \
                to_prophesee(obj_labels, pred_processed)

            # For visualization, we only use the last batch_size items.
            output.update({
                ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-B:],
                ObjDetOutput.PRED_PROPH: yolox_preds_proph[-B:],
                ObjDetOutput.EV_REPR: ev_repr_selector.get_ev_repr_as_list(start_idx=-B),
                ObjDetOutput.SKIP_VIZ: False,
            })

        # Logging
        prefix = f'{mode_2_string[mode]}/'
        log_dict = {f'{prefix}{k}': v for k, v in losses.items()}
        if log:
            self.log_dict(
                log_dict,
                on_step=True,
                on_epoch=True,
                batch_size=B,
                sync_dist=False,
                rank_zero_only=True)
        else:
            log_dict['batch_size'] = B
            output['log_dict'] = log_dict

        if mode in self.mode_2_psee_evaluator and log:
            self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
            self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)
            if self.train_eval_every is not None and \
                    step > 0 and step % self.train_eval_every == 0:
                self.run_psee_evaluator(mode=mode)

        return output

    def _val_test_step_impl(self, batch: Any, mode: Mode) -> STEP_OUTPUT:
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        assert mode in (Mode.VAL, Mode.TEST)
        assert self.mode_2_sampling_mode[mode] == DatasetSamplingMode.STREAM, \
            'Should always test on streaming mode event sequences'
        ev_tensor_sequence = data[DataType.EV_REPR]
        # a `L`-len list, each [B, C, H, W], event reprs
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        # a `L`-len list of `SparselyBatchedObjectLabels`
        #   each contains a `B`-len list of `ObjectLabels` or None
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        # [B], bool. Will reset RNN state if True, i.e. the first sample of seq

        self.mode_2_rnn_states[mode].reset(
            worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        L = len(ev_tensor_sequence)
        assert L > 0
        B = len(sparse_obj_labels[0])
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = B
        else:
            assert self.mode_2_batch_size[mode] == B

        prev_states = \
            self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        # store backbone features at labeled frames to apply detection head
        backbone_feature_selector = BackboneFeatureSelector()
        # visualization purpose
        ev_repr_selector = EventReprSelector()
        obj_labels = list()
        for tidx in range(L):
            ev_tensors = ev_tensor_sequence[tidx]
            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            backbone_features, states = self.mdl.forward_backbone(
                x=ev_tensors, previous_states=prev_states)
            # `backbone_features`: dict{stage_id: feats, [B, C, h, w]}
            # `states`: list[(lstm_h, lstm_c), same shape]
            prev_states = states

            # in streaming, we load the entire event seq only once
            #   so have to predict for all labeled frames in-between
            current_labels, valid_batch_indices = \
                sparse_obj_labels[tidx].get_valid_labels_and_batch_indices()
            # `current_labels`: list[ObjectLabels], length is num_valid_frames
            # `valid_batch_indices`: list[int], idx of valid frames in a batch
            # Store backbone features that correspond to the available labels.
            if len(current_labels) > 0:
                backbone_feature_selector.add_backbone_features(
                    backbone_features=backbone_features,
                    selected_indices=valid_batch_indices)

                obj_labels.extend(current_labels)
                ev_repr_selector.add_ev_repr(
                    ev_repr=ev_tensors, selected_indices=valid_batch_indices)

        self.mode_2_rnn_states[mode].save_states_and_detach(
            worker_id=worker_id, states=prev_states)

        if len(obj_labels) == 0:
            return {ObjDetOutput.SKIP_VIZ: True}

        # backbone features are now dict: {stage_id: [B, C, h, w]}
        selected_backbone_features = \
            backbone_feature_selector.get_batched_backbone_features()

        # predictions: (B, N, 4 + 1 + num_cls), [(x, y, w, h), obj_conf, cls]
        predictions, _ = self.mdl.forward_detect(
            backbone_features=selected_backbone_features)

        # pred_processed: `B`-len List[(N_i, 7)], [(xyxy), obj, cls, cls_idx]
        pred_processed = postprocess(
            prediction=predictions,
            num_classes=self.mdl_config.head.num_classes,
            conf_thre=self.mdl_config.postprocess.confidence_threshold,
            nms_thre=self.mdl_config.postprocess.nms_threshold)

        # process and make both results [N, 7], each: [t, (xywh), cls_idx, cls]
        # actual keys are ['t','x','y','w','h','class_id','class_confidence']
        # the (x, y) here is the **top-left corner** of the bbox
        loaded_labels_proph, yolox_preds_proph = \
            to_prophesee(obj_labels, pred_processed)

        # For visualization, we only use the last item (per batch).
        output = {
            ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-1],
            ObjDetOutput.PRED_PROPH: yolox_preds_proph[-1],
            ObjDetOutput.EV_REPR: ev_repr_selector.get_ev_repr_as_list(start_idx=-1)[0],
            ObjDetOutput.SKIP_VIZ: False,
        }

        if self.started_training:
            self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
            self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)

        return output

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        return self._val_test_step_impl(batch=batch, mode=Mode.VAL)

    def test_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        return self._val_test_step_impl(batch=batch, mode=Mode.TEST)

    def run_psee_evaluator(self, mode: Mode, log: bool = True,
                           reset_buffer: bool = True,
                           ret_pr_curve: bool = False):
        if mode == Mode.VAL:
            assert reset_buffer, 'Not reseting evaluator in validation'
        psee_evaluator = self.mode_2_psee_evaluator[mode]
        batch_size = self.mode_2_batch_size[mode]
        hw_tuple = self.mode_2_hw[mode]
        if psee_evaluator is None:
            warn(f'{mode=} psee_evaluator is None', UserWarning, stacklevel=2)
            return
        assert batch_size is not None
        assert hw_tuple is not None
        if psee_evaluator.has_data():
            metrics = psee_evaluator.evaluate_buffer(img_height=hw_tuple[0],
                                                     img_width=hw_tuple[1],
                                                     ret_pr_curve=ret_pr_curve)
            assert metrics is not None
            if reset_buffer:
                psee_evaluator.reset_buffer()
            if ret_pr_curve:
                pr_keys = [k for k in metrics.keys() if 'PR' in k]
                pr_curves = {k: metrics.pop(k) for k in pr_keys}

            prefix = f'{mode_2_string[mode]}/'
            log_dict = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    value = torch.tensor(v)
                elif isinstance(v, np.ndarray):
                    value = torch.from_numpy(v)
                elif isinstance(v, torch.Tensor):
                    value = v
                else:
                    raise NotImplementedError
                assert value.ndim == 0, f'tensor must be a scalar.\n{v=}\n' \
                    f'{type(v)=}\n{value=}\n{type(value)=}'
                # put them on the current device to avoid this error:
                #   https://github.com/Lightning-AI/lightning/discussions/2529
                log_dict[f'{prefix}{k}'] = value.to(self.device)
            # Somehow self.log does not work when we eval in training epoch.
            if log:
                self.log_dict(
                    log_dict,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size,
                    sync_dist=True)
            if not log and not ret_pr_curve:
                log_dict['batch_size'] = batch_size
                return log_dict
            if ret_pr_curve:
                return pr_curves
        else:
            warn(f'{mode=} psee_evaluator no data', UserWarning, stacklevel=2)

    def on_train_epoch_end(self) -> None:
        mode = Mode.TRAIN
        if mode in self.mode_2_psee_evaluator and \
                self.train_eval_every is None and \
                self.mode_2_hw[mode] is not None:
            # For some reason PL calls this function when resuming.
            # We don't know yet the value of train_hw, so we skip this
            self.run_psee_evaluator(mode=mode)

    def on_validation_epoch_end(self) -> None:
        mode = Mode.VAL
        if self.started_training:
            assert self.mode_2_psee_evaluator[mode].has_data()
            self.run_psee_evaluator(mode=mode)

    def on_test_epoch_end(self) -> None:
        mode = Mode.TEST
        assert self.mode_2_psee_evaluator[mode].has_data()
        self.run_psee_evaluator(mode=mode, reset_buffer=False)

    def configure_optimizers(self) -> Any:
        lr = self.train_config.learning_rate
        weight_decay = self.train_config.weight_decay
        optimizer = th.optim.AdamW(self.mdl.parameters(), lr=lr,
                                   weight_decay=weight_decay)

        scheduler_params = self.train_config.lr_scheduler
        if not scheduler_params.use:
            return optimizer

        total_steps = scheduler_params.total_steps
        assert total_steps is not None
        assert total_steps > 0
        # Here we interpret the final lr as max_lr/final_div_factor.
        final_div_factor_pytorch = \
            scheduler_params.final_div_factor / scheduler_params.div_factor
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            div_factor=scheduler_params.div_factor,
            final_div_factor=final_div_factor_pytorch,
            total_steps=total_steps,
            pct_start=scheduler_params.pct_start,
            cycle_momentum=False,
            anneal_strategy='linear')
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": 'learning_rate',
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def predict_one_seq(self, batch: Any) -> Optional[STEP_OUTPUT]:
        """Run model over one full event sequence."""
        mode = Mode.TEST
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        ev_tensor_sequence = torch.stack(data[DataType.EV_REPR], dim=0).cuda()
        # a `L`-len list, each [B(==1), C, H, W], event reprs
        # make them [L, 1, C, H, W]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        # a `L`-len list of `SparselyBatchedObjectLabels`
        #   each contains a `B`-len list of `ObjectLabels` or None
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]
        # [1], bool. Will reset RNN state if True, i.e. start of a new seq
        assert ev_tensor_sequence.shape[1] == len(sparse_obj_labels[0]) == \
            is_first_sample.shape[0] == 1

        self.mode_2_rnn_states[mode].reset(
            worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        L = len(ev_tensor_sequence)
        assert L > 0

        prev_states, all_backbone_feats, all_preds, BS = None, [], [], 128
        for tidx in trange(L, desc='Predict'):
            ev_tensors = ev_tensor_sequence[tidx]  # [B(==1), C, H, W]
            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            backbone_features, states = self.mdl.forward_backbone(
                x=ev_tensors, previous_states=prev_states)
            # `backbone_features`: dict{stage_id (1/2/3/4): feats, [1,C,h,w]}
            prev_states = states
            all_backbone_feats.append(backbone_features)

            # we perform prediction for a batch of timesteps
            if (tidx + 1) % BS == 0 or tidx == L - 1:
                backbone_features = {
                    k: torch.cat([feats[k] for feats in all_backbone_feats])
                    for k in backbone_features.keys()
                }
                # predictions: (B, N, 4 + 1 + num_cls), [(x,y,w,h),obj,cls]
                predictions, _ = self.mdl.forward_detect(
                    backbone_features=backbone_features)

                # pred_processed: List[(N_i, 7)], [(xyxy), obj, cls, cls_idx]
                pred_processed = postprocess(
                    prediction=predictions,
                    num_classes=self.mdl_config.head.num_classes,
                    conf_thre=self.mdl_config.postprocess.confidence_threshold,
                    nms_thre=self.mdl_config.postprocess.nms_threshold)
                all_preds.extend(pred_processed)
                all_backbone_feats = []

        # all_preds: a `L`-len list, each is [N, 7], i.e. bbox at each timestep
        # ev_seq: [L, C, H, W], grid-like event reprs
        # all_lbl: a `L`-len list of `ObjectLabels` or None
        ev_seq = ev_tensor_sequence[:, 0]
        all_lbl = [lbl[0] for lbl in sparse_obj_labels]
        return all_preds, ev_seq, all_lbl

    def load_weight(self, ckpt_path: str, strict: bool = True) -> None:
        """Load checkpoint from a file.

        Args:
            ckp_path (str): Path to checkpoint file.
            strict (bool, optional): Whether to allow different params for
                the model and checkpoint. Defaults to True.
        """
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        self.load_state_dict(ckpt, strict=strict)
