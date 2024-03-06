from typing import Any, Tuple, List, Optional, Dict
from warnings import warn

import numpy as np

from utils.evaluation.prophesee.evaluation import evaluate_list

LABELMAP = {
    'gen1': ('car', 'ped'),
    'gen4': ('ped', 'cyc', 'car'),
}


def get_labelmap(dst_name: str = None, num_cls: int = None) -> Tuple[str]:
    assert dst_name is None or num_cls is None
    if dst_name is not None:
        return LABELMAP[dst_name.lower()]
    elif num_cls is not None:
        assert num_cls in (2, 3), f'Invalid number of classes: {num_cls}'
        return LABELMAP['gen1'] if num_cls == 2 else LABELMAP['gen4']
    else:
        raise NotImplementedError('Either dst_name or num_cls must be input')


class PropheseeEvaluator:
    LABELS = 'lables'
    PREDICTIONS = 'predictions'

    def __init__(self, dataset: str, downsample_by_2: bool):
        super().__init__()
        assert dataset in {'gen1', 'gen4'}
        self.dataset = dataset
        self.label_map = get_labelmap(dataset)
        self.downsample_by_2 = downsample_by_2

        self._buffer = None
        self._buffer_empty = True
        self._reset_buffer()

    def _reset_buffer(self):
        self._buffer_empty = True
        self._buffer = {
            self.LABELS: list(),
            self.PREDICTIONS: list(),
        }

    def _add_to_buffer(self, key: str, value: List[np.ndarray]):
        assert isinstance(value, list)
        for entry in value:
            assert isinstance(entry, np.ndarray)
        self._buffer_empty = False
        assert self._buffer is not None
        self._buffer[key].extend(value)

    def _get_from_buffer(self, key: str) -> List[np.ndarray]:
        assert not self._buffer_empty
        assert self._buffer is not None
        return self._buffer[key]

    def add_predictions(self, predictions: List[np.ndarray]):
        self._add_to_buffer(self.PREDICTIONS, predictions)

    def add_labels(self, labels: List[np.ndarray]):
        self._add_to_buffer(self.LABELS, labels)

    def reset_buffer(self) -> None:
        # E.g. call in on_validation_epoch_start
        self._reset_buffer()

    def has_data(self):
        return not self._buffer_empty

    def evaluate_buffer(self, img_height: int, img_width: int, ret_pr_curve: bool = False) -> Optional[Dict[str, Any]]:
        # e.g call in on_validation_epoch_end
        if self._buffer_empty:
            warn("Attempt to use prophesee evaluation buffer, but it is empty", UserWarning, stacklevel=2)
            return

        labels = self._get_from_buffer(self.LABELS)
        predictions = self._get_from_buffer(self.PREDICTIONS)
        assert len(labels) == len(predictions)

        # we perform both per-category and overall evaluation
        # overall
        metrics = evaluate_list(result_boxes_list=predictions,
                                gt_boxes_list=labels,
                                height=img_height,
                                width=img_width,
                                apply_bbox_filters=True,
                                downsampled_by_2=self.downsample_by_2,
                                camera=self.dataset)
        # per-category
        for cls_id, cls_name in enumerate(self.label_map):
            lbls = [lbl[lbl['class_id'] == cls_id] for lbl in labels]
            preds = [pred[pred['class_id'] == cls_id] for pred in predictions]
            cls_metric = evaluate_list(result_boxes_list=preds,
                                       gt_boxes_list=lbls,
                                       height=img_height,
                                       width=img_width,
                                       apply_bbox_filters=True,
                                       downsampled_by_2=self.downsample_by_2,
                                       camera=self.dataset)
            cls_metric = {f'{k}_{cls_name}': v for k, v in cls_metric.items()}
            metrics.update(cls_metric)

        if not ret_pr_curve:
            del_keys = [k for k in metrics.keys() if 'PR' in k]
            for k in del_keys:
                del metrics[k]
        return metrics
