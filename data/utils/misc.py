from typing import Tuple, List

import os

import h5py
import numpy as np

from data.genx_utils.labels import ObjectLabelFactory, ObjectLabels


def get_labels_npz_fn(seq_dir: str) -> str:
    """Get labels npz file name."""
    # seq_dir: path/to/dataset/train/18-03-29_13-15-02_5_605
    # labels_npz_fn: .../18-03-29_13-15-02_5_605/labels_v2/labels.npz
    labels_npz_fn = os.path.join(seq_dir, 'labels_v2/labels.npz')
    return labels_npz_fn


def read_npz_labels(label_fn: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read labels from npz file."""
    # if it's just .../18-03-29_13-15-02_5_605, we get the true npz first
    if 'labels_v2' not in label_fn:
        label_fn = get_labels_npz_fn(label_fn)
    labels = np.load(label_fn)
    return labels['labels'], labels['objframe_idx_2_label_idx']


def read_labels_as_list(seq_dir: str, dst_cfg,
                        L: int, start_idx: int = 0) -> List[ObjectLabels]:
    """Read the original full-frequency GT labels."""
    # seq_dir: .../18-03-29_13-15-02_5_605
    labels, objframe_idx_2_label_idx = read_npz_labels(seq_dir)
    hw = tuple(dst_cfg.ev_repr_hw)
    if dst_cfg.downsample_by_factor_2:
        hw = tuple(s * 2 for s in hw)
    label_factory = ObjectLabelFactory.from_structured_array(
        labels, objframe_idx_2_label_idx, hw,
        2 if dst_cfg.downsample_by_factor_2 else None)
    ev_dir = get_ev_dir(seq_dir)
    objframe_idx_2_repr_idx = np.load(
        os.path.join(ev_dir, 'objframe_idx_2_repr_idx.npy'))
    obj_labels = [None] * L
    for objframe_idx, repr_idx in enumerate(objframe_idx_2_repr_idx):
        if start_idx <= repr_idx < start_idx + L:
            obj_labels[repr_idx - start_idx] = label_factory[objframe_idx]
    return obj_labels


def get_ev_dir(seq_dir: str) -> str:
    """Get event representation directory."""
    # seq_dir: path/to/dataset/train/18-03-29_13-15-02_5_605
    # ev_dir: .../18-03-29_13-15-02_5_605/event_representations_v2/stacked_histogram_dt=50_nbins=10/
    ev_dir = os.path.join(seq_dir, 'event_representations_v2',
                          'stacked_histogram_dt=50_nbins=10')
    return ev_dir


def get_objframe_idx_2_repr_idx_fn(ev_dir: str) -> str:
    """Get xxx/objframe_idx_2_repr_idx.npy file name."""
    if 'event_representations_v2' not in ev_dir:
        ev_dir = get_ev_dir(ev_dir)
    # ev_dir: .../18-03-29_13-15-02_5_605/event_representations_v2/stacked_histogram_dt=50_nbins=10/
    # fn: .../18-03-29_13-15-02_5_605/event_representations_v2/stacked_histogram_dt=50_nbins=10/objframe_idx_2_repr_idx.npy
    fn = os.path.join(ev_dir, 'objframe_idx_2_repr_idx.npy')
    return fn


def get_ev_h5_fn(ev_dir: str, dst_name: str = None) -> str:
    """Get event representation h5 file name."""
    # if it's just .../18-03-29_13-15-02_5_605, we get the true ev_dir first
    if 'event_representations_v2' not in ev_dir:
        ev_dir = get_ev_dir(ev_dir)
    # ev_dir: .../18-03-29_13-15-02_5_605/event_representations_v2/stacked_histogram_dt=50_nbins=10/
    # ev_h5_fn: .../18-03-29_13-15-02_5_605/event_representations_v2/stacked_histogram_dt=50_nbins=10/event_representations.h5
    if dst_name is None:
        dst_name = 'gen1' if 'gen1' in ev_dir else 'gen4'
    ev_name = 'event_representations.h5' if dst_name == 'gen1' else \
        'event_representations_ds2_nearest.h5'
    ev_h5_fn = os.path.join(ev_dir, ev_name)
    return ev_h5_fn


def read_ev_repr(h5f: str) -> np.ndarray:
    if 'event_representations_v2' not in h5f:
        h5f = get_ev_h5_fn(h5f)
    with h5py.File(str(h5f), 'r') as h5f:
        ev_repr = h5f['data'][:]
    return ev_repr


def read_objframe_idx_2_repr_idx(npy_fn: str) -> np.ndarray:
    if 'event_representations_v2' not in npy_fn:
        npy_fn = get_objframe_idx_2_repr_idx_fn(npy_fn)
    return np.load(npy_fn)
