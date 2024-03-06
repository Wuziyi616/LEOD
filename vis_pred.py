import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import copy
from tqdm import tqdm
import numpy as np
import cv2
import hdf5plugin  # resolve a weird h5py error

import torch
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')

from torch.cuda.amp import autocast as fp16_autocast
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig, OmegaConf

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_data_module, fetch_model_module
from modules.utils.ssod import filter_pred_boxes as _filter_pred_boxes
from data.utils.types import DataType
from utils.evaluation.prophesee.evaluator import get_labelmap
from utils.evaluation.prophesee.visualize.vis_utils import cv2_draw_bboxes

from nerv.utils import save_video, VideoReader

UPSAMPLE = 2
SKIP = 2
FPS = 30 // SKIP


def get_exp_name(config: DictConfig):
    """Compose the name used in wandb run's name and ckp path."""
    # dataset
    dst_name = config.dataset.name
    # model
    model_name = config.model.name
    vit_dim = config.model.backbone.embed_dim
    if vit_dim == 64:
        size = 'base'
    elif vit_dim == 48:
        size = 'small'
    elif vit_dim == 32:
        size = 'tiny'
    else:
        raise NotImplementedError(f'Unknown ViT dim {vit_dim=}')
    exp_name = f'{dst_name}_{model_name}_{size}/pred'
    if config.reverse:
        exp_name += '_reverse'
    return exp_name


def filter_boxes_ssod(boxes, dataset_name='gen1', downsampled_by_2=False):
    if boxes is None or len(boxes) == 0:
        return None, None
    # boxes: [N, 7]
    xyxy = boxes[:, :4].clone()
    new_xyxy, keep = _filter_pred_boxes(xyxy, dataset_name, downsampled_by_2)
    # update the bbox that we will keep
    boxes[keep, :4] = new_xyxy[keep]
    return boxes[keep], boxes[~keep]


@torch.inference_mode()
def event2rgb(events, cpu=False):
    """Ignore the polarity of events."""
    # events: [L, C, H, W], torch.Tensor
    cpu = cpu or ('T4' in torch.cuda.get_device_name())
    events = events.cpu() if cpu else events.cuda()
    L, C, H, W = events.shape
    C = C // 2
    events = events.reshape(L, 2, C, H, W)  # [L, 2, C, H, W]
    pos = events[:, 0].sum(1, keepdim=True)  # [L, 1, H, W]
    neg = events[:, 1].sum(1, keepdim=True)  # [L, 1, H, W]
    img = torch.ones((L, 3, H, W), dtype=torch.float32, device=events.device)
    # make any pixel that have events as black
    mask = ((pos > 0) | (neg > 0)).repeat(1, 3, 1, 1)  # [L, 3, H, W]
    img[mask] = 0.25  # 0
    # upsample
    img = F.interpolate(
        img, scale_factor=UPSAMPLE, mode='bilinear', align_corners=False)
    img = torch.round(img * 255.).to(torch.uint8)
    # img: [L, 3, H, W], torch.uint8 CPU Tensor
    return img.cpu()


def hstack_array(arrs, pad=5):
    # each arr is of shape [..., H, W, 3]
    arr_shape = arrs[0].shape
    assert all(arr.shape == arr_shape for arr in arrs)
    W = arr_shape[-2]
    num_arrs = len(arrs)
    arr_shape = list(arr_shape)
    arr_shape[-2] = pad * (num_arrs - 1) + W * num_arrs
    stack_arr = np.zeros(arr_shape, dtype=arrs[0].dtype)
    for i, arr in enumerate(arrs):
        start_idx = i * (W + pad)
        stack_arr[..., start_idx:start_idx + W, :] = arr
    return stack_arr


def vstack_array(arrs, pad=5):
    # each arr is of shape [..., H, W, 3]
    arr_shape = arrs[0].shape
    assert all(arr.shape == arr_shape for arr in arrs)
    H = arr_shape[-3]
    num_arrs = len(arrs)
    arr_shape = list(arr_shape)
    arr_shape[-3] = pad * (num_arrs - 1) + H * num_arrs
    stack_arr = np.zeros(arr_shape, dtype=arrs[0].dtype)
    for i, arr in enumerate(arrs):
        start_idx = i * (H + pad)
        stack_arr[..., start_idx:start_idx + H, :, :] = arr
    return stack_arr


def process_one_frame(ev_img, keep_pred, remove_pred, label, label_map):
    # ev_img: [3, H, W], torch.uint8
    # pred: torch.Tensor, [N, 7 ((x1, y1, x2, y2), obj_conf, cls_conf, cls_id)]
    # label: ObjectLabels (t, x, y, w, h, cls_id, cls_conf) or None

    def _draw_pred_bbox(img, pred, color=(0, 255, 0)):
        if pred is None or len(pred) == 0:
            return img
        pred = pred.cpu()
        bbox = pred[:, :4]  # [N, 4]
        obj_conf, cls_conf = pred[:, 4].numpy(), pred[:, 5].numpy()
        cls_idx = pred[:, 6].numpy()
        labels = [
            f'{label_map[int(c)]}\n{obj:.3f}x{conf:.3f}\n{obj*conf:.3f}'
            for c, obj, conf in zip(cls_idx, obj_conf, cls_conf)
        ]
        img = cv2_draw_bboxes(
            img,
            bbox * UPSAMPLE,
            labels=labels,
            colors=color,
            fontsize=0.25 * UPSAMPLE,
            thickness=1 * UPSAMPLE)
        return img

    # ori_img = ev_img.permute(1, 2, 0).contiguous().numpy()  # [H, W, 3]
    ev_img = ev_img.permute(1, 2, 0).contiguous().numpy()  # [H, W, 3]
    ev_img = _draw_pred_bbox(ev_img, keep_pred, color=(0, 255, 0))  # green
    ev_img = _draw_pred_bbox(ev_img, remove_pred, color=(255, 0, 0))  # red

    if label is not None:
        bbox = label.get_xyxy()  # [N, 4]
        class_id = label.get('class_id').numpy()  # [N]
        assert len(bbox) == len(class_id)
        class_names = [label_map[int(c)] for c in class_id]

        # draw bbox on ev_img
        ev_img = cv2_draw_bboxes(
            ev_img,
            bbox * UPSAMPLE,
            labels=class_names,
            colors=(0, 0, 0),  # black
            fontsize=0.25 * UPSAMPLE,
            thickness=1 * UPSAMPLE)

    # stack them vertically
    img = ev_img
    # img = vstack_array([ori_img, ev_img], pad=5)

    return img


@torch.inference_mode()
def pred_one_seq(model, seq, filter_box_fn, label_map, prev_t=0.):
    """Run the model on one event sequence and visualize it."""
    # short seq means it's end of an entire event sequence
    end_of_seq = seq['data'][DataType.IS_LAST_SAMPLE][0].item()
    pad_mask = torch.cat(seq['data'][DataType.IS_PADDED_MASK])  # [L, B(==1)]
    if pad_mask.any():
        # truncate till the first pad
        pad_idx = pad_mask.nonzero()[0, 0].item()
        seq['data'] = {
            k: v[:pad_idx] if isinstance(v, list) else v
            for k, v in seq['data'].items()
        }

    torch.cuda.empty_cache()
    all_preds, ev_seq, all_lbl = model.predict_one_seq(seq)
    # all_preds: a `L`-len list, each is [N, 7], i.e. bbox at each timestep
    # ev_seq: [L, C, H, W], grid-like event reprs, on GPU
    # all_lbl: a `L`-len list of `ObjectLabels` or None
    torch.cuda.empty_cache()
    ev_imgs = event2rgb(ev_seq)  # [L, 3, H, W], torch.uint8 CPU Tensor

    all_imgs = []
    for i, (ev_img, pred, lbl) in \
            enumerate(tqdm(zip(ev_imgs, all_preds, all_lbl), desc='Plot')):
        keep_pred, remove_pred = filter_box_fn(pred)
        ev_img = process_one_frame(
            ev_img, keep_pred, remove_pred, lbl, label_map=label_map)
        # write timestamp to the top-left corner
        t = i * 0.05 + prev_t  # ms -> s
        cv2.putText(
            ev_img,
            f'{t:.2f}s (N={int(t / 0.05):04d})',
            org=(10, 10 + 10 * UPSAMPLE),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5 * UPSAMPLE,
            color=(255, 0, 0),
            thickness=1 * UPSAMPLE)
        if lbl is None:
            all_imgs.append(ev_img)
        else:  # pause for a while
            if len(label_map) == 3:  # Gen4 labeling freq is high
                all_imgs.extend([ev_img] * 3)  # 0.2s
            else:  # 0.5s
                all_imgs.extend([ev_img] * (FPS // 2))

    video = np.stack(all_imgs[::SKIP], axis=0)  # [T, H, W, 3]
    return video, end_of_seq, t + 0.05


@hydra.main(config_path='config', config_name='vis', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    is_gen1 = (config.dataset.name == 'gen1')
    num_video = config.num_video
    config.batch_size.eval = 1
    config.hardware.num_workers.eval = 1
    config.dataset.test_ratio = num_video / 400 if is_gen1 else num_video / 100
    config.dataset.sequence_length = 640 if is_gen1 else 256

    # ---------------------
    # Data
    # ---------------------
    label_map = get_labelmap(dst_name=config.dataset.name)
    data_module = fetch_data_module(config=config)
    data_module.setup(stage='test')
    loader = data_module.test_dataloader()
    filter_boxes_fn = lambda x: filter_boxes_ssod(
        x, config.dataset.name, config.dataset.downsample_by_factor_2)

    # ---------------------
    # Model
    # ---------------------
    module = fetch_model_module(config=config)
    module.load_weight(config.checkpoint)
    module.setup(stage='test')

    # ---------------------
    # Inference on each event sequence
    # ---------------------
    module = module.eval().cuda()
    vis_dir = os.path.join('./vis/', get_exp_name(config))
    os.makedirs(vis_dir, exist_ok=True)
    sub_videos, video_cnt, prev_t = [], 0, 0.
    for batch in tqdm(loader, desc='Event sequence'):
        with fp16_autocast():
            video, eoe, prev_t = pred_one_seq(
                module, batch, filter_boxes_fn, label_map, prev_t=prev_t)
        # skip video_0 which is stupid on Gen1
        if video_cnt == 0 and is_gen1:
            video_cnt, prev_t = video_cnt + int(eoe), 0.
            continue
        sub_videos.append(video)
        if eoe:
            video = np.concatenate(sub_videos, axis=0)
            sub_videos, video_cnt, prev_t = [], video_cnt + 1, 0.
            seq_name = os.path.basename(batch['data'][DataType.PATH][0])
            save_fn = os.path.join(vis_dir, f'{seq_name}.mp4')
            save_video(video, save_fn, fps=FPS)
        if video_cnt >= num_video:
            break

    if not config.reverse:
        exit(-1)
    # Get another dataloader where the temporal order of events is reversed.
    rev_config = copy.deepcopy(config)
    rev_config.dataset.reverse_event_order = True
    rev_data_module = fetch_data_module(config=rev_config)
    rev_data_module.setup(stage='test')
    rev_loader = rev_data_module.test_dataloader()

    sub_videos, video_cnt, prev_t = [], 0, 0.
    for batch in tqdm(rev_loader, desc='Reverse event sequence'):
        with fp16_autocast():
            video, eoe, prev_t = pred_one_seq(
                module, batch, filter_boxes_fn, label_map, prev_t=prev_t)
        if video_cnt == 0 and is_gen1:
            video_cnt, prev_t = video_cnt + int(eoe), 0.
            continue
        sub_videos.append(video)
        if eoe:
            rev_video = np.concatenate(sub_videos, axis=0)
            sub_videos, video_cnt, prev_t = [], video_cnt + 1, 0.
            # load the original video prediction from file
            seq_name = os.path.basename(batch['data'][DataType.PATH][0])
            save_fn = os.path.join(vis_dir, f'{seq_name}.mp4')
            video = np.stack(VideoReader(save_fn, to_rgb=False).read_video())
            # reverse the temporal order of the `rev_video`
            # and stack it with video horizontally
            rev_video = np.ascontiguousarray(rev_video[::-1])
            # weird, the shape might change before/after loading to disk
            (T1, H1, W1), (T2, H2, W2) = video.shape[:3], rev_video.shape[:3]
            video = video[:min(T1, T2), :min(H1, H2), :min(W1, W2), :]
            rev_video = rev_video[:min(T1, T2), :min(H1, H2), :min(W1, W2), :]
            video = hstack_array([video, rev_video], pad=5)
            save_video(video, save_fn.replace('.mp4', '_both.mp4'), fps=FPS)
            os.remove(save_fn)
        if video_cnt >= num_video:
            break


if __name__ == '__main__':
    main()
