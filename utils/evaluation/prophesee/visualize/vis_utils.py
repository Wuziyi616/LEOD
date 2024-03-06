"""
Functions to display events and boxes
Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import print_function

import copy
import cv2
import numpy as np

import torch

import bbox_visualizer as bbv

FONT = cv2.FONT_HERSHEY_SIMPLEX

LABELMAP_GEN1 = ("car", "pedestrian")
LABELMAP_GEN4 = ('pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light')
LABELMAP_GEN4_SHORT = ('pedestrian', 'two wheeler', 'car')


def get_labelmap(dst_name):
    return LABELMAP_GEN1 if dst_name.lower() == 'gen1' else LABELMAP_GEN4_SHORT


def make_binary_histo(events, img=None, width=304, height=240):
    """
    simple display function that shows negative events as blacks dots and positive as white one
    on a gray background
    args :
        - events structured numpy array
        - img (numpy array, height x width x 3) optional array to paint event on.
        - width int
        - height int
    return:
        - img numpy array, height x width x 3)
    """
    if img is None:
        img = 127 * np.ones((height, width, 3), dtype=np.uint8)
    else:
        # if an array was already allocated just paint it grey
        img[...] = 127
    if events.size:
        assert events['x'].max() < width, "out of bound events: x = {}, w = {}".format(events['x'].max(), width)
        assert events['y'].max() < height, "out of bound events: y = {}, h = {}".format(events['y'].max(), height)

        img[events['y'], events['x'], :] = 255 * events['p'][:, None]
    return img


def draw_bboxes_bbv(img, boxes, labelmap=LABELMAP_GEN1) -> np.ndarray:
    """
    draw bboxes in the image img
    """
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    if labelmap == LABELMAP_GEN1:
        classid2colors = {
            0: (255, 255, 0),  # car -> yellow (rgb)
            1: (0, 0, 255),  # ped -> blue (rgb)
        }
        scale_multiplier = 4
    else:
        assert labelmap == LABELMAP_GEN4_SHORT
        classid2colors = {
            0: (0, 0, 255),  # ped -> blue (rgb)
            1: (0, 255, 255),  # 2-wheeler cyan (rgb)
            2: (255, 255, 0),  # car -> yellow (rgb)
        }
        scale_multiplier = 2

    add_score = True
    ht, wd, ch = img.shape
    dim_new_wh = (int(wd * scale_multiplier), int(ht * scale_multiplier))
    if scale_multiplier != 1:
        img = cv2.resize(img, dim_new_wh, interpolation=cv2.INTER_AREA)
    for i in range(boxes.shape[0]):
        pt1 = (int(boxes['x'][i]), int(boxes['y'][i]))
        size = (int(boxes['w'][i]), int(boxes['h'][i]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        bbox = (pt1[0], pt1[1], pt2[0], pt2[1])
        bbox = tuple(x * scale_multiplier for x in bbox)

        score = boxes['class_confidence'][i]
        class_id = boxes['class_id'][i]
        class_name = labelmap[class_id % len(labelmap)]
        bbox_txt = class_name
        if add_score:
            bbox_txt += f' {score:.2f}'
        color_tuple_rgb = classid2colors[class_id]
        img = bbv.draw_rectangle(img, bbox, bbox_color=color_tuple_rgb)
        img = bbv.add_label(img, bbox_txt, bbox, text_bg_color=color_tuple_rgb, top=True)

    return img


def draw_bboxes(img, boxes, labelmap=LABELMAP_GEN1) -> None:
    """
    draw bboxes in the image img
    """
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    for i in range(boxes.shape[0]):
        pt1 = (int(boxes['x'][i]), int(boxes['y'][i]))
        size = (int(boxes['w'][i]), int(boxes['h'][i]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        score = boxes['class_confidence'][i]
        class_id = boxes['class_id'][i]
        class_name = labelmap[class_id % len(labelmap)]
        color = colors[class_id * 60 % 255]
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.rectangle(img, pt1, pt2, color, 1)
        cv2.putText(img, class_name, (center[0], pt2[1] - 1), FONT, 0.5, color)
        cv2.putText(img, str(score), (center[0], pt1[1] - 1), FONT, 0.5, color)


def cv2_put_text_lines(img, label, org, color, fontScale=0.5, thickness=1):
    """
    put text in the image img
    """
    lines = label.split('\n')
    for i, line in enumerate(lines):
        (l_w, l_h_no_baseline), baseline = cv2.getTextSize(
            line,
            fontFace=FONT,
            fontScale=fontScale,
            thickness=thickness,
        )
        l_h = l_h_no_baseline + baseline + 2
        cv2.putText(img, line, org=org, fontFace=FONT, fontScale=fontScale, color=color, thickness=thickness)
        org = (org[0], org[1] + l_h)


def cv2_draw_bboxes(img, boxes, labels, colors, fontsize=0.5, thickness=1) -> None:
    """
    draw bboxes in the image img
    """
    img = copy.deepcopy(img)

    if colors is None:
        colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
        colors = [tuple(*item) for item in colors.tolist()]
    elif len(colors) == 3 and isinstance(colors[0], int):
        colors = [colors] * len(boxes)

    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).contiguous().cpu().numpy()  # [H, W, 3]
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    for i in range(boxes.shape[0]):
        pt1 = (int(boxes[i, 0]), int(boxes[i, 1]))
        pt2 = (int(boxes[i, 2]), int(boxes[i, 3]))
        label = labels[i]
        color = colors[i % 255]
        cv2.rectangle(img, pt1, pt2, color, thickness=thickness)
        cv2_put_text_lines(img, label, org=(pt1[0], pt1[1]), color=color, fontScale=fontsize, thickness=thickness)

    return img
