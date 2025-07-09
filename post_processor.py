import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper


def unclip(box, unclip_ratio=3.0):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box.astype(np.int32), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    if len(expanded) == 0:
        return box
    return np.array(expanded[0], dtype=np.float32)

def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = cv2.boxPoints(bounding_box)
    points = np.array(points, dtype=np.float32)
    start_idx = points.sum(axis=1).argmin()
    box = np.roll(points, 4 - start_idx, axis=0)
    return box, min(bounding_box[1])

def box_score_fast(bitmap, box):
    h, w = bitmap.shape
    box = np.round(box, decimals=0).astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [box], 1)
    return cv2.mean(bitmap, mask)[0]

def postprocess_det(pred, ori_shape, resize_shape, box_thresh=0.6, unclip_ratio=2.5, min_size=2):
    pred = pred[0, 0]
    height, width = pred.shape
    # 阈值化
    _, binary = cv2.threshold(pred, box_thresh, 1, cv2.THRESH_BINARY)
    binary = (binary * 255).astype(np.uint8)
    # 找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    scores = []
    for contour in contours:
        if cv2.contourArea(contour) < min_size:
            continue
        box, sside = get_mini_boxes(contour)
        if sside < min_size+0.5:
            continue
        score = box_score_fast(pred, box)
        if score < box_thresh:
            continue
        box = unclip(box, unclip_ratio=unclip_ratio)
        if len(box) < 4:
            continue
        box, sside = get_mini_boxes(box)
        if sside < min_size + 2:
            continue
        # 缩放回原图
        h_ori, w_ori = ori_shape
        h_resize, w_resize = resize_shape
        ratio_h = float(h_ori) / float(h_resize)
        ratio_w = float(w_ori) / float(w_resize)
        box[:, 0] = np.clip(np.round(box[:, 0] * ratio_w), 0, w_ori)
        box[:, 1] = np.clip(np.round(box[:, 1] * ratio_h), 0, h_ori)
        boxes.append(box.astype(np.int32))
    return boxes

def postprocess_cls(logits):
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    if logits.ndim == 1:
        return [1] if probs[1] > 0.5 else [0]
    else:
        return [1 if p[1] > 0.5 else 0 for p in probs]



def ctc_decode(preds, charset, remove_duplicate=True):
    max_index = np.argmax(preds, axis=1)  # 直接取argmax
    result = []
    for i, idx in enumerate(max_index):
        if remove_duplicate and i > 0 and idx == max_index[i-1]:
            continue
        if idx > 0 and idx < len(charset):  # 跳过空白符
            result.append(charset[idx])
    return ''.join(result)
