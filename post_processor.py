import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper


def postprocess_det(pred, ori_shape, resize_shape, box_thresh=0.3, unclip_ratio=2.0):
    """
    将模型输出的概率图转换为文本框坐标（带 unclip）
    :param pred: 模型输出，shape = (1, 1, H, W)
    :param ori_shape: 原图尺寸 [H_ori, W_ori]
    :param resize_shape: 模型输入图尺寸 [H_resized, W_resized]
    :param box_thresh: 二值化阈值
    :param unclip_ratio: 多边形膨胀比（模拟 DBNet 的行为）
    :return: list of boxes， 每个 box 是 shape=(N, 2) 的坐标点组成的 numpy array
    """
    pred = pred[0, 0]  # 去除 batch 和 channel

    # 1. 二值化概率图
    _, bin_map = cv2.threshold(pred, box_thresh, 1, cv2.THRESH_BINARY)
    bin_map = (bin_map * 255).astype(np.uint8)

    # 2. 找轮廓
    contours, _ = cv2.findContours(bin_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    h_ori, w_ori = ori_shape
    h_resize, w_resize = resize_shape
    ratio_h = h_ori / h_resize
    ratio_w = w_ori / w_resize

    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.float32)

        #  膨胀 polygon
        box = unclip_polygon(box, unclip_ratio=unclip_ratio)

        # 有可能变成非矩形（例如 6 点），检查一下合法性
        if box.shape[0] < 4:
            continue

        # 缩放回原图尺寸
        box[:, 0] *= ratio_w
        box[:, 1] *= ratio_h

        boxes.append(box)

    return boxes


def postprocess_cls(logits):
    """
    :param logits: 分类输出 [B, 2]，表示 [正向, 反向] 概率
    :return: 每张图是否需要旋转
    """
    pred = np.argmax(logits, axis=1)  # 0 表示不转，1 表示旋转 180°
    return pred.tolist()

def ctc_decode(preds, charset, remove_duplicate=True):
    """
    :param preds: shape = [T, num_classes]，经过 softmax 的概率
    :param charset: 字符集列表，如 ["a", "b", ..., "z", "0", ..., "9"]
    :return: 解码得到的字符串
    """
    max_index = np.argmax(preds, axis=1)
    last_char = -1
    result = []

    for i in max_index:
        # CTC 解码时，确保 charset[0] 是 blank，或显式指定 blank 索引
        # if i != blank_index and ...
        if i != 0 and (not remove_duplicate or i != last_char):
            # charset[i] 需确保 i < len(charset)
            result.append(charset[i])
        last_char = i

    return ''.join(result)


def unclip_polygon(box, unclip_ratio=2.0):
    """对 box 多边形做膨胀"""
    poly = Polygon(box)
    area = poly.area
    perimeter = poly.length
    if perimeter == 0:
        return box
    distance = area * unclip_ratio / perimeter

    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box.astype(np.int32), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(distance)
    if len(expanded) == 0:
        return box
    return np.array(expanded[0], dtype=np.float32)
