import cv2
import numpy as np

def postprocess_det(pred, ori_shape, resize_shape, box_thresh=0.5):
    """
    :param pred: 模型输出的概率图，shape = (1, 1, H, W)
    :param ori_shape: 原图尺寸 [H_ori, W_ori]
    :param resize_shape: 缩放后尺寸 [H_resized, W_resized]
    :return: list of boxes (每个 box 是 4 点坐标)
    """
    pred = pred[0, 0]  # 去除 batch 和通道维度

    # 1. 二值化概率图
    # 建议 pred = (pred * 255).astype(np.uint8) 后再 threshold，或 threshold 后不再乘 255
    _, bin_map = cv2.threshold((pred * 255).astype(np.uint8), int(box_thresh * 255), 255, cv2.THRESH_BINARY)
    # 这样更符合 OpenCV 习惯
    #_, bin_map = cv2.threshold(pred, box_thresh, 1, cv2.THRESH_BINARY)
    bin_map = (bin_map * 255).astype(np.uint8)

    # 2. 找轮廓
    contours, _ = cv2.findContours(...)  # 这样写兼容性更好
    # contours, _ = cv2.findContours(bin_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.float32)
         #boxes.append(box.astype(np.int32))  # 如果后续用于绘制，建议转 int

        # 3. 缩放回原图尺寸
        h_ori, w_ori = ori_shape
        h_resize, w_resize = resize_shape
        ratio_h = h_ori / h_resize
        ratio_w = w_ori / w_resize
        box[:, 0] *= ratio_w
        box[:, 1] *= ratio_h
        #boxes.append(box)
        boxes.append(box.astype(np.int32))# 如果后续用于绘制，建议转 int

    return boxes  # 返回 n 个 box，每个 box 为 4 点坐标


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
