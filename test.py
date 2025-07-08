import cv2
import numpy as np
import paddle.inference as paddle_infer

import pre_processor
import post_processor



def load_paddle_model(model_dir):
    """
    加载 Paddle 推理模型（.pdmodel/.pdiparams）
    :param model_dir: 模型所在文件夹路径（里面应有 *.pdmodel 和 *.pdiparams）
    """
    config = paddle_infer.Config(
        model_dir + "/inference.pdmodel",
        model_dir + "/inference.pdiparams"
    )
    config.enable_use_gpu(1024, 0)  # 使用 GPU，如使用 CPU 可注释掉这行
    predictor = paddle_infer.create_predictor(config)
    return predictor



def run_det_model(predictor, input_tensor):
    """
    :param predictor: Paddle detection predictor
    :param input_tensor: shape = [1, 3, H, W]
    :return: shape = [1, 1, H, W]（概率图）
    """
    input_tensor = input_tensor.astype('float32')
    input_name = predictor.get_input_names()[0]
    input_handle = predictor.get_input_handle(input_name)
    input_handle.copy_from_cpu(input_tensor)

    predictor.run()

    output_name = predictor.get_output_names()[0]
    output_handle = predictor.get_output_handle(output_name)
    output_data = output_handle.copy_to_cpu()

    return output_data  # shape: [1, 1, H, W]



def run_cls_model(predictor, input_tensor):
    """
    调用分类模型执行预测
    :param predictor: Paddle predictor
    :param input_tensor: shape [1, C, H, W]
    :return: softmax概率 [1, 2]
    """
    input_tensor = input_tensor.astype('float32')

    # 获取输入名和句柄
    input_name = predictor.get_input_names()[0]
    input_handle = predictor.get_input_handle(input_name)
    input_handle.copy_from_cpu(input_tensor)

    # 推理
    predictor.run()

    # 获取输出
    output_name = predictor.get_output_names()[0]
    output_handle = predictor.get_output_handle(output_name)
    output_data = output_handle.copy_to_cpu()  # shape [1, 2]

    return output_data  # 直接返回 logits 或概率

def run_rec_model(predictor, input_tensor):
    """
    :param predictor: Paddle recognition predictor
    :param input_tensor: shape = [1, 3, 32, W]
    :return: shape = [T, num_classes]，T为时间步
    """
    input_tensor = input_tensor.astype('float32')
    input_name = predictor.get_input_names()[0]
    input_handle = predictor.get_input_handle(input_name)
    input_handle.copy_from_cpu(input_tensor)

    predictor.run()

    output_name = predictor.get_output_names()[0]
    output_handle = predictor.get_output_handle(output_name)
    output_data = output_handle.copy_to_cpu()  # shape: [1, T, num_classes]

    # 去掉 batch 维度，得到 [T, num_classes]
    return output_data[0]

def load_charset(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        charset = [line.strip() for line in f]
    return ['blank'] + charset  # 第一个为 CTC blank

charset = load_charset('ppocr_keys_v1.txt')

def visualize_boxes(image_path, boxes):
    img = cv2.imread(image_path)
    for box in boxes:
        pts = np.array(box, dtype=np.int32).reshape((-1,1,2))
        cv2.polylines(img, [pts], isClosed=True, color=(0,255,0), thickness=2)
    cv2.imwrite('debug_boxes.jpg', img)


def ocr_pipeline(image_path):
    # === 1. 检测阶段 ===
    det_config = {
        'limit_side_len': 960,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }
    det_data = pre_processor.preprocess_for_task(image_path, task_type='det',**det_config)
    print("检测预处理结果:", det_data['image'].shape)
    det_input = np.expand_dims(det_data['image'], axis=0)  # [1, C, H, W]
    det_model = load_paddle_model('./ch_PP-OCRv3_det_infer')
    # 在循环外加载分类和识别模型，避免重复加载
    cls_model = load_paddle_model("./ch_ppocr_mobile_v2.0_cls_infer")
    rec_model = load_paddle_model('./ch_PP-OCRv3_rec_infer')
    det_pred = run_det_model(det_model,det_input)

    ori_shape = det_data['ori_shape']
    resize_shape = det_data['shape']
    boxes = post_processor.postprocess_det(det_pred, ori_shape, resize_shape)

    print(f"[检测] 共检测到 {len(boxes)} 个文本框")

    # === 2. 对每个框进行抠图 ===
    image = cv2.imread(image_path)
    texts = []

    for box in boxes:
        # 获得旋转矩形框
        rect = cv2.minAreaRect(box)
        center, size, angle = rect
        size = tuple([int(s) for s in size])

        # 获取仿射矩阵并裁剪
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        cropped = cv2.getRectSubPix(rotated, size, center)

        # === 3. 分类模型：判断是否需要旋转 ===
        rec_config = {
            'target_height': 32,
            'max_width': 320,
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5]
        }
        cls_data = pre_processor.preprocess_for_task(cropped, task_type='cls', **rec_config)
        #print("识别预处理结果:", rec_data['image'].shape)
        # 分类模型推理
        cls_input = np.expand_dims(cls_data['image'], axis=0)
        cls_result = run_cls_model(cls_model, cls_input)[0]
        is_rotated = post_processor.postprocess_cls([cls_result])[0]

        if is_rotated == 1:
            cropped = cv2.rotate(cropped, cv2.ROTATE_180)

        # === 4. 识别模型 ===
        rec_data = pre_processor.preprocess_for_task(cropped, task_type='rec',batch_size=2)

        # 识别模型推理
        rec_input = np.expand_dims(rec_data['image'], axis=0)
        rec_pred = run_rec_model(rec_model, rec_input)  # 直接传 rec_input，保持 batch 维度

        # === 5. 解码 ===
        text = post_processor.ctc_decode(rec_pred, charset)
        texts.append(text)

    # === 输出结果 ===
    print("[识别结果]")
    visualize_boxes('road.jpg', boxes)
    for i, text in enumerate(texts):
        print(f"Box {i + 1}: {text}")

# === 执行测试 ===
if __name__ == "__main__":
    ocr_pipeline("road.jpg")
