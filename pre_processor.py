import cv2
import numpy as np


class DecodeImage:
    """图像解码算子（支持路径/字节流输入）"""

    def __init__(self, img_mode='BGR', channel_first=False):
        self.img_mode = img_mode
        self.channel_first = channel_first

    def __call__(self, data):
        # 支持文件路径或二进制数据输入
        if isinstance(data['image'], str):
            with open(data['image'], 'rb') as f:
                data['image'] = f.read()

        if isinstance(data['image'], bytes):
            # 从二进制数据解码
            data['image'] = np.frombuffer(data['image'], dtype=np.uint8)
            data['image'] = cv2.imdecode(data['image'], cv2.IMREAD_COLOR)

        # 颜色空间转换
        if self.img_mode == 'RGB':
            data['image'] = cv2.cvtColor(data['image'], cv2.COLOR_BGR2RGB)
        elif self.img_mode == 'GRAY':
            if len(data['image'].shape) == 3:
                data['image'] = cv2.cvtColor(data['image'], cv2.COLOR_BGR2GRAY)
            data['image'] = np.expand_dims(data['image'], axis=-1)  # 保持三维

        # 记录原始尺寸 [H, W]
        data['ori_shape'] = np.array(data['image'].shape[:2])

        return data


class DetResizeForTest:
    """检测专用尺寸调整算子"""

    def __init__(self, limit_side_len=960, limit_type='max', divisor=32):
        self.limit_side_len = limit_side_len
        self.limit_type = limit_type
        self.divisor = divisor

    def __call__(self, data):
        img = data['image']
        h, w = img.shape[:2]

        # 计算缩放比例
        if self.limit_type == 'max':
            ratio = float(self.limit_side_len) / max(h, w)
        else:
            ratio = float(self.limit_side_len) / min(h, w)
        if ratio > 1.0:
            ratio = 1.0
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)
        resize_h = max(self.divisor, int(round(resize_h / self.divisor)) * self.divisor)
        resize_w = max(self.divisor, int(round(resize_w / self.divisor)) * self.divisor)

        # 执行缩放
        resized_img = cv2.resize(img, (resize_w, resize_h))

        # 更新数据
        data['image'] = resized_img
        data['shape'] = np.array([resize_h, resize_w])
        data['ratio_h'] = resize_h / float(h)
        data['ratio_w'] = resize_w / float(w)
        return data


class ClsResizeForTest:
    """分类专用尺寸调整算子"""

    def __init__(self, image_shape=(3, 48, 192)):
        # 初始化目标尺寸：目标高度和宽度 (默认48高x192宽)
        self.image_shape = image_shape

    def __call__(self, data):
        # 获取输入图像
        img = data['image']
        imgC, imgH, imgW = self.image_shape
        h, w = img.shape[:2]
        ratio = float(w) / float(h)
        resized_w = imgW
        if np.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(np.ceil(imgH * ratio))
        resized_img = cv2.resize(img, (resized_w, imgH))

        # 如果新宽度小于目标宽度，进行右侧填充
        if resized_w < imgW:
            # 创建全零填充图像（目标高度x目标宽度）
            padding_img = np.zeros((imgH, imgW, imgC), dtype=np.uint8)
            # 将缩放后的图像复制到填充图像的左侧
            padding_img[:, 0:resized_w, :] = resized_img
            # 用填充后的图像替换原缩放图像
            resized_img = padding_img

        # 更新数据字典中的图像
        data['image'] = resized_img
        # 存储调整后的尺寸信息
        data['resized_shape'] = (imgH, imgW)
        return data


class RecResizeForTest:
    """识别专用尺寸调整算子"""

    def __init__(self, image_shape=(3, 32, 320)):
        # 初始化参数：目标高度、最大宽度限制、最小宽度限制
        self.image_shape = image_shape

    def __call__(self, data):
        # 获取输入图像
        img = data['image']
        imgC, imgH, imgW = self.image_shape
        h, w = img.shape[:2]
        ratio = w / float(h)
        if np.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(np.ceil(imgH * ratio))
        resized_img = cv2.resize(img, (resized_w, imgH))

        # 如果新宽度小于目标宽度，进行右侧填充
        if resized_w < imgW:
            # 创建全零填充图像（目标高度x目标宽度）
            padding_img = np.zeros((imgH, imgW, imgC), dtype=np.uint8)
            # 将缩放后的图像复制到填充图像的左侧
            padding_img[:, 0:resized_w, :] = resized_img
            # 用填充后的图像替换原缩放图像
            resized_img = padding_img

        # 更新数据字典
        data['image'] = resized_img
        # 存储调整后的尺寸
        data['resized_shape'] = (imgH, imgW)
        return data


class NormalizeImage:
    """图像归一化算子"""

    def __init__(self, mean=None, std=None, scale=1. / 255.):
        # 初始化归一化参数
        # 均值：可以是标量或数组
        self.mean = np.array(mean).astype('float32') if mean else None
        # 标准差：可以是标量或数组
        self.std = np.array(std).astype('float32') if std else None
        # 缩放因子（默认1/255将像素值转换到0-1范围）
        self.scale = scale

    def __call__(self, data):
        # 获取图像并转换为float32类型
        img = data['image'].astype('float32')

        # 第一步：像素值缩放（通常缩放到0-1范围）
        if self.scale:
            img *= self.scale

        # 第二步：标准化处理（减去均值，除以标准差）
        if self.mean is not None and self.std is not None:
            img = (img - self.mean.reshape(1, 1, -1)) / self.std.reshape(1, 1, -1)

        # 更新数据字典中的图像
        data['image'] = img
        return data


class ToCHWImage:
    """通道顺序转换算子 (HWC -> CHW)"""

    def __call__(self, data):
        img = data['image']
        if len(img.shape) == 2:  # 灰度图 (H, W) -> (1, H, W)
            img = np.expand_dims(img, axis=0)
        elif len(img.shape) == 3:  # 彩色图 (H, W, C) -> (C, H, W)
            img = img.transpose((2, 0, 1))
        data['image'] = img
        return data


class KeepKeys:
    """保留指定键值算子"""

    def __init__(self, keep_keys=['image', 'shape', 'ratio_h', 'ratio_w', 'ori_shape']):
        self.keep_keys = keep_keys

    def __call__(self, data):
        return {k: data[k] for k in self.keep_keys if k in data}


class BatchPadding:
    """批处理填充算子（支持多张图像）"""

    def __init__(self, pad_value=0, max_width=None):
        self.pad_value = pad_value
        self.max_width = max_width

    def __call__(self, batch_data):
        # 获取最大尺寸
        max_h = max([d['image'].shape[1] for d in batch_data])
        max_w = max([d['image'].shape[2] for d in batch_data])

        if self.max_width:
            max_w = min(max_w, self.max_width)

        # 创建批处理数组
        batch_images = []

        for data in batch_data:
            img = data['image']
            c, h, w = img.shape

            # 创建填充后的图像
            padded_img = np.full((c, max_h, max_w), self.pad_value, dtype=img.dtype)
            padded_img[:, :h, :w] = img[:, :, :min(w, max_w)]

            batch_images.append(padded_img)

        return {'image': np.array(batch_images)}


def create_preprocess_pipeline(task_type, config):
    """创建预处理流水线"""
    pipeline = []

    # 通用初始处理
    pipeline.append(DecodeImage(img_mode=config.get('img_mode', 'BGR')))

    # 任务特定处理
    if task_type == 'det':
        pipeline.append(DetResizeForTest(
            limit_side_len=config.get('limit_side_len', 960),
            limit_type=config.get('limit_type', 'max'),
            divisor=config.get('divisor', 32)
        ))
        pipeline.append(NormalizeImage(
            mean=config.get('mean', [0.485, 0.456, 0.406]),
            std=config.get('std', [0.229, 0.224, 0.225]),
            scale=config.get('scale', 1. / 255.)
        ))
        pipeline.append(ToCHWImage())
        pipeline.append(KeepKeys(keep_keys=['image', 'shape', 'ratio_h', 'ratio_w', 'ori_shape']))
    elif task_type == 'cls':
        pipeline.append(ClsResizeForTest(
            image_shape=config.get('image_shape', (3, 48, 192))
        ))
        pipeline.append(NormalizeImage(
            mean=config.get('mean', [0.5, 0.5, 0.5]),
            std=config.get('std', [0.5, 0.5, 0.5]),
            scale=config.get('scale', 1. / 255.)
        ))
        pipeline.append(ToCHWImage())
        pipeline.append(KeepKeys(keep_keys=['image']))
    elif task_type == 'rec':
        pipeline.append(RecResizeForTest(
            image_shape=config.get('image_shape', (3, 32, 320))
        ))
        pipeline.append(NormalizeImage(
            mean=config.get('mean', [0.5, 0.5, 0.5]),
            std=config.get('std', [0.5, 0.5, 0.5]),
            scale=config.get('scale', 1. / 255.)
        ))
        pipeline.append(ToCHWImage())
        pipeline.append(KeepKeys(keep_keys=['image']))
    return pipeline


def preprocess_for_task(images, task_type='det', batch_size=1, **config):
    """
    完整预处理流程
    :param images: 单张图像或多张图像列表
    :param task_type: 任务类型 (det/cls/rec)
    :param batch_size: 批处理大小
    :param config: 预处理配置参数
    :return: 批处理数据
    """
    if not isinstance(images, list):
        images = [images]

    # 创建预处理流水线
    pipeline = create_preprocess_pipeline(task_type, config)

    # 处理单张图像
    if batch_size == 1:
        data = {'image': images[0]}
        for op in pipeline:
            data = op(data)
        return data

    # 批处理
    processed_batch = []
    for img in images:
        data = {'image': img}
        for op in pipeline:
            data = op(data)
        processed_batch.append(data)

    # 批处理填充
    batcher = BatchPadding(
        pad_value=config.get('pad_value', 0),
        max_width=config.get('max_batch_width', None)
    )
    return batcher(processed_batch)


# 使用示例
if __name__ == "__main__":
    # 检测任务示例
    det_config = {
        'limit_side_len': 960,
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5]
    }
    det_data = preprocess_for_task('mars.png', 'det', **det_config)
    print("检测预处理结果:", det_data['image'].shape)

    # 识别任务示例

    rec_config = {
        'image_shape': (3, 32, 320),  # 明确指定图像形状
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'scale': 1.0 / 255.0
    }
    rec_data = preprocess_for_task('mars.png', 'rec', **rec_config)
    print("识别预处理结果:", rec_data['image'].shape)

    # 批处理示例 (2张图像)
    batch_data = preprocess_for_task(['mars.png'], 'det', batch_size=1)
    print("批处理结果:", batch_data['image'].shape)
