import os

from click.core import batch
from fastapi import HTTPException
import paddle2onnx
import onnxruntime as ort


class ModelTrans:
    def __init__(self):
        self.supported_models = ['det', 'rec', 'cls']

    def convert_to_onnx(self, model_dir, model_type='det'):
        """
        将PaddleOCR模型转换为ONNX格式

        参数:
            model_dir: 模型目录路径
            model_type: 模型类型 ('det', 'rec', 'cls')
        """
        if model_type not in self.supported_models:
            raise ValueError(f"不支持的模型类型: {model_type}, 支持的模型类型: {self.supported_models}")

        # 检查模型文件是否存在
        model_files = ['inference.pdmodel', 'inference.pdiparams']
        for file in model_files:
            if not os.path.exists(os.path.join(model_dir, file)):
                raise FileNotFoundError(f"模型文件 {file} 不存在于 {model_dir}")

        # 设置输出路径
        output_dir = os.path.join("onnx_models", model_type)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{model_type}_model.onnx")

        try:
            paddle2onnx.export(
                model_filename=os.path.join(model_dir, "inference.pdmodel"),
                params_filename=os.path.join(model_dir, "inference.pdiparams"),
                save_file=output_path,
                opset_version=12,
                enable_onnx_checker=True,

            )
            # 验证转换后的模型
            self._validate_onnx_model(output_path)

            return {"status": "success", "message": f"{model_type}模型转换成功", "output_path": output_path}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"模型转换失败: {str(e)}")

    def _validate_onnx_model(self, model_path):
        """验证ONNX模型是否可以加载"""
        try:
            ort_session = ort.InferenceSession(model_path)
            print(f"模型验证成功 - 输入: {[inp.name for inp in ort_session.get_inputs()]}")
            print(f"模型验证成功 - 输出: {[out.name for out in ort_session.get_outputs()]}")
            return True
        except Exception as e:
            raise ValueError(f"ONNX模型验证失败: {str(e)}")

    def batch_convert(self, model_config):
        """批量转换多个模型"""
        results = {}
        for model_type, model_dir in model_config.items():
            if model_type in self.supported_models:
                try:
                    result = self.convert_to_onnx(model_dir, model_type)
                    results[model_type] = result
                except Exception as e:
                    results[model_type] = {"status": "failed", "message": str(e)}
        return results

if __name__ == "__main__":
    # 初始化模型转换器
    model_trans = ModelTrans()

    # 配置模型路径（根据图中目录结构）
    model_config = {
        'det': 'ch_PP-OCRv3_det_infer',  # 检测模型
        'rec': 'ch_PP-OCRv3_rec_infer',  # 识别模型
        'cls': 'ch_ppocr_mobile_v2.0_cls_infer'  # 分类模型
    }

    results = model_trans.batch_convert(model_config)

    for model_type, result in results.items():
        status = "成功" if result.get("status") == "success" else "失败"
        print(f"{model_type.upper()}模型: {status}")
        if "message" in result:
            print(f"   - {result['message']}")
        if "output_path" in result:
            print(f"   - ONNX模型路径: {result['output_path']}")
        if result.get("status") == "failed":
            print(f"   - 错误详情: {result['message']}")
        print("-" * 50)