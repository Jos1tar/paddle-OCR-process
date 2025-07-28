import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os, shutil, zipfile
import numpy as np
import sys
from onnx_ini import ocr_pipeline
import model_trans
from tempfile import TemporaryDirectory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


app = FastAPI()
model_trans = model_trans.ModelTrans()  # 创建ModelTrans实例

@app.get("/")
async def root():
    return {"message": "greeting service"}


"""
def upload_to_cloud(file_path: str, filename: str) -> str:
    auth = oss2.Auth('your-access-key-id', 'your-access-key-secret')
    bucket = oss2.Bucket(auth, 'https://oss-cn-region.aliyuncs.com', 'your-bucket-name')
    with open(file_path, 'rb') as f:
        bucket.put_object(f"images/{filename}", f)
    return f"https://your-bucket-name.oss-cn-region.aliyuncs.com/images/{filename}"

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    # 本地临时保存路径
    temp_folder = "temp_uploads"
    os.makedirs(temp_folder, exist_ok=True)

    # 使用 UUID 防止文件名冲突
    suffix = os.path.splitext(file.filename)[-1]
    unique_filename = f"{uuid.uuid4().hex}{suffix}"
    temp_path = os.path.join(temp_folder, unique_filename)

    # 写入本地临时文件
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # 上传到云空间，并获取URL
        url = upload_to_cloud(temp_path, unique_filename)
        return JSONResponse(content={"status": "success", "url": url})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

"""




# 添加模型转换接口
@app.post("/convert/paddleOCR")
async def convert_models(file: UploadFile = File(None)):
    try:
        if file is None:
            # 使用默认路径下的模型
            model_config = {
                'det': 'ch_PP-OCRv3_det_infer',
                'rec': 'ch_PP-OCRv3_rec_infer',
                'cls': 'ch_ppocr_mobile_v2.0_cls_infer'
            }
            results = model_trans.batch_convert(model_config)
            return JSONResponse(content={"status": "success", "results": results})

        # =====================
        # 如果上传了 zip 模型包
        # 临时保存 zip 文件
        with TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, file.filename)
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # 解压 zip 文件
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            # 自动寻找模型子目录（要求 zip 中存在 det/rec/cls 子文件夹）
            model_config = {}
            for sub_name in os.listdir(tmpdir):
                sub_path = os.path.join(tmpdir, sub_name)
                if not os.path.isdir(sub_path):
                    continue  # 跳过非目录文件
                # 判断文件夹名是否包含关键词
                for model_type in ['det', 'rec', 'cls']:
                    if model_type in sub_name.lower():
                        model_file = os.path.join(sub_path, 'inference.pdmodel')
                        if os.path.exists(model_file):
                            model_config[model_type] = sub_path

            if not model_config:
                raise ValueError("上传的模型包中没有包含 det/rec/cls 的模型文件夹")

            results = model_trans.batch_convert(model_config)
            return JSONResponse(content={"status": "success", "results": results})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型转换失败: {str(e)}")

@app.post("/orcpredict/")
async def predict(
    file: UploadFile = File(None),
    use_default: bool = False,
    DEFAULT_IMAGE_PATH: str = "orc_upload/gpt.png"
):
    try:
        # 决定使用哪张图片
        if use_default or file is None:
            image_path = DEFAULT_IMAGE_PATH
            print(f"[DEBUG] 使用默认图片: {image_path}")
            if not os.path.exists(image_path):
                raise HTTPException(status_code=400, detail=f"默认图片不存在，也未上传图片 (路径: {image_path})")
        else:
            # 处理上传图片
            os.makedirs("orc_upload", exist_ok=True)
            file_ext = os.path.splitext(file.filename)[-1]
            filename = f"{uuid.uuid4().hex}{file_ext}"
            image_path = os.path.join("orc_upload", filename)
            with open(image_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            print(f"[DEBUG] 上传图片保存路径: {image_path}")

        # 调用 OCR 处理
        result = ocr_pipeline(image_path)

        # 直接返回所有结果，包括 result_image_path
        return JSONResponse(content={"status": "success", "result": result})

    except HTTPException as he:
        # 直接抛出 HTTPException
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR 处理失败: {str(e)}")



if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8080, reload=True)
