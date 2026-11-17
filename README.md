# PP-OCRv3 inference service

A FastAPI service that runs the three-stage PP-OCRv3 pipeline (text detection, direction classification,
text recognition) with Paddle Inference. The pre- and post-processing of each stage is reimplemented in
this repo instead of calling the `paddleocr` package.

## Pipeline

1. Detection (DB): resize so the long side is at most 960 and both sides are multiples of 32, ImageNet
   normalization. The probability map is thresholded, contours are turned into minimum-area boxes, scored,
   and expanded with the DB unclip step (pyclipper offset).
2. Crop: each box is perspective-warped into an upright crop.
3. Classification: 48x192 input; crops predicted as rotated 180 degrees are flipped.
4. Recognition: height 32, width padded up to 320, CTC greedy decoding with `ppocr_keys_v1.txt`
   (about 6,600 Chinese and English characters).

## Models

Download the PaddleOCR inference models and unpack them next to `app.py`:

```
ch_PP-OCRv3_det_infer/
ch_ppocr_mobile_v2.0_cls_infer/
ch_PP-OCRv3_rec_infer/
```

Each folder should contain `inference.pdmodel` and `inference.pdiparams`. They are available from the
PaddleOCR model list (PP-OCRv3 Chinese models).

## Run

```
pip install -r requirements.txt     # Python 3.8 - 3.10 (paddlepaddle 2.6)
uvicorn app:app --port 8000
```

```
curl -F "file=@ocr_upload/gpt.png" http://127.0.0.1:8000/ocrpredict/
```

The response contains the recognized lines, the number of boxes and the path of an image with the detected
boxes drawn on it (`result/debug_boxes.jpg`). Set `OCR_USE_GPU=1` to run on GPU.

Without the server: `python ini.py` runs the pipeline on `ocr_upload/gpt.png`.

Docker:

```
docker build -t ppocr-service .
docker run -p 8000:8000 ppocr-service
```

## Files

```
app.py              FastAPI endpoint /ocrpredict/
ini.py              model loading and the det -> cls -> rec pipeline
pre_processor.py    resize / normalize / pad operators for each stage
post_processor.py   DB box extraction and unclip, cls decoding, CTC decoding
ppocr_keys_v1.txt   recognition character set
ocr_upload/         sample images
```

The `onnx` branch contains a version of the pipeline that runs the same models exported to ONNX.

## Changes in Nov 2026

- `app.py` imported `ocr_pipeline` from a `test.py` that no longer exists; it now imports from `ini.py`.
  The upload folder name was misspelled (`orc_upload`) and the endpoint is now `/ocrpredict/`.
- The three models were reloaded on every request; they are now loaded once and cached.
- GPU was hard-coded, which failed on CPU-only machines and in the container; it is now opt-in via `OCR_USE_GPU`.
- The character set path is resolved relative to the script.
- `requirements.txt` was saved as UTF-16 and could not be read by pip in some setups; it is now UTF-8.
- The Dockerfile installed `libgl1-mesa-glx`, which no longer exists in Debian 12 based images.
