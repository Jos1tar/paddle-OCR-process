# 使用 Python 官方轻量镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装基础构建依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 拷贝依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 拷贝项目代码
COPY . .

# 运行 FastAPI 接口（修改 fastapi:app 为你的模块名:app）
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]