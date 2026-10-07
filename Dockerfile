FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 替换 Ubuntu 源为清华镜像，并移除过期的 NVIDIA 仓库列表
RUN sed -i 's|http://archive.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list && \
    rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

# 安装 Python 3.11 和必要的系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

COPY requirements.txt .

# 1. 安装 GPU 版 PyTorch 2.5.1（使用上海交大镜像源，之前已验证可用）
RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 \
    torchvision==0.20.1+cu121 \
    torchaudio==2.5.1+cu121 \
    -i https://mirror.sjtu.edu.cn/pytorch-wheels/cu121/

# 2. 创建约束文件，锁定 torch 版本，防止被后续依赖修改
RUN printf "torch==2.5.1\ntorchvision==0.20.1\ntorchaudio==2.5.1\ntransformers==4.51.0\nsentence-transformers==3.1.1\n" > /app/pins.txt

# 3. 安装其余依赖，并将 transformers 升级到能识别 Qwen3 的 4.51.0
RUN pip install --no-cache-dir \
     -c /app/pins.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    transformers==4.51.0 \
    sentence-transformers==3.1.1 \
    accelerate \
    -r requirements.txt

COPY app/ ./app/
COPY scripts/ ./scripts/
COPY run_api.py .

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
RUN mkdir -p /data/chroma_db && chown -R appuser:appuser /data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.api.app:app", "--host", "0.0.0.0", "--port", "8000"]