FROM python:3.11-slim

RUN sed -i 's|http://deb.debian.org|https://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch \
    --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu/ \
    --trusted-host mirrors.aliyun.com

RUN pip install --no-cache-dir \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
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