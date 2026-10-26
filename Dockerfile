FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod \
    huggingface_hub \
    opencv-python-headless && \
    pip install --no-cache-dir --no-deps \
    spandrel \
    safetensors \
    einops \
    typing_extensions

RUN mkdir -p /models && \
    curl -L -sS "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x2.pth" -o /models/RealESRGAN_x2.pth && \
    curl -L -sS "https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth" -o /models/RealESRGAN_x4.pth && \
    ls -lh /models

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
