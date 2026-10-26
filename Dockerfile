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
    imageio \
    tqdm \
    opencv-python-headless && \
    pip install --no-cache-dir --no-deps \
    spandrel \
    spandrel_extra_arches \
    safetensors \
    einops \
    typing_extensions

RUN mkdir -p /models && \
    curl -L -sS "https://huggingface.co/Shandypur/ESRGAN-4x-UltraSharp/resolve/main/4x-UltraSharp.pth" -o "/models/4x-UltraSharp.pth" && \
    curl -L -sS "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth" -o "/models/4x_foolhardy_Remacri.pth" && \
    curl -L -sS "https://huggingface.co/danhtran2mind/Real-ESRGAN-Anime-finetuning/resolve/main/Real-ESRGAN-Anime-finetuning.pth" -o "/models/Real-ESRGAN-Anime-finetuning.pth" && \
    curl -L -sS "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/AnimeSharp%204x.pth" -o "/models/AnimeSharp4x.pth" && \
    curl -L -sS "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/RealisticRescaler%204x.pth" -o "/models/RealisticRescaler4x.pth" && \
    curl -L -sS "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/Remacri%204x%20ExtraSmoother.pth" -o "/models/Remacri4xExtraSmoother.pth" && \
    curl -L -sS "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/lollypop.pth" -o "/models/lollypop.pth" && \
    curl -L -sS "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/NickelbackFS%204x.pth" -o "/models/NickelbackFS4x.pth" && \
    curl -L -sS "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth" -o "/models/RealESRNet_x4plus.pth" && \
    curl -L -sS "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth" -o "/models/R-ESRGAN_General_WDN_4xV3.pth" && \
    curl -L -sS "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth" -o "/models/R-ESRGAN_AnimeVideo.pth" && \
    curl -L -sS "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" -o "/models/R-ESRGAN_4x+.pth" && \
    curl -L -sS "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth" -o "/models/R-ESRGAN_4x+_Anime6B.pth" && \
    curl -L -sS "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth" -o "/models/ScuNET_GAN.pth" && \
    curl -L -sS "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth" -o "/models/ScuNET_PSNR.pth" && \
    ls -lh /models

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
