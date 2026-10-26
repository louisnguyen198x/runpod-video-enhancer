FROM ghcr.io/louisnguyen198x/runpod-video-enhancer:latest

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
