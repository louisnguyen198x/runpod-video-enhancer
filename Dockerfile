FROM ghcr.io/louisnguyen198x/runpod-video-enhancer:latest

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
