import base64
import gc
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import runpod
import torch
from spandrel import ImageModelDescriptor, ModelLoader

MODEL_DIR = Path("/models")
TEMP_DIR = Path(tempfile.gettempdir()) / "rp_video_enhancer"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

LOADED_MODELS: Dict[int, Any] = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_preloaded_model(scale: int):
    """Load pre-baked RealESRGAN_x2.pth or RealESRGAN_x4.pth from /models into VRAM once."""
    scale = 2 if scale == 2 else 4
    if scale not in LOADED_MODELS:
        filename = "RealESRGAN_x2.pth" if scale == 2 else "RealESRGAN_x4.pth"
        model_path = MODEL_DIR / filename
        if not model_path.exists():
            from huggingface_hub import hf_hub_download

            model_path = Path(
                hf_hub_download(repo_id="ai-forever/Real-ESRGAN", filename=filename)
            )
        descriptor = ModelLoader().load_from_file(str(model_path))
        assert isinstance(descriptor, ImageModelDescriptor)
        m = descriptor.model.to(DEVICE).eval()
        if DEVICE.type == "cuda" and getattr(descriptor, "supports_half", True):
            m = m.half()
        LOADED_MODELS[scale] = m
    return LOADED_MODELS[scale]


def check_nvenc() -> bool:
    """Verify whether h264_nvenc hardware encoding is supported on this GPU worker."""
    try:
        p = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=256x256:d=0.1",
                "-c:v",
                "h264_nvenc",
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        return p.returncode == 0
    except Exception:
        return False


NVENC_AVAILABLE = check_nvenc()


def probe_video(video_path: str) -> Tuple[float, int, int, float, bool]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=index,codec_type,width,height,duration,r_frame_rate:format=duration",
        "-of",
        "json",
        video_path,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    width = height = 0
    duration = 0.0
    fps = 30.0
    has_audio = False

    if p.returncode == 0:
        data = json.loads(p.stdout.decode("utf-8", errors="replace") or "{}")
        for s in data.get("streams", []):
            ctype = s.get("codec_type")
            if ctype == "video" and width == 0:
                width = int(s.get("width") or 0)
                height = int(s.get("height") or 0)
                if s.get("duration"):
                    try:
                        duration = float(s["duration"])
                    except ValueError:
                        pass
                r_fps = s.get("r_frame_rate", "30/1")
                try:
                    if "/" in r_fps:
                        num, den = r_fps.split("/")
                        if float(den) > 0:
                            fps = float(num) / float(den)
                    else:
                        fps = float(r_fps)
                except ValueError:
                    pass
            elif ctype == "audio":
                has_audio = True

        if duration <= 0:
            fmt_dur = data.get("format", {}).get("duration")
            if fmt_dur:
                try:
                    duration = float(fmt_dur)
                except ValueError:
                    pass

    if duration <= 0 or width == 0 or height == 0:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            if width == 0:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cv_fps = cap.get(cv2.CAP_PROP_FPS)
            if cv_fps and cv_fps > 0 and not np.isnan(cv_fps):
                fps = float(cv_fps)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if duration <= 0 and frame_count and frame_count > 0 and fps > 0:
                duration = float(frame_count) / fps
        cap.release()

    return duration, width, height, fps, has_audio


def decode_frames_to_memory(video_path: str, max_frames: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames_rgb: List[np.ndarray] = []
    while len(frames_rgb) < max_frames:
        ret, bgr = cap.read()
        if not ret or bgr is None:
            break
        frames_rgb.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames_rgb


def enhance_frames_on_gpu(
    frames_rgb: List[np.ndarray], scale: int, batch_size: int | None = None
) -> Tuple[List[np.ndarray], int]:
    model = get_preloaded_model(scale)
    total = len(frames_rgb)
    h, w = frames_rgb[0].shape[:2]

    if batch_size is None:
        if DEVICE.type == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            pixels = w * h
            if vram_gb >= 80:
                batch_size = 8 if pixels <= 1920 * 1080 else 4
            elif vram_gb >= 30:
                batch_size = 4 if pixels <= 1920 * 1080 else 2
            elif vram_gb >= 20:
                batch_size = 2 if pixels <= 1920 * 1080 else 1
            else:
                batch_size = 1
        else:
            batch_size = 1

    enhanced: List[np.ndarray] = []
    with torch.inference_mode():
        for start_idx in range(0, total, batch_size):
            batch_np = np.stack(frames_rgb[start_idx : start_idx + batch_size], axis=0)
            tensor = torch.from_numpy(batch_np).to(device=DEVICE, non_blocking=True)
            tensor = tensor.permute(0, 3, 1, 2).to(dtype=torch.float16).mul_(1.0 / 255.0)
            output = model(tensor)
            out_u8 = (
                output.clamp_(0, 1)
                .mul_(255.0)
                .byte()
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            for b in range(out_u8.shape[0]):
                enhanced.append(out_u8[b])

    torch.cuda.synchronize()
    return enhanced, batch_size


def encode_frames_from_memory(
    frames_rgb: List[np.ndarray],
    audio_src: str,
    has_audio: bool,
    out_path: str,
    fps: float = 30.0,
) -> str:
    out_h, out_w = frames_rgb[0].shape[:2]
    pad_w = out_w - (out_w % 2)
    pad_h = out_h - (out_h % 2)

    encoder_used = "h264_nvenc" if NVENC_AVAILABLE else "libx264"
    vcodec_args = (
        ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19", "-pix_fmt", "yuv420p"]
        if NVENC_AVAILABLE
        else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{out_w}x{out_h}",
        "-r",
        f"{fps:.4f}",
        "-i",
        "pipe:0",
    ]
    if has_audio:
        cmd.extend(["-i", audio_src])
    if pad_w != out_w or pad_h != out_h:
        cmd.extend(["-vf", f"crop={pad_w}:{pad_h}:0:0"])
    cmd.extend(vcodec_args)
    if has_audio:
        cmd.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-shortest",
            ]
        )
    cmd.extend(["-movflags", "+faststart", out_path])

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for frame in frames_rgb:
            proc.stdin.write(memoryview(np.ascontiguousarray(frame)))
        proc.stdin.close()
    except BrokenPipeError:
        pass

    stderr_bytes = proc.stderr.read() if proc.stderr else b""
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"FFmpeg encode failed: {stderr_bytes.decode('utf-8', errors='replace')}")
    return encoder_used


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    t_total_start = time.perf_counter()
    inp = job.get("input", {})
    scale = int(inp.get("scale", 4))
    max_seconds = float(inp.get("max_seconds", 16.0))
    batch_size_override = inp.get("batch_size")
    return_video_base64 = bool(inp.get("return_video_base64", True))

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram_gb = (
        round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        if torch.cuda.is_available()
        else 0.0
    )

    job_dir = TEMP_DIR / f"job_{int(time.time() * 1000)}"
    job_dir.mkdir(parents=True, exist_ok=True)
    in_video = job_dir / "input.mp4"
    out_video = job_dir / "output_enhanced.mp4"

    try:
        if "video_base64" in inp and inp["video_base64"]:
            in_video.write_bytes(base64.b64decode(inp["video_base64"]))
        elif "video_url" in inp and inp["video_url"]:
            req = urllib.request.Request(inp["video_url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                in_video.write_bytes(r.read())
        else:
            raise ValueError("Provide either 'video_base64' or 'video_url' in input.")

        duration, w, h, fps, has_audio = probe_video(str(in_video))
        max_frames = max(1, int(round(fps * max_seconds)))

        t_dec0 = time.perf_counter()
        frames_rgb = decode_frames_to_memory(str(in_video), max_frames)
        decode_sec = time.perf_counter() - t_dec0
        if not frames_rgb:
            raise RuntimeError("No frames decoded from input video.")

        num_frames = len(frames_rgb)
        torch.cuda.reset_peak_memory_stats()
        t_gpu0 = time.perf_counter()
        enhanced_frames, used_batch_size = enhance_frames_on_gpu(
            frames_rgb,
            scale=scale,
            batch_size=int(batch_size_override) if batch_size_override else None,
        )
        gpu_sec = time.perf_counter() - t_gpu0
        peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 2)

        out_h, out_w = enhanced_frames[0].shape[:2]
        del frames_rgb
        gc.collect()

        t_enc0 = time.perf_counter()
        encoder_used = encode_frames_from_memory(
            enhanced_frames, str(in_video), has_audio, str(out_video), fps
        )
        encode_sec = time.perf_counter() - t_enc0

        del enhanced_frames
        gc.collect()
        torch.cuda.empty_cache()

        total_sec = time.perf_counter() - t_total_start
        res: Dict[str, Any] = {
            "gpu_name": gpu_name,
            "vram_gb": vram_gb,
            "peak_vram_gb": peak_vram_gb,
            "input_resolution": f"{w}x{h}",
            "output_resolution": f"{out_w}x{out_h}",
            "scale": scale,
            "fps_source": round(fps, 2),
            "frames_processed": num_frames,
            "batch_size": used_batch_size,
            "encoder_used": encoder_used,
            "decode_sec": round(decode_sec, 3),
            "gpu_inference_sec": round(gpu_sec, 3),
            "gpu_fps": round(num_frames / gpu_sec, 2),
            "encode_sec": round(encode_sec, 3),
            "total_sec": round(total_sec, 3),
            "output_size_mb": round(out_video.stat().st_size / (1024**2), 2),
        }
        if return_video_base64:
            res["video_base64"] = base64.b64encode(out_video.read_bytes()).decode("ascii")
        return res
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    # Warmup and pre-load both 2x and 4x models into VRAM before claiming jobs
    try:
        if torch.cuda.is_available():
            get_preloaded_model(2)
            get_preloaded_model(4)
            print("Pre-loaded RealESRGAN 2x & 4x models into VRAM successfully.", flush=True)
    except Exception as e:
        print(f"Preload warning: {e}", flush=True)

    runpod.serverless.start({"handler": handler})
