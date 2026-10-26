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
import imageio
import numpy as np
import runpod
import spandrel
import torch
import torch.nn.functional as F

try:
    import spandrel_extra_arches

    spandrel.MAIN_REGISTRY.add(*spandrel_extra_arches.EXTRA_REGISTRY)
except Exception:
    pass

MODEL_DIR = Path("/models")
SHM_BASE = Path("/dev/shm") if Path("/dev/shm").exists() else Path(tempfile.gettempdir())
TEMP_DIR = SHM_BASE / "rp_video_upscaler_stablepy"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

FFMPEG_BIN = "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg"
FFPROBE_BIN = "/usr/bin/ffprobe" if os.path.exists("/usr/bin/ffprobe") else "ffprobe"

UPSCALER_CATALOG: Dict[str, Tuple[str, str]] = {
    "4x-UltraSharp": (
        "4x-UltraSharp.pth",
        "https://huggingface.co/Shandypur/ESRGAN-4x-UltraSharp/resolve/main/4x-UltraSharp.pth",
    ),
    "4x_foolhardy_Remacri": (
        "4x_foolhardy_Remacri.pth",
        "https://huggingface.co/FacehugmanIII/4x_foolhardy_Remacri/resolve/main/4x_foolhardy_Remacri.pth",
    ),
    "Real-ESRGAN-Anime-finetuning": (
        "Real-ESRGAN-Anime-finetuning.pth",
        "https://huggingface.co/danhtran2mind/Real-ESRGAN-Anime-finetuning/resolve/main/Real-ESRGAN-Anime-finetuning.pth",
    ),
    "AnimeSharp4x": (
        "AnimeSharp4x.pth",
        "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/AnimeSharp%204x.pth",
    ),
    "RealisticRescaler4x": (
        "RealisticRescaler4x.pth",
        "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/RealisticRescaler%204x.pth",
    ),
    "Remacri4xExtraSmoother": (
        "Remacri4xExtraSmoother.pth",
        "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/Remacri%204x%20ExtraSmoother.pth",
    ),
    "lollypop": (
        "lollypop.pth",
        "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/lollypop.pth",
    ),
    "NickelbackFS4x": (
        "NickelbackFS4x.pth",
        "https://huggingface.co/hollowstrawberry/upscalers-backup/resolve/main/ESRGAN/NickelbackFS%204x.pth",
    ),
    "RealESRNet_x4plus": (
        "RealESRNet_x4plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth",
    ),
    "R-ESRGAN General WDN 4xV3": (
        "R-ESRGAN_General_WDN_4xV3.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    ),
    "R-ESRGAN AnimeVideo": (
        "R-ESRGAN_AnimeVideo.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    ),
    "R-ESRGAN 4x+": (
        "R-ESRGAN_4x+.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ),
    "R-ESRGAN 4x+ Anime6B": (
        "R-ESRGAN_4x+_Anime6B.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    ),
    "ScuNET GAN": (
        "ScuNET_GAN.pth",
        "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_gan.pth",
    ),
    "ScuNET PSNR": (
        "ScuNET_PSNR.pth",
        "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
    ),
}

LOADED_DESCRIPTORS: Dict[Tuple[str, bool], spandrel.ModelDescriptor] = {}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_descriptor(upscaler_name: str, half_precision: bool = True) -> spandrel.ModelDescriptor:
    if upscaler_name not in UPSCALER_CATALOG:
        upscaler_name = "4x-UltraSharp"

    key = (upscaler_name, half_precision)
    if key not in LOADED_DESCRIPTORS:
        if len(LOADED_DESCRIPTORS) >= 2:
            LOADED_DESCRIPTORS.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        fname, url = UPSCALER_CATALOG[upscaler_name]
        local_path = MODEL_DIR / fname
        if not local_path.exists():
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, str(local_path))

        desc = spandrel.ModelLoader(device=DEVICE).load_from_file(str(local_path))
        desc.model.eval()
        if DEVICE.type == "cuda" and half_precision and getattr(desc, "supports_half", True):
            desc.model.half()
        else:
            desc.model.float()
        LOADED_DESCRIPTORS[key] = desc

    return LOADED_DESCRIPTORS[key]


def check_nvenc() -> bool:
    try:
        p = subprocess.run(
            [
                FFMPEG_BIN,
                "-hide_banner",
                "-loglevel",
                "error",
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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return p.returncode == 0
    except Exception:
        return False


NVENC_AVAILABLE = check_nvenc()


def has_audio_stream(video_path: str) -> bool:
    try:
        p = subprocess.run(
            [
                FFPROBE_BIN,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=noprint_wrappers=1",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return bool(p.stdout.decode("utf-8", errors="replace").strip())
    except Exception:
        return False


def read_video_frames(video_path: str, max_seconds: float = 16.0) -> Tuple[List[np.ndarray], float, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240 or np.isnan(fps):
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width > 1920 or height > 1920:
        cap.release()
        raise ValueError(f"Video resolution too high ({width}x{height}). Maximum allowed is 1920x1920.")

    max_allowed_frames = max(1, int(round(fps * max_seconds)))
    raw_frames: List[np.ndarray] = []
    while len(raw_frames) < max_allowed_frames:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            break
        raw_frames.append(frame_bgr)
    cap.release()
    return raw_frames, float(fps), width, height


def upscale_frames_gpu(
    raw_frames_bgr: List[np.ndarray],
    upscaler_name: str = "4x-UltraSharp",
    upscaler_factor: float = 2.0,
    half_precision: bool = True,
    batch_size: int | None = None,
) -> Tuple[List[np.ndarray], int]:
    desc = get_model_descriptor(upscaler_name, half_precision=half_precision)
    model = desc.model
    use_half = (
        DEVICE.type == "cuda"
        and half_precision
        and getattr(desc, "supports_half", True)
    )
    native_scale = getattr(desc, "scale", 4) or 4

    total = len(raw_frames_bgr)
    h, w = raw_frames_bgr[0].shape[:2]
    target_w = int(round(w * upscaler_factor))
    target_h = int(round(h * upscaler_factor))
    target_w -= target_w % 2
    target_h -= target_h % 2

    if batch_size is None:
        if DEVICE.type == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            pixels = w * h
            if vram_gb >= 80:
                batch_size = 8 if pixels <= 1920 * 1080 else 4
            elif vram_gb >= 30:
                batch_size = 4 if pixels <= 1920 * 1080 else 2
            elif vram_gb >= 20:
                batch_size = 4 if pixels <= 960 * 540 else (2 if pixels <= 1920 * 1080 else 1)
            else:
                batch_size = 1
        else:
            batch_size = 1

    enhanced_bgr: List[np.ndarray] = []
    need_resize = abs(upscaler_factor - native_scale) > 0.05 or (
        (h * native_scale) != target_h or (w * native_scale) != target_w
    )

    with torch.inference_mode():
        for start_idx in range(0, total, batch_size):
            batch_np = np.stack(raw_frames_bgr[start_idx : start_idx + batch_size], axis=0)
            t = torch.from_numpy(batch_np).to(device=DEVICE, non_blocking=True)
            t = t.flip(-1).permute(0, 3, 1, 2).to(
                dtype=torch.float16 if use_half else torch.float32
            ).mul_(1.0 / 255.0)

            out_t = model(t)

            if need_resize:
                out_t = F.interpolate(
                    out_t,
                    size=(target_h, target_w),
                    mode="bicubic" if target_w > out_t.shape[-1] else "area",
                    antialias=(target_w > out_t.shape[-1]),
                )

            out_u8_bgr = (
                out_t.clamp_(0, 1)
                .mul_(255.0)
                .byte()
                .permute(0, 2, 3, 1)
                .flip(-1)
                .cpu()
                .numpy()
            )
            for b in range(out_u8_bgr.shape[0]):
                enhanced_bgr.append(out_u8_bgr[b])

    torch.cuda.synchronize()
    return enhanced_bgr, batch_size


def write_output_video(
    frames_bgr: List[np.ndarray],
    src_video_path: str,
    output_path: str,
    fps: float,
    speed_factor: float = 1.0,
    as_gif: bool = False,
) -> str:
    new_fps = fps * speed_factor if speed_factor > 0 else fps
    if as_gif:
        frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        imageio.mimsave(output_path, frames_rgb, duration=(1.0 / new_fps))
        return "imageio_gif"

    out_h, out_w = frames_bgr[0].shape[:2]

    # Step 1: Write intermediate stream into /dev/shm RAM disk (0 pipe deadlock, ~0.3s)
    shm_raw_mp4 = str(Path(output_path).parent / "shm_raw_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(shm_raw_mp4, fourcc, new_fps, (out_w, out_h))
    for frame in frames_bgr:
        vw.write(frame)
    vw.release()

    # Step 2: Transcode to H.264 (NVENC or libx264) + mux original audio cleanly
    keep_audio = abs(speed_factor - 1.0) < 0.05 and has_audio_stream(src_video_path)
    encoder_used = "h264_nvenc" if NVENC_AVAILABLE else "libx264"
    vcodec_args = (
        ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19", "-pix_fmt", "yuv420p"]
        if NVENC_AVAILABLE
        else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
    )

    cmd = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error", "-i", shm_raw_mp4]
    if keep_audio:
        cmd.extend(["-i", src_video_path])
    cmd.extend(vcodec_args)
    if keep_audio:
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
    cmd.extend(["-movflags", "+faststart", output_path])

    p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
    if p.returncode != 0:
        # Fallback to libx264 if NVENC failed on specific resolution
        if encoder_used == "h264_nvenc":
            encoder_used = "libx264"
            cmd_fb = [FFMPEG_BIN, "-y", "-hide_banner", "-loglevel", "error", "-i", shm_raw_mp4]
            if keep_audio:
                cmd_fb.extend(["-i", src_video_path])
            cmd_fb.extend(["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"])
            if keep_audio:
                cmd_fb.extend(["-c:a", "aac", "-b:a", "192k", "-map", "0:v:0", "-map", "1:a:0?", "-shortest"])
            cmd_fb.extend(["-movflags", "+faststart", output_path])
            p2 = subprocess.run(cmd_fb, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
            if p2.returncode != 0:
                shutil.move(shm_raw_mp4, output_path)
                encoder_used = "mp4v_direct"
        else:
            shutil.move(shm_raw_mp4, output_path)
            encoder_used = "mp4v_direct"

    if os.path.exists(shm_raw_mp4):
        os.remove(shm_raw_mp4)
    return encoder_used


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    t_total_start = time.perf_counter()
    inp = job.get("input", {})

    upscaler_name = str(inp.get("upscaler_name", "4x-UltraSharp"))
    upscaler_factor = float(inp.get("upscaler_factor", 2.0))
    as_gif = bool(inp.get("as_gif", False))
    speed_factor = float(inp.get("speed_factor", 1.0))
    half_precision = bool(inp.get("half_precision", True))
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
    ext = "gif" if as_gif else "mp4"
    in_video = job_dir / "input.mp4"
    out_video = job_dir / f"upscaled.{ext}"

    try:
        if "video_base64" in inp and inp["video_base64"]:
            in_video.write_bytes(base64.b64decode(inp["video_base64"]))
        elif "video_url" in inp and inp["video_url"]:
            req = urllib.request.Request(inp["video_url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                in_video.write_bytes(r.read())
        else:
            raise ValueError("Provide either 'video_base64' or 'video_url' in input.")

        t_dec0 = time.perf_counter()
        raw_frames, fps, w, h = read_video_frames(str(in_video), max_seconds=max_seconds)
        decode_sec = time.perf_counter() - t_dec0
        if not raw_frames:
            raise RuntimeError("No frames decoded from input video.")

        num_frames = len(raw_frames)
        torch.cuda.reset_peak_memory_stats()
        t_gpu0 = time.perf_counter()
        enhanced_frames, used_batch_size = upscale_frames_gpu(
            raw_frames,
            upscaler_name=upscaler_name,
            upscaler_factor=upscaler_factor,
            half_precision=half_precision,
            batch_size=int(batch_size_override) if batch_size_override else None,
        )
        gpu_sec = time.perf_counter() - t_gpu0
        peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024**3), 2)

        out_h, out_w = enhanced_frames[0].shape[:2]
        del raw_frames
        gc.collect()

        t_enc0 = time.perf_counter()
        encoder_used = write_output_video(
            enhanced_frames,
            str(in_video),
            str(out_video),
            fps=fps,
            speed_factor=speed_factor,
            as_gif=as_gif,
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
            "upscaler_name": upscaler_name,
            "upscaler_factor": upscaler_factor,
            "input_resolution": f"{w}x{h}",
            "output_resolution": f"{out_w}x{out_h}",
            "fps_source": round(fps, 2),
            "frames_processed": num_frames,
            "batch_size": used_batch_size,
            "encoder_used": encoder_used,
            "decode_sec": round(decode_sec, 3),
            "gpu_inference_sec": round(gpu_sec, 3),
            "gpu_fps": round(num_frames / gpu_sec, 2),
            "encode_sec": round(encode_sec, 3),
            "total_sec": round(total_sec, 3),
            "output_format": ext,
            "output_size_mb": round(out_video.stat().st_size / (1024**2), 2),
        }
        if return_video_base64:
            res["video_base64"] = base64.b64encode(out_video.read_bytes()).decode("ascii")
        return res
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        if torch.cuda.is_available():
            get_model_descriptor("4x-UltraSharp", half_precision=True)
            print("Pre-loaded default 4x-UltraSharp into VRAM.", flush=True)
    except Exception as e:
        print(f"Preload warning: {e}", flush=True)

    runpod.serverless.start({"handler": handler})
