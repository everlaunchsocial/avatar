"""RunPod Serverless Handler for HunyuanVideo-Avatar renders."""
import os
import subprocess
import time
import requests
from pathlib import Path

import runpod

WORKSPACE = Path("/workspace/HunyuanVideo-Avatar")
CKPT_REL = "./weights/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
FRAMES_MAP = {"5s": 129, "10s": 257, "15s": 385, "30s": 769}


def download_file(url, dest):
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for c in r.iter_content(262144):
            f.write(c)
    return dest


def handler(job):
    job_input = job["input"]
    image_url = job_input.get("image_url")
    audio_url = job_input.get("audio_url")
    settings = {k: v for k, v in job_input.items() if k not in ("image_url", "audio_url")}

    if not image_url or not audio_url:
        return {"error": "image_url and audio_url are required"}

    job_id = job.get("id", "test")
    tmp_dir = WORKSPACE / "assets" / "serverless_tmp" / job_id
    out_dir = WORKSPACE / "results" / "serverless" / job_id
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download inputs
        img_ext = image_url.split("?")[0].rsplit(".", 1)[-1].lower()
        if img_ext not in ("jpg", "jpeg", "png"):
            img_ext = "jpg"
        aud_ext = audio_url.split("?")[0].rsplit(".", 1)[-1].lower()
        if aud_ext not in ("wav", "mp3"):
            aud_ext = "wav"

        image_path = download_file(image_url, tmp_dir / f"input.{img_ext}")
        audio_path = download_file(audio_url, tmp_dir / f"input.{aud_ext}")

        # Build render command
        steps = int(settings.get("inference_steps", 50))
        cfg = float(settings.get("cfg_scale", 7.5))
        flow_shift = float(settings.get("flow_shift", 5.0))
        seed_val = settings.get("seed")
        seed = int(seed_val) if seed_val not in (None, "", 0) else 128
        length = settings.get("video_length", "5s")
        num_frames = FRAMES_MAP.get(length, 129)
        prompt = (settings.get("prompt") or "A person speaking naturally and confidently").replace(",", " ").strip()

        rel_image = image_path.relative_to(WORKSPACE)
        rel_audio = audio_path.relative_to(WORKSPACE)
        csv_path = tmp_dir / "input.csv"
        csv_path.write_text(f"videoid,image,audio,prompt,fps\n1,{rel_image},{rel_audio},{prompt},25\n")

        cmd = [
            "python3", "hymm_sp/sample_gpu_poor.py",
            "--input", str(csv_path.relative_to(WORKSPACE)),
            "--ckpt", CKPT_REL,
            "--sample-n-frames", str(num_frames),
            "--seed", str(seed),
            "--image-size", "704",
            "--cfg-scale", str(cfg),
            "--infer-steps", str(steps),
            "--use-deepcache", "1",
            "--flow-shift-eval-video", str(flow_shift),
            "--save-path", str(out_dir.relative_to(WORKSPACE)),
            "--use-fp8",
        ]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(WORKSPACE)
        env["MODEL_BASE"] = str(WORKSPACE / "weights")
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["CUDA_VISIBLE_DEVICES"] = "0"

        start = time.time()
        result = subprocess.run(cmd, cwd=str(WORKSPACE), env=env, capture_output=True, text=True, timeout=1800)

        if result.returncode != 0:
            return {"error": f"render failed: {(result.stderr or result.stdout or '')[-1000:]}"}

        # Find output video
        mp4s = sorted(out_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not mp4s:
            return {"error": "no video output produced"}

        elapsed_ms = int((time.time() - start) * 1000)

        # Read video bytes and return as base64 or URL
        # For now, return the file path (RunPod can serve files from the worker)
        video_path = mp4s[0]
        video_size = video_path.stat().st_size

        return {
            "status": "done",
            "render_time_ms": elapsed_ms,
            "video_size_bytes": video_size,
            "video_path": str(video_path),
            "message": f"Rendered in {elapsed_ms}ms, {video_size} bytes"
        }

    except subprocess.TimeoutExpired:
        return {"error": "render timed out after 1800s"}
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            for p in tmp_dir.iterdir():
                p.unlink()
            tmp_dir.rmdir()
        except Exception:
            pass


runpod.serverless.start({"handler": handler})
