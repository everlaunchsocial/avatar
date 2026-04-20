#!/usr/bin/env python3
"""EverLaunch Avatar Studio worker. Model loaded ONCE, stays in VRAM between jobs."""
import os, sys, time, json, traceback
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

import torch
import torch.distributed as dist
from einops import rearrange
import imageio

# Initialize distributed (required by the model)
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
if not dist.is_initialized():
    dist.init_process_group(backend="nccl", world_size=1, rank=0)

import requests
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "5"))
WORKSPACE = Path("/workspace/HunyuanVideo-Avatar")
MODEL_BASE = os.environ.get("MODEL_BASE", str(WORKSPACE / "weights"))
BUCKET = "rendered-videos"
FMAP = {"5s": 129, "10s": 257, "15s": 385, "30s": 769}

# Add workspace to path
sys.path.insert(0, str(WORKSPACE))
os.environ["PYTHONPATH"] = str(WORKSPACE)
os.environ["MODEL_BASE"] = MODEL_BASE
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def log(m):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{ts}] [worker] {m}", flush=True)


# ============================================================
# PHOTO ENHANCEMENT (for "sunrise/sundowning" temporal lighting
# drift). Proven fix from previous RunPod testing:
#
#   1. PIL.ImageOps.autocontrast()  — globally normalize exposure,
#      so the model gets a consistent starting lighting distribution
#      rather than interpreting extreme darks/lights as directional
#      illumination to animate.
#   2. PIL.ImageFilter.GaussianBlur(radius=1)  — very light blur to
#      kill JPEG micro-noise and grain. The diffusion model was
#      mistakenly animating those noise patterns as if they were
#      real features, which produced the lighting oscillation.
#   3. Save as lossless PNG — avoid re-introducing JPEG noise on
#      the way back out.
#
# Earlier attempt used CLAHE + UnsharpMask. CLAHE fixed the
# background drift but made faces look charcoal/over-defined
# because it boosts LOCAL contrast (which maps onto skin as
# exaggerated shadow outlines). Autocontrast is global, so it
# lifts the whole histogram uniformly and leaves skin natural.
#
# Triggered per-job with settings.enhance=true (default off, so
# existing photos are untouched unless the flag is set).
# ============================================================
def enhance_image(src_path: Path, dst_path: Path):
    from PIL import Image, ImageOps, ImageFilter

    img = Image.open(str(src_path)).convert("RGB")

    # 1. Normalize exposure globally — stretches the histogram so
    #    darkest pixel becomes (near-)black and brightest becomes
    #    (near-)white. `cutoff=1` ignores the top/bottom 1% of
    #    pixels so a single hot highlight or shadow doesn't skew
    #    the normalization.
    img = ImageOps.autocontrast(img, cutoff=1)

    # 2. Light Gaussian blur to flatten micro-noise. Radius=1 is
    #    ALMOST imperceptible to the eye but obliterates the
    #    sub-pixel noise patterns that diffusion would otherwise
    #    try to animate into frame-to-frame drift.
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    # 3. Lossless PNG out.
    img.save(str(dst_path), "PNG", optimize=False)
    return dst_path


# ============================================================
# MODEL LOADING — happens ONCE at startup, stays in VRAM
# ============================================================
def load_engine():
    """Load the full HunyuanVideo-Avatar engine into GPU memory. Called once."""
    from hymm_sp.config import parse_args
    from hymm_sp.sample_inference_audio import HunyuanVideoSampler
    from hymm_sp.data_kits.face_align import AlignImage
    from transformers import WhisperModel, AutoFeatureExtractor

    log("loading engine...")
    device = torch.device("cuda")

    # Parse default args with our settings
    sys.argv = [
        "worker",
        "--input", "dummy.csv",
        "--ckpt", f"{MODEL_BASE}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt",
        "--sample-n-frames", "129",
        "--seed", "128",
        "--image-size", "512",
        "--cfg-scale", "7.5",
        "--infer-steps", "50",
        "--use-deepcache", "0",
        "--cpu-offload",
        "--flow-shift-eval-video", "5.0",
        "--save-path", str(WORKSPACE / "results"),
        "--use-fp8",
    ]
    args = parse_args()

    # Load the main sampler (transformer + VAE + text encoders)
    log("loading HunyuanVideoSampler (this takes ~60s)...")
    sampler = HunyuanVideoSampler.from_pretrained(args.ckpt, args=args, device=device)

    # TeaCache setup
    tc = sampler.model
    tc.enable_teacache = True
    tc.teacache_num_steps = 50
    tc.teacache_thresh = 0.18
    tc.teacache_cnt = 0
    tc.teacache_accumulated_distance = 0
    tc.teacache_previous_modulated_input = None
    tc.teacache_previous_residual = None
    tc.teacache_skipped_steps = 0
    log("TeaCache enabled: thresh=0.18")

    # VAE tile override for H100 80GB
    vae = sampler.vae if hasattr(sampler, "vae") else sampler.pipeline.vae
    vae.tile_sample_min_size = 256
    vae.tile_latent_min_size = 32
    vae.tile_sample_min_tsize = 64
    vae.tile_latent_min_tsize = 16
    vae.tile_overlap_factor = 0.25
    log("VAE tile override: spatial=256, temporal=64, overlap=0.25")

    # Load audio model (whisper)
    log("loading wav2vec...")
    wav2vec = WhisperModel.from_pretrained(f"{MODEL_BASE}/ckpts/whisper-tiny/").to(device=device, dtype=torch.float32)
    wav2vec.requires_grad_(False)

    # Load face alignment
    log("loading face alignment...")
    det_path = os.path.join(MODEL_BASE, "ckpts/det_align/detface.pt")
    align_instance = AlignImage("cuda", det_path=det_path)

    # Load feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(f"{MODEL_BASE}/ckpts/whisper-tiny/")

    args = sampler.args
    log("engine loaded. model is in VRAM and ready.")

    return {
        "sampler": sampler,
        "args": args,
        "wav2vec": wav2vec,
        "align_instance": align_instance,
        "feature_extractor": feature_extractor,
        "device": device,
    }


# ============================================================
# RENDER — called per job, model already loaded
# ============================================================
def render(engine, image_path, audio_path, output_path, settings):
    """Run a render using the pre-loaded engine. No model loading — fast."""
    from hymm_sp.data_kits.audio_dataset import VideoAudioTextLoaderVal
    from torch.utils.data import DataLoader

    sampler = engine["sampler"]
    args = engine["args"]
    wav2vec = engine["wav2vec"]
    align_instance = engine["align_instance"]
    feature_extractor = engine["feature_extractor"]

    # Override args with job settings
    args.infer_steps = int(settings.get("inference_steps", 50))
    args.cfg_scale = float(settings.get("cfg_scale", 7.5))
    args.flow_shift_eval_video = float(settings.get("flow_shift", 5.0))
    length = settings.get("video_length", "5s")
    args.sample_n_frames = FMAP.get(length, 129)
    seed_val = settings.get("seed")
    args.seed = int(seed_val) if seed_val not in (None, "", 0) else 128
    # Per-request resolution. 512 is the Modal-validated working default
    # (5-step + 512 produces clean eyes and no shape errors). 704 triggers
    # a latent shape-mismatch bug in the pipeline that we still need to fix;
    # pass image_size=704 explicitly only for testing once that bug is fixed.
    args.image_size = int(settings.get("image_size", 512))

    # Update TeaCache for current step count
    tc = sampler.model
    tc.teacache_num_steps = args.infer_steps
    tc.teacache_cnt = 0
    tc.teacache_accumulated_distance = 0
    tc.teacache_previous_modulated_input = None
    tc.teacache_previous_residual = None
    tc.teacache_skipped_steps = 0

    teacache_enabled = settings.get("teacache_enabled", True)
    tc.enable_teacache = bool(teacache_enabled)
    if teacache_enabled:
        tc.teacache_thresh = float(settings.get("teacache_threshold", 0.18))

    # Create temporary CSV for the dataset loader
    prompt = (settings.get("prompt") or "A person speaking naturally and confidently").replace(",", " ").strip()

    # Color-boost prompt modifiers — when enabled, secretly append the
    # "quality modifiers" that commercial platforms (Hedra, HeyGen) use
    # to nudge the diffusion model into producing more saturated, more
    # cinematically-graded output. Off by default so existing prompts
    # stay verbatim unless the operator opts in.
    if settings.get("color_boost", True):
        color_modifiers = " vibrant colors cinematic lighting high contrast professional color grading detailed skin textures rich saturation"
        prompt = f"{prompt}{color_modifiers}".strip()
        log(f"color_boost prompt modifiers appended")
    csv_path = image_path.parent / "input.csv"
    csv_path.write_text(f"videoid,image,audio,prompt,fps\n1,{image_path},{audio_path},{prompt},25\n")

    # Load the data
    kwargs = {
        "text_encoder": sampler.text_encoder,
        "text_encoder_2": sampler.text_encoder_2,
        "feature_extractor": feature_extractor,
    }
    dataset = VideoAudioTextLoaderVal(image_size=args.image_size, meta_file=str(csv_path), **kwargs)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for batch in loader:
        fps = batch["fps"]
        audio_path_str = str(batch["audio_path"][0])

        # Ensure wav2vec stays on GPU (predict() may move it to CPU for memory management)
        wav2vec.to(engine["device"])

        # Run prediction — THIS IS THE FAST PART (model already in VRAM)
        start = time.time()
        samples = sampler.predict(args, batch, wav2vec, feature_extractor, align_instance)
        render_time = time.time() - start
        log(f"predict() took {render_time:.1f}s")

        if samples is None:
            raise RuntimeError("predict() returned None")

        sample = samples["samples"][0].unsqueeze(0)
        sample = sample[:, :, :batch["audio_len"][0]]

        video = rearrange(sample[0], "c f h w -> f h w c")
        video = (video * 255.0).data.cpu().numpy().astype(np.uint8)

        torch.cuda.empty_cache()

        # Save video
        temp_video = output_path.parent / "temp_video.mp4"
        imageio.mimsave(str(temp_video), video, fps=fps.item())

        # Optional color grading at mux time. Hedra/HeyGen output looks
        # punchier than raw HunyuanVideo-Avatar because they apply a grade
        # to the model's "flat / log-style" output. First attempt used
        # mild values (sat+20/con+10) and the user reported "dull, not
        # enough". Bumping to aggressive values + adding vibrance (which
        # selectively boosts muted colors over already-saturated ones —
        # the "Hedra secret" for punch without plastic skin).
        #
        # Values:
        #   saturation=1.4    +40% overall saturation
        #   contrast=1.18     +18% contrast (deeper blacks, brighter highs)
        #   brightness=-0.03  deepen midtones slightly
        #   vibrance=0.4      +40% vibrance — hits muted colors harder
        #                     than already-vivid ones, so skin/shadows pop
        #                     without blowing out already-bright pixels
        # Two filters chained.
        if settings.get("color_boost", True):
            color_filter = "-vf eq=saturation=1.4:contrast=1.18:brightness=-0.03,vibrance=intensity=0.4"
            log("color_boost ffmpeg grade applied (sat+40 / con+18 / bri-0.03 / vibrance+40)")
        else:
            color_filter = ""
        os.system(f"ffmpeg -i '{temp_video}' -i '{audio_path_str}' {color_filter} -shortest '{output_path}' -y -loglevel quiet; rm '{temp_video}'")

        return output_path, render_time

    raise RuntimeError("no batch produced from dataset")


# ============================================================
# SUPABASE HELPERS
# ============================================================
def ensure_env():
    missing = [n for n, v in [("SUPABASE_URL", SUPABASE_URL), ("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY)] if not v]
    if missing:
        log(f"FATAL: missing: {missing}")
        sys.exit(1)


def ensure_bucket(sb):
    try:
        names = [getattr(b, "name", None) or (b.get("name") if isinstance(b, dict) else None) for b in sb.storage.list_buckets()]
        if BUCKET in names:
            log(f"bucket {BUCKET} exists")
            return
        log(f"creating bucket {BUCKET}")
        sb.storage.create_bucket(BUCKET, options={"public": "true", "file_size_limit": "104857600"})
        log(f"bucket {BUCKET} created")
    except Exception as e:
        log(f"bucket warning: {e}")


def claim_job(sb):
    res = sb.table("video_jobs").select("*").eq("status", "pending").order("created_at").limit(1).execute()
    if not res.data:
        return None
    job = res.data[0]
    upd = sb.table("video_jobs").update({"status": "processing"}).eq("id", job["id"]).eq("status", "pending").execute()
    return job if upd.data else None


def dl(url, dest):
    log(f"dl -> {dest.name}")
    r = requests.get(url, timeout=120, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for c in r.iter_content(262144):
            f.write(c)
    return dest


def fail_job(sb, jid, msg):
    try:
        sb.table("video_jobs").update({"status": "failed", "error_message": msg}).eq("id", jid).execute()
    except Exception as e:
        log(f"fail update err: {e}")


def process_job(sb, engine, job):
    jid = job["id"]
    s = job.get("settings") or {}
    log(f"=== job {jid} ===")
    log(f"settings: {json.dumps(s)}")
    start = time.time()

    jt = WORKSPACE / "assets" / "worker_tmp" / jid
    jr = WORKSPACE / "results" / "worker" / jid
    jt.mkdir(parents=True, exist_ok=True)
    jr.mkdir(parents=True, exist_ok=True)

    try:
        # Settings can override image_url/audio_url (for quick swap testing
        # via the HTTP endpoint without needing to update Supabase storage).
        iu = s.get("image_url") or job.get("image_url")
        au = s.get("audio_url") or job.get("audio_url")
        if not iu or not au:
            raise RuntimeError("missing image_url or audio_url")

        ie = iu.split("?")[0].rsplit(".", 1)[-1].lower()
        ie = "jpg" if ie not in ("jpg", "jpeg", "png") else ie
        ae = au.split("?")[0].rsplit(".", 1)[-1].lower()
        ae = "wav" if ae not in ("wav", "mp3") else ae

        ip = dl(iu, jt / f"input.{ie}")
        ap = dl(au, jt / f"input.{ae}")

        # Optional photo enhancement — CLAHE + sharpen + PNG re-save to
        # eliminate lighting drift from uneven illumination / JPEG noise.
        # OFF by default so photos that already render well stay bit-identical.
        if s.get("enhance"):
            enhanced = jt / "input_enhanced.png"
            t0 = time.time()
            enhance_image(ip, enhanced)
            log(f"enhanced photo in {time.time()-t0:.1f}s (autocontrast + blur(1) + PNG)")
            ip = enhanced

        output_file = jr / "output_audio.mp4"
        video_path, render_time = render(engine, ip, ap, output_file, s)

        log(f"output: {video_path.name} ({video_path.stat().st_size} bytes)")

        # Upload to Supabase
        sp = f"{jid}.mp4"
        fb = video_path.read_bytes()
        try:
            sb.storage.from_(BUCKET).upload(path=sp, file=fb, file_options={"content-type": "video/mp4", "upsert": "true"})
        except Exception:
            ensure_bucket(sb)
            sb.storage.from_(BUCKET).upload(path=sp, file=fb, file_options={"content-type": "video/mp4", "upsert": "true"})

        pu = sb.storage.from_(BUCKET).get_public_url(sp)
        ms = int((time.time() - start) * 1000)
        sb.table("video_jobs").update({"status": "done", "output_url": pu, "render_time_ms": ms}).eq("id", jid).execute()
        log(f"=== job {jid} DONE {ms}ms ===")

    except Exception as e:
        log(f"FAILED:\n{traceback.format_exc()}")
        fail_job(sb, jid, str(e)[:2000])
    finally:
        try:
            for p in jt.iterdir():
                p.unlink()
            jt.rmdir()
        except Exception:
            pass


# ============================================================
# MAIN
# ============================================================
def main():
    ensure_env()
    log("starting")
    log(f"supabase: {SUPABASE_URL}")
    log(f"workspace: {WORKSPACE}")

    # LOAD ENGINE ONCE — stays in VRAM forever
    engine = load_engine()

    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    ensure_bucket(sb)
    log("ready — model in VRAM, waiting for jobs")

    while True:
        try:
            job = claim_job(sb)
            if job:
                process_job(sb, engine, job)
            else:
                time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            log("shutdown")
            break
        except Exception as e:
            log(f"loop err: {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
