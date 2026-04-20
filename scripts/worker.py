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

    # FA3 Hopper diagnostic — critical. The pre-built image ships
    # flash_attn_interface + flashattn_hopper_cuda.so from the hopper/
    # subdir at tag v2.7.4.post1. If that import fails, the code in
    # models_audio.py line 11-13 silently falls back to FA2 which is
    # 5-7x slower on H100. This print tells us at boot which path is
    # actually live. Per forensic audit, this silent fallback was THE
    # #1 cause of speed regression.
    try:
        import flash_attn_interface as _fa3
        log(f"[FA3-DIAG] ✅ FA3 Hopper ACTIVE — loaded from {_fa3.__file__}")
        try:
            from flash_attn_interface import flash_attn_varlen_func as _fa3_func
            log(f"[FA3-DIAG] ✅ flash_attn_varlen_func resolved from: {_fa3_func.__module__}")
        except Exception as _e:
            log(f"[FA3-DIAG] ⚠️ flash_attn_varlen_func import failed: {_e}")
    except ImportError as _e:
        log(f"[FA3-DIAG] ❌ FA3 NOT ACTIVE — falling back to FA2/SDPA. This is the #1 speed regression. Error: {_e}")
        try:
            from flash_attn.flash_attn_interface import flash_attn_varlen_func as _fa2_func
            log(f"[FA3-DIAG] ℹ️  FA2 fallback available from: {_fa2_func.__module__}")
        except Exception as _e2:
            log(f"[FA3-DIAG] ❌❌ FA2 also missing — attention will use torch SDPA (slowest). Error: {_e2}")

    # Parse default args with our settings.
    # NOTE: these are the ENGINE-LOAD defaults. Per-render user settings
    # (cfg_scale, flow_shift, steps, etc.) override these later in render().
    # Values below reflect the RunPod "Golden Baseline" config restored
    # 2026-04-20 per forensic audit — NOT the prior Modal-dull config.
    sys.argv = [
        "worker",
        "--input", "dummy.csv",
        "--ckpt", f"{MODEL_BASE}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt",
        "--sample-n-frames", "129",
        "--seed", "128",
        "--image-size", "512",
        "--cfg-scale", "7.5",
        "--infer-steps", "50",
        # DeepCache ON — RunPod golden config had this enabled. Provides
        # additional per-step caching of attention outputs. Previously 0.
        "--use-deepcache", "1",
        # --cpu-offload removed: runs ~7x slower than non-offload path.
        # H100 has 80GB, full model is ~33GB — offload is unnecessary.
        "--flow-shift-eval-video", "5.0",
        "--save-path", str(WORKSPACE / "results"),
        "--use-fp8",
        # VAE bf16 cast (RunPod golden config). Default is fp16 which
        # compresses highlight/shadow range and was responsible for the
        # "dull/darker eyes and background" symptom on Modal.
        "--vae-precision", "bf16",
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

    # Override args with job settings. Defaults restored to RunPod
    # Golden Baseline ("4-4-3-0.18" per forensic audit, chat-3:8598):
    # Steps 4 / CFG 4 / Flow 3 / TeaCache 0.18 + Fix 6. This produced
    # ~93s renders with sharp eyes at low step count. User can override
    # via Lovable settings JSON per render.
    args.infer_steps = int(settings.get("inference_steps", 5))
    args.cfg_scale = float(settings.get("cfg_scale", 4.0))
    args.flow_shift_eval_video = float(settings.get("flow_shift", 3.0))
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

    # NOTE: previously appended "vibrant colors / cinematic lighting /
    # high contrast / detailed skin textures" quality modifiers to the
    # prompt here. In testing this caused the model to OVER-amplify
    # facial features — specifically lips/mouth "jumping out" even at
    # low CFG (3.0). The model interpreted "detailed skin textures" +
    # "high contrast" as a directive to exaggerate facial motion.
    # Removed. color_boost now ONLY does the ffmpeg post-render grade,
    # which is applied AFTER the model and cannot influence how faces
    # are animated.
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
            # Post-render quality pass — three stages at mux time:
            #   1. 2x upscale via Lanczos (cleaner edges, no AI required)
            #   2. Unsharp mask (restores perceived detail lost to VAE decode)
            #   3. Color grade (saturation + contrast, no vibrance — was orange)
            # Roughly 30% of the way to Hedra-level perceived quality
            # without any new dependencies. Path B (Real-ESRGAN + GFPGAN)
            # goes significantly further but requires container rebuild.
            color_filter = (
                "-vf scale=iw*2:ih*2:flags=lanczos,"
                "unsharp=5:5:0.9:5:5:0.4,"
                "eq=saturation=1.25:contrast=1.12:brightness=-0.02"
            )
            log("color_boost applied: 2x upscale + unsharp + sat+25/con+12/bri-0.02")
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
        # Accept a wide range of audio/video-container formats. We always
        # normalize to 16 kHz mono wav below (Whisper's native format), so
        # the exact input extension only matters for the download filename.
        ae = au.split("?")[0].rsplit(".", 1)[-1].lower()
        if ae not in ("wav", "mp3", "webm", "ogg", "m4a", "aac", "flac", "opus", "mp4", "mov", "mkv"):
            ae = "bin"  # unknown — ffmpeg will sniff the container on decode

        ip = dl(iu, jt / f"input.{ie}")
        ap_raw = dl(au, jt / f"input_raw.{ae}")

        # Universal audio normalization to 16 kHz mono wav. Whisper-tiny
        # (used internally for audio embedding) requires this format, and
        # normalizing up-front protects against every weird audio format
        # a customer might upload: webm, m4a, opus, flac, even video
        # containers like mp4/mov where ffmpeg extracts the audio track.
        # If the source is already 16 kHz mono wav, this is a near-instant
        # re-encode. If not, it converts. Either way the model gets a
        # predictable input.
        ap = jt / "input.wav"
        t0 = time.time()
        ret = os.system(f"ffmpeg -i '{ap_raw}' -ar 16000 -ac 1 -y '{ap}' -loglevel quiet")
        if ret != 0 or not ap.exists() or ap.stat().st_size == 0:
            log(f"audio normalization failed (ret={ret}), falling back to raw")
            ap = ap_raw
        else:
            log(f"audio normalized to 16kHz mono wav in {time.time()-t0:.2f}s (from .{ae})")

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
