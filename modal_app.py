"""EverLaunch Avatar — Modal deployment.
Event-driven: Supabase Edge Function → Modal HTTP call → renders one video → returns.
Scale-to-zero: container only runs when a job is active.
"""
import modal

app = modal.App("everlaunch-avatar")

# ─────────────────────────────────────────────────────────────
# PERSISTENT VOLUME (holds 50GB of model weights)
# ─────────────────────────────────────────────────────────────
# Use explicit POSIX strings (not pathlib) — pathlib produces Windows
# backslashes when this file runs on Windows and Modal's image builder
# rejects them as invalid escape sequences.
MODEL_DIR = "/models"
model_volume = modal.Volume.from_name("everlaunch-avatar-weights", create_if_missing=True)

# ─────────────────────────────────────────────────────────────
# CONTAINER IMAGE (pulls our pre-built image with FA3 Hopper)
# ─────────────────────────────────────────────────────────────
image = (
    modal.Image.from_registry(
        "ghcr.io/everlaunchsocial/avatar:b76e42724415626b4504993597b7f9af484874e8",
        secret=modal.Secret.from_name("github-ghcr2"),
    )
    .entrypoint([])
    # Clone the code INTO the image at build time — not at runtime.
    # The echo with a commit SHA busts Modal's image cache whenever we push new code.
    # Update this SHA when you push a worker.py change and want it picked up.
    .run_commands(
        "echo 'cache_bust_basicsr_torchvision_shim'",
        "rm -rf /workspace/HunyuanVideo-Avatar",
        "git clone https://github.com/everlaunchsocial/avatar.git /workspace/HunyuanVideo-Avatar",
    )
    # Phase B: pinned AI upscale/restore deps for the optional
    # quality_pass flag (GFPGAN for face restoration + Real-ESRGAN for
    # general upscale). Versions pinned per forensic audit Report 3 #8:
    # these specific versions are known-compatible with torch 2.5.1 and
    # the basicsr shared dep. Do NOT bump without re-verifying the pin.
    .pip_install([
        "gfpgan==1.3.8",
        "realesrgan==0.3.0",
        "basicsr==1.4.2",
        "facexlib==0.3.0",
    ])
    .env({
        "MODEL_BASE": MODEL_DIR,
        "HF_HOME": MODEL_DIR + "/hf",
        # TORCH_HOME on persistent volume so GFPGAN/facexlib auxiliary
        # weights (~100MB detector + parser nets) don't re-download on
        # every cold start. Default would be ~/.cache/torch/hub/
        # (ephemeral). Pre-populated by download_quality_weights.
        "TORCH_HOME": MODEL_DIR + "/cache/torch_hub",
        "PYTHONPATH": "/workspace/HunyuanVideo-Avatar",
        "TOKENIZERS_PARALLELISM": "false",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29500",
        # CPU_OFFLOAD removed 2026-04-20: RunPod ran this model WITHOUT
        # offload at ~7 sec/step on H100. With offload on, we were hitting
        # ~50 sec/step because of constant .to(cpu)/.to(cuda) shuffling.
        # H100 has 80 GB; full model is ~33 GB; offload is unnecessary and
        # was the primary cause of our ~7x slowdown vs RunPod.
        "DISABLE_SP": "1",
        # PyTorch allocator safety setting per forensic audit (Report 3 #1).
        # `max_split_size_mb:256` keeps fragmentation from breaking VAE
        # decode on long (30s+) renders. CRITICAL: do NOT add
        # `expandable_segments:True` — that setting crashes Modal's
        # CUDA allocator with INTERNAL ASSERT FAILED (killed the driver
        # in prior Verda / Modal attempts, see commit history from yesterday).
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256",
    })
    .workdir("/workspace/HunyuanVideo-Avatar")
)

# ─────────────────────────────────────────────────────────────
# SECRETS (create in Modal dashboard as "supabase-keys")
# Keys needed: SUPABASE_URL, SUPABASE_SERVICE_KEY
# ─────────────────────────────────────────────────────────────
supabase_secret = modal.Secret.from_name("supabase-keys")


# ─────────────────────────────────────────────────────────────
# ONE-TIME: Download weights directly into the Modal Volume
# Run this ONCE before deploying: `modal run modal_app.py::download_weights`
# ─────────────────────────────────────────────────────────────
@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=3600,
    cpu=4,
    memory=32768,
)
def download_weights():
    import subprocess
    print(f"Downloading HunyuanVideo-Avatar into {MODEL_DIR} ...")
    subprocess.run(
        ["huggingface-cli", "download", "tencent/HunyuanVideo-Avatar", "--local-dir", MODEL_DIR],
        check=True,
    )
    model_volume.commit()
    print("✅ Weights committed to Volume")


# ─────────────────────────────────────────────────────────────
# ONE-TIME: Download GFPGAN + Real-ESRGAN weights for quality_pass.
# Run: `modal run modal_app.py::download_quality_weights`
# Weights live on the same Modal Volume as the main model,
# persistent across container restarts.
# ─────────────────────────────────────────────────────────────
@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=600,
    cpu=2,
    memory=8192,
)
def download_quality_weights():
    import os
    import urllib.request
    import ssl
    dest_dir = f"{MODEL_DIR}/cache/gfpgan"
    os.makedirs(dest_dir, exist_ok=True)

    # GFPGAN / Real-ESRGAN go in /models/cache/gfpgan
    # facexlib helpers (used internally by GFPGAN for face detection)
    # must live at $TORCH_HOME/checkpoints because that's where
    # facexlib.utils.load_file_from_url looks (via torch.hub).
    torch_hub_checkpoints = f"{MODEL_DIR}/cache/torch_hub/checkpoints"
    os.makedirs(torch_hub_checkpoints, exist_ok=True)

    targets = [
        # (filename, url, destination_dir)
        ("GFPGANv1.4.pth",
         "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
         dest_dir),
        ("RealESRGAN_x2plus.pth",
         "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
         dest_dir),
        ("detection_Resnet50_Final.pth",
         "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
         torch_hub_checkpoints),
        ("parsing_parsenet.pth",
         "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
         torch_hub_checkpoints),
    ]

    ctx = ssl.create_default_context()
    for fname, url, target_dir in targets:
        target = os.path.join(target_dir, fname)
        if os.path.exists(target) and os.path.getsize(target) > 1024 * 1024:
            size_mb = os.path.getsize(target) / (1024 * 1024)
            print(f"✅ {fname} already present ({size_mb:.1f} MB) at {target_dir}")
            continue
        print(f"⬇️  Downloading {fname} → {target_dir}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx) as r, open(target, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        size_mb = os.path.getsize(target) / (1024 * 1024)
        print(f"✅ {fname} saved ({size_mb:.1f} MB)")

    model_volume.commit()
    print(f"✅ All quality_pass weights committed to volume")


# ─────────────────────────────────────────────────────────────
# MAIN RENDERER — one class, one container, scale-to-zero
# ─────────────────────────────────────────────────────────────
@app.cls(
    image=image,
    gpu="H100",
    volumes={MODEL_DIR: model_volume},
    cpu=12,
    memory=65536,
    scaledown_window=600,    # TESTING MODE: stay warm 10 min (was 180s). Revert to 180 before launch.
    timeout=3600,            # 60 min max per render (step 50 + TeaCache off can push past 30 min)
    max_containers=1,        # TESTING MODE: serialize so render #2 queues on same warm container (was 10). Revert before launch.
    secrets=[supabase_secret],
)
class AvatarRenderer:

    @modal.enter()
    def load_engine(self):
        """Runs ONCE when the container boots. Loads the full engine into VRAM."""
        import os, sys, time
        from pathlib import Path as _Path

        # CRITICAL: set these BEFORE any hymm_sp import.
        # hymm_sp's text_encoder/vae/models modules read CPU_OFFLOAD and DISABLE_SP at import time.
        os.environ["CPU_OFFLOAD"] = "1"
        os.environ["DISABLE_SP"] = "1"
        # Explicitly unset expandable_segments which may be inherited from a
        # cached image layer. Combining it with max_split_size_mb or certain
        # allocation patterns triggers a PyTorch internal assert.
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)

        repo_dir = "/workspace/HunyuanVideo-Avatar"
        os.environ["MODEL_BASE"] = MODEL_DIR
        sys.path.insert(0, repo_dir)
        sys.path.insert(0, repo_dir + "/scripts")

        # Symlink weights volume into the repo's expected path
        expected = _Path(repo_dir + "/weights")
        if not expected.exists():
            expected.symlink_to(MODEL_DIR)
            print(f"Linked weights: {expected} -> {MODEL_DIR}")

        # Initialize torch distributed (required by the model, needs GPU)
        import torch
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", world_size=1, rank=0)

        # Import the real loader from our scripts/worker.py
        try:
            from worker import load_engine
        except Exception as e:
            print(f"DEBUG sys.path: {sys.path}")
            print(f"DEBUG cwd: {os.getcwd()}")
            print(f"DEBUG /workspace listing: {os.listdir('/workspace') if os.path.exists('/workspace') else 'MISSING'}")
            print(f"DEBUG hymm_sp exists: {os.path.isdir(repo_dir + '/hymm_sp')}")
            print(f"DEBUG hymm_sp/__init__.py exists: {os.path.isfile(repo_dir + '/hymm_sp/__init__.py')}")
            raise
        start = time.time()
        try:
            self.engine = load_engine()
        except Exception as e:
            print(f"DEBUG load_engine() failed — sys.path: {sys.path}")
            print(f"DEBUG cwd: {os.getcwd()}")
            print(f"DEBUG hymm_sp exists: {os.path.isdir(repo_dir + '/hymm_sp')}")
            raise
        print(f"✅ Engine loaded in {time.time() - start:.1f}s")

        # ─── DEVICE PLACEMENT FIX ─────────────────────────────────
        # CPU_OFFLOAD=1 causes hymm_sp.modules.models_audio to initialize the
        # 13B transformer on CPU. The pipeline has runtime .to('cuda') logic
        # for text_encoder and vae, but NOT for the transformer — so when
        # pipeline.transformer(...) is called, its weights are on CPU while
        # input tensors (motion_exp, latents, etc.) are on CUDA. This throws:
        #   "argument mat1 in method wrapper_CUDA_addmm"
        # Fix: force the transformer onto CUDA here. We have 79 GiB free and
        # the transformer is ~13 GB FP8, so this is comfortable.
        import gc
        sampler = self.engine["sampler"]

        transformer = getattr(sampler, "model", None)
        if transformer is not None and hasattr(transformer, "to"):
            try:
                transformer.to("cuda")
                print("Moved transformer (sampler.model) to CUDA")
            except Exception as e:
                print(f"Could not move transformer to CUDA: {e}")

        # Leave text_encoder / text_encoder_2 where load_engine() left them —
        # the pipeline already has cpu_offload logic that moves them to CUDA
        # for prompt encoding and back to CPU afterwards (see logs:
        # "encode prompt: move text_encoder to cuda / ... to cpu").

        # Enable VAE tiling+slicing so decode doesn't OOM at the end.
        # VAE tile sizes are already overridden in worker.load_engine().
        vae = getattr(sampler, "vae", None) or getattr(getattr(sampler, "pipeline", None), "vae", None)
        if vae is not None:
            if hasattr(vae, "enable_tiling"):
                vae.enable_tiling()
                print("VAE tiling enabled")
            if hasattr(vae, "enable_slicing"):
                vae.enable_slicing()
                print("VAE slicing enabled")

        # Flush allocator cache so freed VRAM becomes available
        import torch as _torch
        gc.collect()
        _torch.cuda.empty_cache()
        _torch.cuda.synchronize()

        free, total = _torch.cuda.mem_get_info()
        print(f"VRAM after eviction: {(total-free)/1024**3:.1f} GiB used / {total/1024**3:.1f} GiB total")

        # Supabase client for job pickup
        from supabase import create_client
        self.sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

        # Ensure output bucket exists
        from worker import ensure_bucket
        ensure_bucket(self.sb)

    @modal.method()
    def render_job(self, job_id: str, settings_override: dict = None):
        """Called by the Supabase Edge Function (or any HTTP trigger) with a job_id.

        settings_override (dict, optional): merged into the job's `settings`
        dict before rendering. Lets the HTTP caller tune `inference_steps`,
        `cfg_scale`, `video_length`, `teacache_threshold`, etc. per request
        without having to update the Supabase row first. Useful for
        iteration during stabilisation and quality tuning.
        """
        # Fetch the job row from Supabase
        res = self.sb.table("video_jobs").select("*").eq("id", job_id).execute()
        if not res.data:
            return {"status": "error", "message": f"Job {job_id} not found"}
        job = res.data[0]

        # Apply per-request setting overrides (if any).
        if settings_override:
            merged = dict(job.get("settings") or {})
            merged.update(settings_override)
            job["settings"] = merged
            print(f"[render_job] settings override applied: {settings_override}")
            print(f"[render_job] effective settings: {merged}")

        # Mark as processing (Supabase schema doesn't have this state by default,
        # but we add it here so the row moves out of "pending").
        # Also clear stale error_message / output_url from any previous run.
        # Bump updated_at so the poller's stale-sweep doesn't kill this
        # mid-render (renders can take 5-10+ min; sweep threshold is 30 min).
        from datetime import datetime, timezone
        self.sb.table("video_jobs").update({
            "status": "processing",
            "error_message": None,
            "output_url": None,
            "render_time_ms": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", job_id).execute()

        # Reuse the exact process_job logic from scripts/worker.py
        from worker import process_job
        process_job(self.sb, self.engine, job)

        # Return updated row for caller convenience
        final = self.sb.table("video_jobs").select("status,output_url,render_time_ms,error_message").eq("id", job_id).execute()
        return final.data[0] if final.data else {"status": "unknown"}


# ─────────────────────────────────────────────────────────────
# BACKGROUND POLLER — scans Supabase `video_jobs` every 30 seconds
# for any pending rows and fires a render for each.
#
# This bridges Lovable's pull-based architecture (insert a row,
# wait for a worker to pick it up) with Modal's compute. You do
# NOT need to call the HTTP endpoint below from Lovable — just
# drop rows into `video_jobs` with status='pending' and this
# cron will spawn a render container within ~30 seconds.
#
# Also cleans up stuck rows: anything in 'processing' that hasn't
# updated for 30+ minutes is marked failed so it stops blocking
# the UI from creating a new render for the same affiliate.
# ─────────────────────────────────────────────────────────────
@app.function(
    image=image,
    secrets=[supabase_secret],
    schedule=modal.Period(seconds=30),
    timeout=60,
    max_containers=1,  # serialize polls so we don't double-fire jobs
)
def poll_pending_jobs():
    """Runs every 30 seconds. Claims pending rows and spawns renders."""
    import os
    from datetime import datetime, timezone, timedelta
    from supabase import create_client

    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

    # --- 1. Sweep stale 'processing' rows (older than 30 min) to failed.
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    try:
        stale = sb.table("video_jobs").select("id,updated_at,created_at").eq("status", "processing").lt("updated_at", cutoff).execute()
        for row in (stale.data or []):
            print(f"[poll] sweeping stale processing row {row['id']} (updated_at={row.get('updated_at')})")
            sb.table("video_jobs").update({
                "status": "failed",
                "error_message": "Timed out — worker did not complete within 30 minutes",
            }).eq("id", row["id"]).eq("status", "processing").execute()
    except Exception as e:
        # updated_at may not exist in some schemas — fall back silently.
        print(f"[poll] stale-sweep skipped: {e}")

    # --- 2. Find pending rows and spawn renders for each.
    res = sb.table("video_jobs").select("id,created_at").eq("status", "pending").order("created_at").limit(10).execute()
    pending = res.data or []
    if not pending:
        return {"pending": 0, "fired": 0}

    now_iso = datetime.now(timezone.utc).isoformat()
    renderer = AvatarRenderer()
    fired = 0
    for row in pending:
        jid = row["id"]
        # Atomic claim: flip to processing only if still pending, to prevent
        # a second poll (or another worker) from double-firing the same row.
        # CRITICAL: bump updated_at to NOW so the stale-sweep on the next
        # poll iteration doesn't think this row is timed-out (its original
        # updated_at is from when Lovable first inserted the row, which
        # is already older than the stale threshold for long-pending rows).
        claim = sb.table("video_jobs").update({
            "status": "processing",
            "updated_at": now_iso,
        }).eq("id", jid).eq("status", "pending").execute()
        if not claim.data:
            print(f"[poll] skipped {jid} (already claimed)")
            continue
        print(f"[poll] firing render for {jid}")
        renderer.render_job.spawn(jid)
        fired += 1

    return {"pending": len(pending), "fired": fired}


# ─────────────────────────────────────────────────────────────
# HTTP ENDPOINT — Supabase Edge Function POSTs to this URL
# After deploy, URL is: https://<your-workspace>--everlaunch-avatar-render-endpoint.modal.run
#
# Optional path. With the poller above running, you do not need
# to call this from Lovable — the poller picks up pending rows
# automatically. This endpoint remains for manual/CLI testing
# and for overriding settings per-request.
# ─────────────────────────────────────────────────────────────
@app.function(
    image=image,
    secrets=[supabase_secret],
    timeout=3600,
)
@modal.fastapi_endpoint(method="POST")
def render_endpoint(
    job_id: str,
    inference_steps: int = None,
    cfg_scale: float = None,
    video_length: str = None,
    flow_shift: float = None,
    teacache_enabled: bool = None,
    teacache_threshold: float = None,
    seed: int = None,
    prompt: str = None,
    image_size: int = None,
    enhance: bool = None,
    color_boost: bool = None,
    wav2vec_gain: float = None,
    motion_scale: float = None,
    quality_pass: bool = None,
    image_url: str = None,
    audio_url: str = None,
):
    """POST /?job_id=<uuid>[&inference_steps=30&cfg_scale=6.5&...]

    Triggers a render for a Supabase video_jobs row. Any query-string
    parameter that matches a known render setting is merged on top of
    whatever is stored in the row, so quality/step tuning can happen
    without editing Supabase between tests.
    """
    overrides = {}
    if inference_steps is not None:
        overrides["inference_steps"] = inference_steps
    if cfg_scale is not None:
        overrides["cfg_scale"] = cfg_scale
    if video_length is not None:
        overrides["video_length"] = video_length
    if flow_shift is not None:
        overrides["flow_shift"] = flow_shift
    if teacache_enabled is not None:
        overrides["teacache_enabled"] = teacache_enabled
    if teacache_threshold is not None:
        overrides["teacache_threshold"] = teacache_threshold
    if seed is not None:
        overrides["seed"] = seed
    if prompt is not None:
        overrides["prompt"] = prompt
    if image_size is not None:
        overrides["image_size"] = image_size
    if enhance is not None:
        overrides["enhance"] = enhance
    if color_boost is not None:
        overrides["color_boost"] = color_boost
    if wav2vec_gain is not None:
        overrides["wav2vec_gain"] = wav2vec_gain
    if motion_scale is not None:
        overrides["motion_scale"] = motion_scale
    if quality_pass is not None:
        overrides["quality_pass"] = quality_pass
    if image_url is not None:
        overrides["image_url"] = image_url
    if audio_url is not None:
        overrides["audio_url"] = audio_url

    renderer = AvatarRenderer()
    result = renderer.render_job.remote(job_id, overrides or None)
    return result


# ─────────────────────────────────────────────────────────────
# DIAGNOSTIC — GET /?job_id=<uuid> returns the full row so we can
# see exactly what settings a render was fired with. Read-only.
# ─────────────────────────────────────────────────────────────
@app.function(
    image=image,
    secrets=[supabase_secret],
    timeout=30,
)
@modal.fastapi_endpoint(method="GET")
def job_info(job_id: str):
    """GET /?job_id=<uuid> — returns the full video_jobs row. Read-only."""
    import os
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    res = sb.table("video_jobs").select("*").eq("id", job_id).execute()
    if not res.data:
        return {"error": f"job {job_id} not found"}
    return res.data[0]


# ─────────────────────────────────────────────────────────────
# DIAGNOSTIC — GET /recent returns the last N jobs with their
# full settings so we can audit what configs produced good/bad
# outputs without digging through rotating logs.
# ─────────────────────────────────────────────────────────────
@app.function(
    image=image,
    secrets=[supabase_secret],
    timeout=30,
)
@modal.fastapi_endpoint(method="GET")
def recent_jobs(limit: int = 10):
    """GET /?limit=10 — returns the last N video_jobs rows, newest first."""
    import os
    from supabase import create_client
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    res = sb.table("video_jobs").select(
        "id,status,settings,created_at,updated_at,render_time_ms,error_message,output_url"
    ).order("created_at", desc=True).limit(int(limit)).execute()
    return {"count": len(res.data or []), "jobs": res.data or []}