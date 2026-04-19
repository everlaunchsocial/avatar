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
        "echo 'cache_bust_manual_eviction'",
        "rm -rf /workspace/HunyuanVideo-Avatar",
        "git clone https://github.com/everlaunchsocial/avatar.git /workspace/HunyuanVideo-Avatar",
    )
    .env({
        "MODEL_BASE": MODEL_DIR,
        "HF_HOME": MODEL_DIR + "/hf",
        "PYTHONPATH": "/workspace/HunyuanVideo-Avatar",
        "TOKENIZERS_PARALLELISM": "false",
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "29500",
        # CRITICAL: Hunyuan's text_encoder/vae/models read these from ENV VARS
        # at import time. Both are needed to trigger the "very low VRAM"
        # single-GPU code path that offloads LLaVA (~16GB) to CPU.
        "CPU_OFFLOAD": "1",
        "DISABLE_SP": "1",
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
# MAIN RENDERER — one class, one container, scale-to-zero
# ─────────────────────────────────────────────────────────────
@app.cls(
    image=image,
    gpu="H100",
    volumes={MODEL_DIR: model_volume},
    cpu=12,
    memory=65536,
    scaledown_window=180,    # stay warm 3 min after last job
    timeout=1800,            # 30 min max per render
    max_containers=10,       # cap parallel renders (raise later if needed)
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
        # Removed PYTORCH_CUDA_ALLOC_CONF entirely — expandable_segments was
        # triggering a PyTorch internal assert. Stick with defaults and rely on
        # manual eviction + VAE tiling for memory savings.

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

        # ─── MANUAL VRAM RECOVERY ─────────────────────────────────
        # Even with CPU_OFFLOAD=1, modules are loaded on GPU first then
        # "should" be moved. Force them to CPU now.
        import gc
        sampler = self.engine["sampler"]

        # Evict text encoders (LLaVA ~16GB, CLIP ~1GB) to CPU — they only
        # run at the start of each job, not during the main diffusion.
        for attr in ("text_encoder", "text_encoder_2"):
            enc = getattr(sampler, attr, None)
            if enc is not None and hasattr(enc, "to"):
                try:
                    enc.to("cpu")
                    print(f"Evicted {attr} to CPU")
                except Exception as e:
                    print(f"Could not evict {attr}: {e}")

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
    def render_job(self, job_id: str):
        """Called by the Supabase Edge Function (or any HTTP trigger) with a job_id."""
        # Fetch the job row from Supabase
        res = self.sb.table("video_jobs").select("*").eq("id", job_id).execute()
        if not res.data:
            return {"status": "error", "message": f"Job {job_id} not found"}
        job = res.data[0]

        # Mark as processing (Supabase schema doesn't have this state by default,
        # but we add it here so the row moves out of "pending")
        self.sb.table("video_jobs").update({"status": "processing"}).eq("id", job_id).execute()

        # Reuse the exact process_job logic from scripts/worker.py
        from worker import process_job
        process_job(self.sb, self.engine, job)

        # Return updated row for caller convenience
        final = self.sb.table("video_jobs").select("status,output_url,render_time_ms,error_message").eq("id", job_id).execute()
        return final.data[0] if final.data else {"status": "unknown"}


# ─────────────────────────────────────────────────────────────
# HTTP ENDPOINT — Supabase Edge Function POSTs to this URL
# After deploy, URL is: https://<your-workspace>--everlaunch-avatar-render-endpoint.modal.run
# ─────────────────────────────────────────────────────────────
@app.function(
    image=image,
    secrets=[supabase_secret],
    timeout=1800,
)
@modal.fastapi_endpoint(method="POST")
def render_endpoint(job_id: str):
    """POST /?job_id=<uuid> — triggers a render. Returns the result."""
    renderer = AvatarRenderer()
    result = renderer.render_job.remote(job_id)
    return result