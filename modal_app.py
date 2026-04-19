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
image = modal.Image.from_registry(
    "ghcr.io/everlaunchsocial/avatar:b76e42724415626b4504993597b7f9af484874e8",
    secret=modal.Secret.from_name("github-ghcr2"),
).entrypoint([]).env({
    # NOTE: intentionally NOT setting WORKER_MODE here — the baked-in
    # docker-entrypoint.sh would otherwise auto-run the worker, which
    # breaks CPU-only jobs like download_weights. The GPU renderer class
    # calls scripts/worker.py directly in its @modal.enter(), so it doesn't
    # need the entrypoint script.
    "MODEL_BASE": MODEL_DIR,
    "HF_HOME": MODEL_DIR + "/hf",
    "PYTHONPATH": "/workspace/HunyuanVideo-Avatar",
    "TOKENIZERS_PARALLELISM": "false",
    "MASTER_ADDR": "localhost",
    "MASTER_PORT": "29500",
})

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
        import os, sys, time, subprocess
        from pathlib import Path as _Path
        os.environ["MODEL_BASE"] = MODEL_DIR

        # Clone the code repo (entrypoint is disabled, so we do it here)
        repo_dir = "/workspace/HunyuanVideo-Avatar"
        if not _Path(repo_dir + "/.git").exists():
            print(f"Cloning code into {repo_dir}...")
            subprocess.run(
                ["git", "clone", "https://github.com/everlaunchsocial/avatar.git", repo_dir],
                check=True,
            )
        # scripts/ has no __init__.py, so import scripts/worker.py directly by path
        sys.path.insert(0, repo_dir)
        sys.path.insert(0, repo_dir + "/scripts")
        # Also set PYTHONPATH for any subprocess calls
        os.environ["PYTHONPATH"] = repo_dir + ":" + os.environ.get("PYTHONPATH", "")

        # Change working directory — some imports in the repo use relative paths
        os.chdir(repo_dir)

        # Symlink weights into expected path
        expected = _Path(repo_dir + "/weights")
        if not expected.exists():
            expected.symlink_to(MODEL_DIR)
            print(f"Linked weights: {expected} -> {MODEL_DIR}")

        # Debug: verify hymm_sp is actually importable before we proceed
        hymm_dir = _Path(repo_dir) / "hymm_sp"
        print(f"sys.path[0:3]: {sys.path[:3]}")
        print(f"hymm_sp exists: {hymm_dir.exists()}, __init__.py: {(hymm_dir / '__init__.py').exists()}")

        # Initialize torch distributed (required by the model, needs GPU)
        import torch
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", world_size=1, rank=0)

        # Import the real loader from our scripts/worker.py
        from worker import load_engine
        start = time.time()
        self.engine = load_engine()
        print(f"✅ Engine loaded in {time.time() - start:.1f}s, model in VRAM")

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