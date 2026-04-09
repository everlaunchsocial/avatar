#!/usr/bin/env python3
# EverLaunch Avatar Studio worker. Polls video_jobs, runs renders, uploads results.
import os, sys, time, json, subprocess, traceback
from pathlib import Path
from datetime import datetime, timezone
import requests
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "5"))
RENDER_TIMEOUT = int(os.environ.get("RENDER_TIMEOUT", "1800"))
WORKSPACE = Path("/workspace/HunyuanVideo-Avatar")
ASSETS_TMP = WORKSPACE / "assets" / "worker_tmp"
RESULTS_DIR = WORKSPACE / "results" / "worker"
CKPT_REL = "./weights/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
CKPT_PATH = WORKSPACE / "weights/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
BUCKET = "rendered-videos"
FMAP = {"5s": 129, "10s": 257, "15s": 385, "30s": 769}


def log(m):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{ts}] [worker] {m}", flush=True)


def ensure_env():
    miss = [n for n, v in [("SUPABASE_URL", SUPABASE_URL), ("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY)] if not v]
    if miss:
        log(f"FATAL: missing: {miss}")
        sys.exit(1)
    if not CKPT_PATH.exists():
        log(f"FATAL: no ckpt: {CKPT_PATH}")
        sys.exit(1)
    ASSETS_TMP.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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


def build_cmd(img, aud, out, csv, s):
    steps = int(s.get("inference_steps", 50))
    cfg = float(s.get("cfg_scale", 7.5))
    fs = float(s.get("flow_shift", 5.0))
    sv = s.get("seed")
    seed = int(sv) if sv not in (None, "", 0) else 128
    nf = FMAP.get(s.get("video_length", "5s"), 129)
    p = (s.get("prompt") or "A person speaking naturally and confidently").replace(",", " ").strip()
    ri = img.relative_to(WORKSPACE)
    ra = aud.relative_to(WORKSPACE)
    csv.write_text(f"videoid,image,audio,prompt,fps\n1,{ri},{ra},{p},25\n")
    cmd = [
        "python3", "hymm_sp/sample_gpu_poor.py",
        "--input", str(csv.relative_to(WORKSPACE)),
        "--ckpt", CKPT_REL,
        "--sample-n-frames", str(nf),
        "--seed", str(seed),
        "--image-size", "704",
        "--cfg-scale", str(cfg),
        "--infer-steps", str(steps),
        "--use-deepcache", "1",
        "--flow-shift-eval-video", str(fs),
        "--save-path", str(out.relative_to(WORKSPACE)),
        "--use-fp8",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE)
    env["MODEL_BASE"] = str(WORKSPACE / "weights")
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return cmd, env


def fail_job(sb, jid, msg):
    try:
        sb.table("video_jobs").update({
            "status": "failed",
            "error_message": msg,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", jid).execute()
    except Exception as e:
        log(f"fail update err: {e}")


def process_job(sb, job):
    jid = job["id"]
    s = job.get("settings") or {}
    log(f"=== job {jid} ===")
    log(f"settings: {json.dumps(s)}")
    start = time.time()
    jt = ASSETS_TMP / jid
    jr = RESULTS_DIR / jid
    jt.mkdir(parents=True, exist_ok=True)
    jr.mkdir(parents=True, exist_ok=True)
    try:
        iu = job.get("image_url")
        au = job.get("audio_url")
        if not iu or not au:
            raise RuntimeError("missing image_url or audio_url")
        ie = iu.split("?")[0].rsplit(".", 1)[-1].lower()
        ie = "jpg" if ie not in ("jpg", "jpeg", "png") else ie
        ae = au.split("?")[0].rsplit(".", 1)[-1].lower()
        ae = "wav" if ae not in ("wav", "mp3") else ae
        ip = dl(iu, jt / f"input.{ie}")
        ap = dl(au, jt / f"input.{ae}")
        cp = jt / "input.csv"
        cmd, env = build_cmd(ip, ap, jr, cp, s)
        log(f"cmd: {' '.join(cmd)}")
        r = subprocess.run(cmd, cwd=str(WORKSPACE), env=env, capture_output=True, text=True, timeout=RENDER_TIMEOUT)
        if r.returncode != 0:
            raise RuntimeError(f"render exit {r.returncode}: {(r.stderr or r.stdout or '')[-2000:]}")
        mp4s = sorted(jr.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not mp4s:
            raise RuntimeError("no .mp4 output")
        ov = mp4s[0]
        log(f"output: {ov.name} ({ov.stat().st_size} bytes)")
        sp = f"{jid}.mp4"
        fb = ov.read_bytes()
        try:
            sb.storage.from_(BUCKET).upload(path=sp, file=fb, file_options={"content-type": "video/mp4", "upsert": "true"})
        except Exception:
            ensure_bucket(sb)
            sb.storage.from_(BUCKET).upload(path=sp, file=fb, file_options={"content-type": "video/mp4", "upsert": "true"})
        pu = sb.storage.from_(BUCKET).get_public_url(sp)
        ms = int((time.time() - start) * 1000)
        sb.table("video_jobs").update({
            "status": "done",
            "output_url": pu,
            "render_time_ms": ms,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", jid).execute()
        log(f"=== job {jid} DONE {ms}ms ===")
    except subprocess.TimeoutExpired:
        fail_job(sb, jid, f"timeout {RENDER_TIMEOUT}s")
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


def main():
    ensure_env()
    log("starting")
    log(f"supabase: {SUPABASE_URL}")
    log(f"workspace: {WORKSPACE}")
    sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    ensure_bucket(sb)
    log("ready")
    while True:
        try:
            job = claim_job(sb)
            if job:
                process_job(sb, job)
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
