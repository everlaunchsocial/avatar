"""
LLaVA Embedding Cache -- Monkey-Patch
"""
import os
import hashlib
import time
import functools
import torch

LLAVA_CACHE_ENABLED = True
# Moved to persistent Modal Volume (was /workspace which is EPHEMERAL).
# Every deploy wiped the cache, causing 17-sec LLaVA re-encode on first
# render after code pushes. Now on the everlaunch-avatar-weights volume.
LLAVA_CACHE_DIR = "/models/cache/llava_embeddings"
LLAVA_CACHE_MAX_MB = 4096
LLAVA_CACHE_LOG = True

_stats = {"hits": 0, "misses": 0, "passthrough": 0}
_installed_pipelines = set()


def _log(msg):
    if LLAVA_CACHE_LOG:
        print(f"[llava_cache] {msg}", flush=True)


def _is_llava_call(kwargs):
    pv = kwargs.get("pixel_value_llava", None)
    return isinstance(pv, torch.Tensor)


def _hash_image(t):
    flat = t.detach().float().flatten().cpu()
    n = flat.numel()
    if n <= 1024:
        sample = flat
    else:
        idx = torch.linspace(0, n - 1, 1024).long()
        sample = flat[idx]
    h = hashlib.md5()
    h.update(str(tuple(t.shape)).encode())
    h.update(sample.numpy().tobytes())
    return h.hexdigest()


def _make_key(args, kwargs):
    prompt = kwargs.get("prompt", args[0] if args else "")
    if isinstance(prompt, list):
        prompt = "||".join(prompt)
    pv = kwargs.get("pixel_value_llava")
    cfg = kwargs.get("do_classifier_free_guidance", True)
    nimg = kwargs.get("num_images_per_prompt", 1)
    h = hashlib.md5()
    h.update(str(prompt).encode())
    h.update(_hash_image(pv).encode())
    h.update(str(bool(cfg)).encode())
    h.update(str(int(nimg)).encode())
    return h.hexdigest()


def _cache_path(key):
    return os.path.join(LLAVA_CACHE_DIR, f"{key}.pt")


def _evict_lru():
    if LLAVA_CACHE_MAX_MB <= 0:
        return
    try:
        files = []
        total = 0
        for f in os.listdir(LLAVA_CACHE_DIR):
            if not f.endswith(".pt"):
                continue
            p = os.path.join(LLAVA_CACHE_DIR, f)
            st = os.stat(p)
            files.append((st.st_atime, st.st_size, p))
            total += st.st_size
        max_bytes = LLAVA_CACHE_MAX_MB * 1024 * 1024
        if total <= max_bytes:
            return
        files.sort()
        for _, size, p in files:
            if total <= max_bytes:
                break
            try:
                os.remove(p)
                total -= size
                _log(f"evicted {os.path.basename(p)}")
            except Exception:
                pass
    except Exception as e:
        _log(f"evict error: {e}")


def _save(key, result):
    try:
        os.makedirs(LLAVA_CACHE_DIR, exist_ok=True)
        cpu_result = tuple(
            (x.detach().cpu() if isinstance(x, torch.Tensor) else x) for x in result
        )
        tmp = _cache_path(key) + ".tmp"
        torch.save(cpu_result, tmp)
        os.rename(tmp, _cache_path(key))
        _evict_lru()
    except Exception as e:
        _log(f"save error: {e}")


def _load(key, device):
    try:
        path = _cache_path(key)
        if not os.path.exists(path):
            return None
        data = torch.load(path, map_location="cpu")
        os.utime(path, None)
        return tuple(
            (x.to(device) if isinstance(x, torch.Tensor) else x) for x in data
        )
    except Exception as e:
        _log(f"load error: {e} -- deleting corrupt entry")
        try:
            os.remove(_cache_path(key))
        except Exception:
            pass
        return None


def install_llava_cache(pipeline, target_device="cuda"):
    if not LLAVA_CACHE_ENABLED:
        _log("disabled via flag")
        return
    if id(pipeline) in _installed_pipelines:
        _log("already installed on this pipeline")
        return
    if not hasattr(pipeline, "encode_prompt_audio_text_base"):
        _log("ERROR: pipeline has no encode_prompt_audio_text_base")
        return

    original_fn = pipeline.encode_prompt_audio_text_base
    os.makedirs(LLAVA_CACHE_DIR, exist_ok=True)

    @functools.wraps(original_fn)
    def wrapped(*args, **kwargs):
        try:
            if not _is_llava_call(kwargs):
                _stats["passthrough"] += 1
                if LLAVA_CACHE_LOG:
                    _log("CLIP encoder call -- pass-through")
                return original_fn(*args, **kwargs)

            key = _make_key(args, kwargs)
            cached = _load(key, target_device)
            if cached is not None:
                _stats["hits"] += 1
                _log(f"HIT  | key={key[:12]} | hits={_stats['hits']}")
                return cached

            _stats["misses"] += 1
            t0 = time.time()
            result = original_fn(*args, **kwargs)
            dt = time.time() - t0
            _log(f"MISS | key={key[:12]} | encode={dt:.2f}s | misses={_stats['misses']}")
            _save(key, result)
            return result
        except Exception as e:
            _log(f"wrapper error ({type(e).__name__}: {e}) -- fallback to original")
            return original_fn(*args, **kwargs)

    pipeline.encode_prompt_audio_text_base = wrapped
    pipeline._llava_cache_original = original_fn
    _installed_pipelines.add(id(pipeline))
    _log(f"installed | dir={LLAVA_CACHE_DIR} | max={LLAVA_CACHE_MAX_MB}MB")


def uninstall_llava_cache(pipeline):
    if hasattr(pipeline, "_llava_cache_original"):
        pipeline.encode_prompt_audio_text_base = pipeline._llava_cache_original
        del pipeline._llava_cache_original
        _installed_pipelines.discard(id(pipeline))
        _log("uninstalled")


def print_cache_stats():
    _log(f"stats: hits={_stats['hits']} misses={_stats['misses']} passthrough={_stats['passthrough']}")
