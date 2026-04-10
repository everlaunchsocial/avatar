FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Layer 1: OS packages (rarely changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg ninja-build libcairo2-dev pkg-config libgirepository1.0-dev git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: Python base packages from lock file (changes when you update requirements)
COPY requirements.lock.txt /tmp/requirements.lock.txt
RUN grep -v "^flash\|^pycairo\|^opencv" /tmp/requirements.lock.txt > /tmp/reqs_clean.txt && \
    pip install --no-cache-dir -r /tmp/reqs_clean.txt \
    --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir "numpy<2" opencv-python-headless decord deepcache \
    insightface onnxruntime-gpu moviepy pandas omegaconf timm ftfy sentencepiece \
    supabase requests runpod && \
    rm -f /tmp/reqs_clean.txt /tmp/requirements.lock.txt

# Layer 3: FlashAttention 3 Hopper (pre-compiled binary, no compile needed)
COPY prebuilt_fa3/flash_attn_interface.py /usr/local/lib/python3.11/dist-packages/
COPY prebuilt_fa3/flashattn_hopper_cuda.cpython-311-x86_64-linux-gnu.so /usr/local/lib/python3.11/dist-packages/
COPY prebuilt_fa3/flashattn_hopper_cuda.py /usr/local/lib/python3.11/dist-packages/

# Layer 4: Startup script (pulls fresh code + checks weights on boot)
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Environment variables
ENV PYTHONPATH=/workspace/HunyuanVideo-Avatar
ENV MODEL_BASE=/workspace/HunyuanVideo-Avatar/weights
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /workspace

ENTRYPOINT ["/docker-entrypoint.sh"]
