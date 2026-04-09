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
    supabase requests && \
    rm -f /tmp/reqs_clean.txt /tmp/requirements.lock.txt

# Layer 3: FlashAttention 3 Hopper compile (changes rarely, cached aggressively)
RUN cd /workspace && \
    git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    git checkout v2.6.3 && \
    git submodule update --init --recursive && \
    cd hopper && \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    TORCH_CUDA_ARCH_LIST="9.0a" \
    MAX_JOBS=1 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE \
    python3 setup.py install && \
    cd /workspace && rm -rf flash-attention

# Layer 4: Startup script (pulls fresh code + checks weights on boot)
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Environment variables
ENV PYTHONPATH=/workspace/HunyuanVideo-Avatar
ENV MODEL_BASE=/workspace/HunyuanVideo-Avatar/weights
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /workspace

ENTRYPOINT ["/docker-entrypoint.sh"]
