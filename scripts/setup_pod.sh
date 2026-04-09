#!/bin/bash
# EverLaunch HunyuanVideo-Avatar pod recovery script
# Target: RunPod H100 SXM with runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 template
# Usage: bash scripts/setup_pod.sh
set -e

echo "=== 1. OS packages ==="
apt-get update
apt-get install -y ffmpeg ninja-build libcairo2-dev pkg-config libgirepository1.0-dev

echo "=== 2. Python packages from lock file (excluding flash_attn which we build from source) ==="
grep -v "^flash_attn\|^flash-attn\|^pycairo" /workspace/HunyuanVideo-Avatar/requirements.lock.txt > /tmp/reqs_nofa.txt
pip install -r /tmp/reqs_nofa.txt --extra-index-url https://download.pytorch.org/whl/cu124
pip install "numpy<2"
pip install opencv-python-headless decord deepcache insightface onnxruntime-gpu moviepy pandas omegaconf timm ftfy sentencepiece protobuf pyyaml packaging easydict

echo "=== 3. Clone flash-attention at v2.6.3 (Tencent's specified version) ==="
cd /workspace
if [ ! -d "flash-attention" ]; then
    git clone https://github.com/Dao-AILab/flash-attention.git
fi
cd flash-attention
git fetch --tags
git checkout v2.6.3
git submodule update --init --recursive

echo "=== 4. Compile FA3 Hopper kernels for sm_90a ==="
cd /workspace/flash-attention/hopper
CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH TORCH_CUDA_ARCH_LIST="9.0a" MAX_JOBS=4 FLASH_ATTENTION_FORCE_BUILD=TRUE python3 setup.py install || echo "setup.py install failed (expected due to easy_install bug), using pip install next"

echo "=== 5. Install FA3 via pip (bypasses easy_install UTF-8 bug) ==="
cd /workspace/flash-attention/hopper
pip install . --no-build-isolation --no-deps

echo "=== 6. Verify FA3 Hopper import ==="
python3 -c "from flash_attn_interface import flash_attn_varlen_func; import inspect; print('FA3 signature:', inspect.signature(flash_attn_varlen_func))"

echo ""
echo "=== SETUP COMPLETE ==="
echo "Required env vars for rendering:"
echo "  export PYTHONPATH=/workspace/HunyuanVideo-Avatar"
echo "  export MODEL_BASE=/workspace/HunyuanVideo-Avatar/weights"
echo "  export TOKENIZERS_PARALLELISM=false"
echo ""
echo "Test render command:"
echo "  cd /workspace/HunyuanVideo-Avatar"
echo "  CUDA_VISIBLE_DEVICES=0 python3 hymm_sp/sample_gpu_poor.py --input 'assets/my_test.csv' --ckpt ./weights/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt --sample-n-frames 129 --seed 128 --image-size 704 --cfg-scale 7.5 --infer-steps 50 --use-deepcache 1 --flow-shift-eval-video 5.0 --save-path ./results --use-fp8"
