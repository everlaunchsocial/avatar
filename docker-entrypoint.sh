#!/bin/bash
set -e

REPO_DIR="/workspace/HunyuanVideo-Avatar"
WEIGHTS_DIR="$REPO_DIR/weights/ckpts/hunyuan-video-t2v-720p/transformers"

echo "=== EverLaunch Avatar Studio ==="

# Link external weights mount (Verda /mnt/avatar-weights or MODEL_BASE) to expected path
EXTERNAL_WEIGHTS=""
if [ -n "$MODEL_BASE" ] && [ -d "$MODEL_BASE" ]; then
    EXTERNAL_WEIGHTS="$MODEL_BASE"
elif [ -d "/mnt/avatar-weights" ]; then
    EXTERNAL_WEIGHTS="/mnt/avatar-weights"
fi
if [ -n "$EXTERNAL_WEIGHTS" ]; then
    mkdir -p "$REPO_DIR"
    if [ -e "$REPO_DIR/weights" ] && [ ! -L "$REPO_DIR/weights" ]; then
        rm -rf "$REPO_DIR/weights"
    fi
    if [ ! -L "$REPO_DIR/weights" ]; then
        ln -s "$EXTERNAL_WEIGHTS" "$REPO_DIR/weights"
        echo "Linked weights: $REPO_DIR/weights -> $EXTERNAL_WEIGHTS"
    fi
fi

# Pull latest code from GitHub (or clone if first boot)
if [ -d "$REPO_DIR/.git" ]; then
    echo "Pulling latest code..."
    cd "$REPO_DIR"
    git config pull.rebase false
    git config user.email "pod@everlaunch.local"
    git config user.name "runpod"
    git pull origin main 2>/dev/null || echo "Git pull failed (offline or auth issue) - using existing code"
else
    echo "Cloning repository..."
    cd /workspace
    git clone https://github.com/everlaunchsocial/avatar.git HunyuanVideo-Avatar 2>/dev/null || \
    echo "Clone failed - if repo needs auth, clone manually after boot"
fi

# Check if model weights exist
if [ -f "$WEIGHTS_DIR/mp_rank_00_model_states_fp8.pt" ]; then
    echo "Model weights found."
else
    echo ""
    echo "WARNING: Model weights not found at $WEIGHTS_DIR"
    echo "Run this to download (~50GB, ~1 hour):"
    echo "  cd $REPO_DIR/weights && huggingface-cli download tencent/HunyuanVideo-Avatar --local-dir ./"
    echo ""
fi

# Verify FA3
python3 -c "from flash_attn_interface import flash_attn_varlen_func; print('FA3 Hopper: OK')" 2>/dev/null || \
    echo "WARNING: FA3 Hopper not available"

echo "=== Ready ==="
echo ""
echo "Quick render test:"
echo "  cd $REPO_DIR"
echo "  CUDA_VISIBLE_DEVICES=0 python3 hymm_sp/sample_gpu_poor.py --input assets/my_test.csv --ckpt ./weights/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt --sample-n-frames 129 --seed 128 --image-size 704 --cfg-scale 7.5 --infer-steps 50 --use-deepcache 1 --flow-shift-eval-video 5.0 --save-path ./results --use-fp8"
echo ""

# Choose mode based on environment variable
echo "SERVERLESS_MODE=$SERVERLESS_MODE"
echo "WORKER_MODE=$WORKER_MODE"
if [ "$SERVERLESS_MODE" = "true" ]; then
    echo "Starting RunPod Serverless handler..."
    cd "$REPO_DIR"
    python3 scripts/runpod_handler.py
elif [ "$WORKER_MODE" = "true" ]; then
    echo "Starting Supabase worker..."
    cd "$REPO_DIR"
    python3 scripts/worker.py
else
    echo "Interactive mode. SSH in and start working."
    sleep infinity
fi
