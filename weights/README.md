---
pipeline_tag: image-to-video
language:
- en
---
<!-- ## **HunyuanVideo-Avatar** -->

<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/646d7592bb95b5d4001e5a04/HDZpvr8F-UaHAHlsF--fh.png"  height=100>
</p>
 
<div align="center">
  <a href="https://github.com/Tencent-Hunyuan/HunyuanVideo-Avatar"><img src="https://img.shields.io/static/v1?label=HunyuanVideo-Avatar%20Code&message=Github&color=blue"></a>
  <a href="https://HunyuanVideo-Avatar.github.io/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=green"></a>
  <a href="https://hunyuan.tencent.com/modelSquare/home/play?modelId=126"><img src="https://img.shields.io/static/v1?label=Playground&message=Web&color=green"></a>
  <a href="https://arxiv.org/pdf/2505.20156"><img src="https://img.shields.io/static/v1?label=Tech Report&message=Arxiv&color=red"></a>
  <a href="https://huggingface.co/tencent/HunyuanVideo-Avatar"><img src="https://img.shields.io/static/v1?label=HunyuanVideo-Avatar&message=HuggingFace&color=yellow"></a>
</div>



![image](assets/teaser.png)

> [**HunyuanVideo-Avatar: High-Fidelity Audio-Driven Human Animation for Multiple Characters**](https://arxiv.org/pdf/2505.20156) <be>

## **Abstract**

Recent years have witnessed significant progress in audio-driven human animation. However, critical challenges remain in (i) generating highly dynamic videos while preserving character consistency, (ii) achieving precise emotion alignment between characters and audio, and (iii) enabling multi-character audio-driven animation. To address these challenges, we propose HunyuanVideo-Avatar, a multimodal diffusion transformer (MM-DiT)-based model capable of simultaneously generating dynamic, emotion-controllable, and multi-character dialogue videos. Concretely, HunyuanVideo-Avatar introduces three key innovations: (i) A character image injection module is designed to replace the conventional addition-based character conditioning scheme, eliminating the inherent condition mismatch between training and inference. This ensures the dynamic motion and strong character consistency; (ii) An Audio Emotion Module (AEM) is introduced to extract and transfer the emotional cues from an emotion reference image to the target generated video, enabling fine-grained and accurate emotion style control; (iii) A Face-Aware Audio Adapter (FAA) is proposed to isolate the audio-driven character with latent-level face mask, enabling independent audio injection via cross-attention for multi-character scenarios. These innovations empower HunyuanVideo-Avatar to surpass state-of-the-art methods on benchmark datasets and a newly proposed wild dataset, generating realistic avatars in dynamic, immersive scenarios. The source code and model weights will be released publicly.

## **HunyuanVideo-Avatar Overall Architecture**

![image](https://cdn-uploads.huggingface.co/production/uploads/646d7592bb95b5d4001e5a04/SAQAlLLsEzC1fURoL89_C.png)

We propose **HunyuanVideo-Avatar**, a multi-modal diffusion transformer(MM-DiT)-based model capable of generating **dynamic**, **emotion-controllable**, and **multi-character dialogue** videos.

## 🎉 **HunyuanVideo-Avatar Key Features**

![image](https://cdn-uploads.huggingface.co/production/uploads/646d7592bb95b5d4001e5a04/RVM42NLlvlwABiQNlTLdd.png)

### **High-Dynamic and Emotion-Controllable Video Generation**

HunyuanVideo-Avatar supports animating any input **avatar images** to **high-dynamic** and **emotion-controllable** videos with simple **audio conditions**. Specifically, it takes as input **multi-style** avatar images at **arbitrary scales and resolutions**. The system supports multi-style avatars encompassing photorealistic, cartoon, 3D-rendered, and anthropomorphic characters. Multi-scale generation spanning portrait, upper-body and full-body. It generates videos with high-dynamic foreground and background, achieving superior realistic and naturalness. In addition, the system supports controlling facial emotions of the characters conditioned on input audio. 

### **Various Applications**

HunyuanVideo-Avatar supports various downstream tasks and applications. For instance, the system generates talking avatar videos, which could be applied to e-commerce, online streaming, social media video production, etc. In addition, its multi-character animation feature enlarges the application such as video content creation, editing, etc. 

## 🚀 Parallel Inference on Multiple GPUs

For example, to generate a video with 8 GPUs, you can use the following command:

```bash
cd HunyuanVideo-Avatar

JOBS_DIR=$(dirname $(dirname "$0"))
export PYTHONPATH=./
export MODEL_BASE="./weights"
checkpoint_path=${MODEL_BASE}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt

torchrun --nnodes=1 --nproc_per_node=8 --master_port 29605 hymm_sp/sample_batch.py \
    --input 'assets/test.csv' \
    --ckpt ${checkpoint_path} \
    --sample-n-frames 129 \
    --seed 128 \
    --image-size 704 \
    --cfg-scale 7.5 \
    --infer-steps 50 \
    --use-deepcache 1 \
    --flow-shift-eval-video 5.0 \
    --save-path ${OUTPUT_BASEPATH} 
```

## 🔑 Single-gpu Inference

For example, to generate a video with 1 GPU, you can use the following command:

```bash
cd HunyuanVideo-Avatar

JOBS_DIR=$(dirname $(dirname "$0"))
export PYTHONPATH=./

export MODEL_BASE=./weights
OUTPUT_BASEPATH=./results-single
checkpoint_path=${MODEL_BASE}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt

export DISABLE_SP=1 
CUDA_VISIBLE_DEVICES=0 python3 hymm_sp/sample_gpu_poor.py \
    --input 'assets/test.csv' \
    --ckpt ${checkpoint_path} \
    --sample-n-frames 129 \
    --seed 128 \
    --image-size 704 \
    --cfg-scale 7.5 \
    --infer-steps 50 \
    --use-deepcache 1 \
    --flow-shift-eval-video 5.0 \
    --save-path ${OUTPUT_BASEPATH} \
    --use-fp8 \
    --infer-min
```

### Run with very low VRAM

```bash
cd HunyuanVideo-Avatar

JOBS_DIR=$(dirname $(dirname "$0"))
export PYTHONPATH=./

export MODEL_BASE=./weights
OUTPUT_BASEPATH=./results-poor

checkpoint_path=${MODEL_BASE}/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt

export CPU_OFFLOAD=1
CUDA_VISIBLE_DEVICES=0 python3 hymm_sp/sample_gpu_poor.py \
    --input 'assets/test.csv' \
    --ckpt ${checkpoint_path} \
    --sample-n-frames 129 \
    --seed 128 \
    --image-size 704 \
    --cfg-scale 7.5 \
    --infer-steps 50 \
    --use-deepcache 1 \
    --flow-shift-eval-video 5.0 \
    --save-path ${OUTPUT_BASEPATH} \
    --use-fp8 \
    --cpu-offload \
    --infer-min
```


## Run a Gradio Server
```bash
cd HunyuanVideo-Avatar

bash ./scripts/run_gradio.sh

```

## 🔗 BibTeX

If you find [HunyuanVideo-Avatar](https://arxiv.org/pdf/2505.20156) useful for your research and applications, please cite using this BibTeX:

```BibTeX
@misc{hu2025HunyuanVideo-Avatar,
      title={HunyuanVideo-Avatar: High-Fidelity Audio-Driven Human Animation for Multiple Characters}, 
      author={Yi Chen and Sen Liang and Zixiang Zhou and Ziyao Huang and Yifeng Ma and Junshu Tang and Qin Lin and Yuan Zhou and Qinglin Lu},
      year={2025},
      eprint={2505.20156},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/pdf/2505.20156}, 
}
```

## Acknowledgements

We would like to thank the contributors to the [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [Xtuner](https://github.com/InternLM/xtuner), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration.