# Vision-Language Models (VLMs) by Family and Size
# just focus on <=3B VLMs

## Model Family Size Matrix

| Family | 0-1B | 1-2B | 2-3B | 3-4B | 4-5B | 5-6B | 6-7B |
|--------|------|------|------|------|------|------|------|
| DeepSeek VL | - | [deepseek-vl-1.3b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-chat) ✅ | - | - | - | - | [deepseek-vl-7b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat) ✅ |
| DeepSeek VL2 (MoE) | [deepseek-vl2-tiny](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny) ❌ (40GB req) | - | [deepseek-vl2-small](https://huggingface.co/deepseek-ai/deepseek-vl2-small) ❌ (80GB req) | - | [deepseek-vl2](https://huggingface.co/deepseek-ai/deepseek-vl2) ❌ (>80GB req) | - | - |
| Qwen VL | - | [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) ✅ | [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) ✅ | - | - | - | [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) ✅ |
| Google | - | - | [paligemma-3b-mix-224](https://huggingface.co/google/paligemma-3b-mix-224) ✅ | [gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) ✅ | - | - | [gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it) |
| SmolVLM | [SmolVLM2-256M](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct) ✅ | [SmolVLM2-500M](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) ✅ | [SmolVLM2-2.2B](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) ✅ | - | - | - | - |
| GLM Edge | - | - | [glm-edge-v-2b](https://huggingface.co/THUDM/glm-edge-v-2b) | - | - | [glm-edge-v-5b](https://huggingface.co/THUDM/glm-edge-v-5b) | - |
| Moondream | - | [moondream-2b](https://huggingface.co/moondream/moondream-2b-2025-04-14-4bit) | - | - | - | - | - |
| Microsoft | - | - | - | [Phi-3.5-vision](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) | - | - | - |

## Legend
- ✅ = Implemented and working in our framework
- ❌ = Not feasible on 8GB GPU (with reason)
- Empty cell = No model available in this size category for this family

## Other Notable Models

### <1B Models
1. **ds4sd/SmolDocling-256M-preview** (0.3B)
2. **ByteDance/Dolphin** (0.4B)
3. **stepfun-ai/GOT-OCR2_0** (0.7B)
4. **gokaygokay/Florence-2-Flux-Large** (0.8B)
5. **5CD-AI/Vintern-1B-v3_5** (0.9B)
6. **lmstudio-community/gemma-3n-E4B-it-MLX-4bit** (1B)
7. **AIDC-AI/Ovis2-1B** (1B)

### Specialized Models
1. **prithivMLmods/Qwen2-VL-OCR-2B-Instruct** (2B) - OCR specialized variant

## Recommended Next Implementations (8GB GPU Compatible)
1. **moondream/moondream-2b-2025-04-14-4bit** (2B)


## After loading and testing 20 VLM models
Remove Memory Optimizations:
- Maintained the memory limits (6GB for GPU, 16GB for CPU)
- Kept the smaller image size (224x224)
- Retained the reduced token generation count (64 tokens)
