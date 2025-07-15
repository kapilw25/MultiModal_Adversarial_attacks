# Goal: to integrate atleast 30 VLM's by Family and Size, [max 10 families]
# with 80% models from <=3B size VLMs

## Model Family Size Matrix

| Family | (0-1]B | (1-2]B | (2-3]B | (3-4]B | (4-5]B | (5-6]B | (6-7]B |
|--------|--------|--------|--------|--------|--------|--------|--------|
| DeepSeek VL | - | [deepseek-vl-1.3b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-1.3b-chat) ✅ | - | - | - | - | [deepseek-vl-7b-chat](https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat) ✅ |
| DeepSeek VL2 (MoE) | [deepseek-vl2-tiny](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny) ❌ (40GB req) | - | [deepseek-vl2-small](https://huggingface.co/deepseek-ai/deepseek-vl2-small) ❌ (80GB req) | - | [deepseek-vl2](https://huggingface.co/deepseek-ai/deepseek-vl2) ❌ (>80GB req) | - | - |
| Qwen VL | - | - | [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) (2.21B BF16) ✅ | [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) (3.75B BF16) ✅ | - | - | [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) ✅ |
| Google | - | - | [paligemma-3b-mix-224](https://huggingface.co/google/paligemma-3b-mix-224) ✅ | [gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) ✅ | - | - | [gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it) ⏳ |
| SmolVLM | [SmolVLM2-256M](https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct) ✅ | [SmolVLM2-500M](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) ✅ | [SmolVLM2-2.2B](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) ✅ | - | - | - | - |
| GLM Edge | - | - | [glm-edge-v-2b](https://huggingface.co/THUDM/glm-edge-v-2b) ✅ | - | - | [glm-edge-v-5b](https://huggingface.co/THUDM/glm-edge-v-5b) ❌ (dtype issues) | - |
| Moondream | - | [moondream-2b](https://huggingface.co/moondream/moondream-2b-2025-04-14-4bit) ✅ | - | - | - | - | - |
| Microsoft | [Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) (0.23B) ✅, [Florence-2-large](https://huggingface.co/microsoft/Florence-2-large) (0.77B) ✅, [udop-large](https://huggingface.co/microsoft/udop-large) (0.74B F32) ❌ (task mismatch) | - | [GUI-Actor-2B-Qwen2-VL](https://huggingface.co/microsoft/GUI-Actor-2B-Qwen2-VL) (2.23B BF16) ❌ (task mismatch) | [GUI-Actor-3B-Qwen2.5-VL](https://huggingface.co/microsoft/GUI-Actor-3B-Qwen2.5-VL) (3.79B BF16) ❌ (task mismatch) | [Phi-3.5-vision](https://huggingface.co/microsoft/Phi-3.5-vision-instruct) (4.15B BF16) ⏳ | - | - |
| LLaVA Hybrid | [Zhang199/TinyLLaVA-Qwen2-0.5B-SigLIP](https://huggingface.co/Zhang199/TinyLLaVA-Qwen2-0.5B-SigLIP) ❌ (custom arch), [jiajunlong/TinyLLaVA-0.89B](https://huggingface.co/jiajunlong/TinyLLaVA-0.89B) ❌ (custom arch) | [Intel/llava-gemma-2b](https://huggingface.co/Intel/llava-gemma-2b) (2.82B F32) ❌ | [tinyllava/TinyLLaVA-Gemma-SigLIP-2.4B](https://huggingface.co/tinyllava/TinyLLaVA-Gemma-SigLIP-2.4B) ❌ (custom arch), [Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP](https://huggingface.co/Zhang199/TinyLLaVA-Qwen2.5-3B-SigLIP) ❌ (custom arch) | [tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B](https://huggingface.co/tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B) ❌ (custom arch) | - | - | - |
| InternVL | [InternVL3-1B](https://huggingface.co/OpenGVLab/InternVL3-1B) ✅ | [InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B) ✅ | - | - | [InternVL2_5-4B](https://huggingface.co/OpenGVLab/InternVL2_5-4B) ✅ | - | - |

## Legend
- ✅ = Implemented and working in our framework
- ❌ = Not feasible on 8GB GPU (with reason)
- ❌ (custom arch) = Model uses custom architecture not compatible with standard Transformers library
- ❌ (task mismatch) = Model not suitable for chart analysis tasks
- ⏳ = Model yet to be built and tested
- Empty cell = No model available in this size category for this family

### GUI-Actor Models
- GUI-Actor models are designed for GUI interaction tasks (pointing/clicking), not chart analysis
- While loadable (2.2B params, ~1.7s), they don't support text generation for VQA tasks
- Pointer prediction requires full GUI-Actor package with CUDA dependencies

### UDOP Model
- UDOP is designed for document understanding with structured OCR data, not chart analysis
- Testing shows it only returns single-word answers like "presentation" for chart questions

### TinyLLaVA Models
- OpenELMForCausalLM lacks 'generate' and '_prepare_generation_config' methods; AttributeError when model.language_model.generate() is called.
- `RuntimeError` : tensor size mismatch (256 vs 729) when processing images through vision_tower due to incompatible embedding dimensions.

### Intel/llava-gemma-2b Model
- Model loads successfully on GPU but fails during image processing with `TypeError: unsupported operand type(s) for //: 'int' and 'NoneType'` due to missing patch_size attribute
- Generates `ValueError: Image features and image tokens do not match: tokens: 1, features 1179648` when attempting inference, indicating incompatibility between image processing pipeline and model architecture
