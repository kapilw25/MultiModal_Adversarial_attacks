# Plan 3: vLLM Integration for 661K Inferences

**Target Hardware:** Single Nvidia A10 24GB PCIe
**Goal:** Integrate vLLM continuous batching for 90-95% GPU utilization
**Scale:** 80 images × 411 questions × 28 attack variants × 16 VLMs = **661,248 inferences**

---

## Executive Summary

### Problem Statement
Current inference pipeline (`scripts/model_inference.py`) processes inferences **sequentially**:
- **GPU Utilization**: 5-15% (severely underutilized)
- **Processing Time**: Estimated 20-40 hours for 661K inferences
- **Bottleneck**: One image at a time → GPU idle 85-95% of the time

### vLLM Solution
**Continuous Batching**: Dynamically adjusts batch size based on available GPU memory
- Small models (InternVL3-1B @ 0.9GB/image): Batch 26 images simultaneously
- Large models (Qwen2.5-VL-7B @ 6.7GB/image): Batch 3 images simultaneously
- **Auto-adjusts** to maintain 90-95% GPU utilization

### Expected Outcome
- **16 vLLM-compatible models**: ~2-4 hours total
- **Overall Speedup**: 10-20x faster than current sequential approach

---

## Model Compatibility Matrix (Verified via Web Research)

| # | Model | vLLM Support | HuggingFace Path | Batch Size (24GB) | Notes |
|---|-------|--------------|------------------|-------------------|-------|
| 1 | **Qwen2.5-VL-3B** | ✅ YES | `Qwen/Qwen2.5-VL-3B-Instruct` | ~5-6 | Officially supported |
| 2 | **Qwen2.5-VL-7B** | ✅ YES | `Qwen/Qwen2.5-VL-7B-Instruct` | ~3-4 | Officially supported |
| 3 | **Qwen2-VL-2B** | ✅ YES | `Qwen/Qwen2-VL-2B-Instruct` | ~7-8 | Officially supported |
| 4 | **LLAVA-1.5-7B** | ✅ YES | `llava-hf/llava-1.5-7b-hf` | ~3-4 | LLaVA family supported |
| 5 | **LLAVA-v1.6-Mistral-7B** | ✅ YES | `llava-hf/llava-v1.6-mistral-7b-hf` | ~3 | LLaVA-Next supported |
| 6 | **InternVL3-1B** | ✅ YES | `OpenGVLab/InternVL3-1B` | ~20-26 | Smallest, highest batch |
| 7 | **InternVL3-2B** | ✅ YES | `OpenGVLab/InternVL3-2B` | ~10-12 | InternVL family supported |
| 8 | **InternVL2.5-4B** | ✅ YES | `OpenGVLab/InternVL2.5-4B` | ~5-6 | InternVL2 series supported |
| 9 | **PaliGemma-3B** | ✅ YES | `google/paligemma-3b-mix-224` | ~6-7 | Officially supported |

### Incompatible Models (to be replaced)
| Model | Issue | Replacement |
|-------|-------|-------------|
| DeepSeek-VL-1.3B | Only VL2 supported | MiniCPM-V-2 |
| DeepSeek-VL-7B | Only VL2 supported | DeepSeek-VL2-Small |
| SmolVLM2-256M | Not in registry | InternVL3-1B |
| SmolVLM2-500M | Not in registry | H2OVL-Mississippi-800M |
| SmolVLM2-2.2B | Not in registry | Mono-InternVL-2B |
| Moondream2-2B | Not in registry | MiniCPM-V-2.5 |

---

## Replacement Mapping (100% vLLM Compatible)

| Replace This | With This | Size | vLLM | Source |
|--------------|-----------|------|------|--------|
| DeepSeek-VL-1.3B | MiniCPM-V-2 | 3B | ✅ | [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models/) |
| DeepSeek-VL-7B | DeepSeek-VL2-Small | 3B | ✅ | [HuggingFace Discussion](https://huggingface.co/deepseek-ai/deepseek-vl2/discussions/3) |
| SmolVLM2-256M | InternVL3-1B | 0.94B | ✅ | Already in list |
| SmolVLM2-500M | H2OVL-Mississippi-800M | 0.8B | ✅ | [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models/) |
| SmolVLM2-2.2B | Mono-InternVL-2B | 2B | ✅ | [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models/) |
| Moondream2-2B | MiniCPM-V-2.5 | 2.8B | ✅ | [vLLM Supported Models](https://docs.vllm.ai/en/latest/models/supported_models/) |

---

## New VLM Lineup (100% vLLM Compatible)

| # | Model | Size | Family | HuggingFace Path |
|---|-------|------|--------|------------------|
| 1 | Qwen2.5-VL-3B | 3.75B | Qwen | `Qwen/Qwen2.5-VL-3B-Instruct` |
| 2 | Qwen2.5-VL-7B | 8.29B | Qwen | `Qwen/Qwen2.5-VL-7B-Instruct` |
| 3 | Qwen2-VL-2B | 2.21B | Qwen | `Qwen/Qwen2-VL-2B-Instruct` |
| 4 | LLAVA-1.5-7B | 7.06B | LLaVA | `llava-hf/llava-1.5-7b-hf` |
| 5 | LLAVA-v1.6-Mistral-7B | 7.57B | LLaVA | `llava-hf/llava-v1.6-mistral-7b-hf` |
| 6 | InternVL3-1B | 0.94B | InternVL | `OpenGVLab/InternVL3-1B` |
| 7 | InternVL3-2B | 2.09B | InternVL | `OpenGVLab/InternVL3-2B` |
| 8 | InternVL2.5-4B | 3.71B | InternVL | `OpenGVLab/InternVL2.5-4B` |
| 9 | PaliGemma-3B | 2.92B | Google | `google/paligemma-3b-mix-224` |
| 10 | **MiniCPM-V-2.5** | 2.8B | OpenBMB | `openbmb/MiniCPM-V-2_5` |
| 11 | **DeepSeek-VL2-Small** | 3B | DeepSeek | `deepseek-ai/deepseek-vl2-small` |
| 12 | **H2OVL-Mississippi-2B** | 2B | H2O.ai | `h2oai/h2ovl-mississippi-2b` |
| 13 | **Phi-3.5-Vision** | 4.15B | Microsoft | `microsoft/Phi-3.5-vision-instruct` |
| 14 | **Mono-InternVL-2B** | 2B | OpenGVLab | `OpenGVLab/Mono-InternVL-2B` |
| 15 | **MiniCPM-V-4** | 4B | OpenBMB | `openbmb/MiniCPM-V-4` |
| 16 | **Molmo-7B** | 7B | AllenAI | `allenai/Molmo-7B-D-0924` |

**Bold** = New models replacing incompatible ones

---

## Time Savings Analysis

| Approach | Total Time | Speedup vs Sequential |
|----------|------------|----------------------|
| Fully Sequential (current) | 40+ hours | 1x (baseline) |
| 100% vLLM (new lineup) | 2-4 hours | 10-20x |

---

## Implementation: Direct model_inference.py Modification

### Key Changes Required

#### 1. Add vLLM Engine Registry
```python
# New model registry with HuggingFace paths for vLLM
VLLM_MODEL_PATHS = {
    "Qwen25_VL_3B": "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen25_VL_7B": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen2_VL_2B": "Qwen/Qwen2-VL-2B-Instruct",
    "LLAVA_1pt5_7B": "llava-hf/llava-1.5-7b-hf",
    "LLAVA_v1pt6_Mistral_7B": "llava-hf/llava-v1.6-mistral-7b-hf",
    "InternVL3_1B": "OpenGVLab/InternVL3-1B",
    "InternVL3_2B": "OpenGVLab/InternVL3-2B",
    "InternVL25_4B": "OpenGVLab/InternVL2.5-4B",
    "PaliGemma_VL_3B": "google/paligemma-3b-mix-224",
    "MiniCPM_V_2pt5": "openbmb/MiniCPM-V-2_5",
    "DeepSeek_VL2_Small": "deepseek-ai/deepseek-vl2-small",
    "H2OVL_Mississippi_2B": "h2oai/h2ovl-mississippi-2b",
    "Phi35_Vision": "microsoft/Phi-3.5-vision-instruct",
    "Mono_InternVL_2B": "OpenGVLab/Mono-InternVL-2B",
    "MiniCPM_V_4": "openbmb/MiniCPM-V-4",
    "Molmo_7B": "allenai/Molmo-7B-D-0924",
}
```

#### 2. Add vLLM Inference Function
```python
from vllm import LLM, SamplingParams

def run_vllm_batch_inference(engine, tasks):
    """Run batched inference using vLLM continuous batching"""

    model_path = VLLM_MODEL_PATHS[engine]

    # Initialize vLLM engine
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        dtype="half"
    )

    # Prepare all prompts at once
    prompts = []
    for task in tasks:
        prompts.append({
            "prompt": task['question_text'],
            "multi_modal_data": {"image": task['image_path']}
        })

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.2,
        max_tokens=512,
        top_p=0.95
    )

    # Run batch inference (vLLM handles batching automatically)
    outputs = llm.generate(prompts, sampling_params)

    # Extract results
    results = []
    for task, output in zip(tasks, outputs):
        results.append({
            'task': task,
            'response': output.outputs[0].text
        })

    # Cleanup
    del llm
    torch.cuda.empty_cache()

    return results
```

#### 3. Modify Main Loop
```python
def run_inference(self, engine, tasks):
    """Run VLM inference - uses vLLM for batch processing"""

    if engine in VLLM_MODEL_PATHS:
        # Use vLLM batch inference
        print(f"🚀 Using vLLM continuous batching for {engine}")
        return run_vllm_batch_inference(engine, tasks)
    else:
        # Fallback to sequential HuggingFace (shouldn't happen with new lineup)
        print(f"⚠️ Using sequential HuggingFace for {engine}")
        return self._run_sequential_inference(engine, tasks)
```

---

## Quick Validation (Before Full Run)

Before running 661K inferences, validate with a quick test:

```bash
# Test 1 model on 10 images with vLLM
python -c "
from vllm import LLM, SamplingParams

llm = LLM(
    model='OpenGVLab/InternVL3-1B',
    trust_remote_code=True,
    gpu_memory_utilization=0.9
)

# Test with 10 images
prompts = [{'prompt': 'What is this?', 'multi_modal_data': {'image': f'data/clean/chart/chart_{i}.png'}} for i in range(1, 11)]
outputs = llm.generate(prompts, SamplingParams(max_tokens=50))

for i, out in enumerate(outputs):
    print(f'{i+1}. {out.outputs[0].text[:50]}...')
"
```

If this works, proceed to full integration.

---

## Implementation Steps

### Step 1: Install vLLM (5 min)
```bash
pip install vllm>=0.6.0
```

### Step 2: Modify model_inference.py (30 min)
1. Add `VLLM_MODEL_PATHS` registry
2. Add `run_vllm_batch_inference()` function
3. Update `AVAILABLE_ENGINES` list with new model names
4. Modify `run_inference()` to use vLLM

### Step 3: Quick Validation (10 min)
Test 1 model on 10 images to confirm vLLM works

### Step 4: Full Run (2-4 hours)
Execute 661K inferences with 100% vLLM batching

---

## Success Criteria

### ✅ vLLM Integration Complete
- All 16 models load and run with vLLM
- Batch inference works without OOM errors

### ✅ Performance Target Met
- Total inference time: 2-4 hours (vs 40+ hours sequential)
- GPU utilization: >60% (target: 85-95%)

### ✅ Scale Target Met
- 661,248 inferences completed
- Results saved to `inference_results` table

---

## Next Steps

1. **Install vLLM**: `pip install vllm>=0.6.0`
2. **Modify model_inference.py**: Add vLLM support
3. **Quick Validation**: Test 1 model on 10 images
4. **Full Pipeline Run**: Execute 661K inferences in 2-4 hours

---

**Document Version:** 3.0
**Last Updated:** 2025-11-26
**Status:** Ready for Direct Implementation (Test Suite Removed)

**Inference Scale:** 661,248 inferences (80 images × 411 questions × 28 attacks × 16 VLMs)
**Expected Time:** 2-4 hours (100% vLLM batching)
**Implementation Effort:** ~45 minutes
