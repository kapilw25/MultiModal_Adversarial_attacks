━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Summary: vLLM Integration & Testing Strategy

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Testing Order on A10

1. Quick validation (5 min): Test 1 model (InternVL3-1B) on 10 images
2. Single model full run (15 min): Run InternVL3-1B on all 675 adversarial images
3. Full pipeline (2-4 hours): Run all 16 models on 661K inferences

# Step 1: Quick validation
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='OpenGVLab/InternVL3-1B', trust_remote_code=True, gpu_memory_utilization=0.9)
print('✅ vLLM initialized successfully')
"

# Step 2: Full pipeline (all 4 scripts synced with epsilon schema)
python scripts/model_inference_vLLM.py   # 207K inferences (25 images)
python scripts/model_evaluation.py
python scripts/model_benchmark_robustness.py  # interactive menu or --auto

---
## Unit Test

**File:** `unit_test/test_inference_comparison.py`

```bash
# List all engines
python3 unit_test/test_inference_comparison.py --list

# Common engines (sequential vs vLLM comparison)
python3 unit_test/test_inference_comparison.py --engine Qwen25_VL_3B
python3 unit_test/test_inference_comparison.py --all

# vLLM-only engines (no comparison, just format check)
python3 unit_test/test_inference_comparison.py --vllm-only --engine MiniCPM_V_2pt5
python3 unit_test/test_inference_comparison.py --vllm-only
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Attack Images: 100% Complete

- 675/675 images generated (25 images × 9 attacks × 3 epsilon levels)
- attack_runner.py work is done - no GPU needed for this

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  LEGACY (Sequential HuggingFace) - archived to scripts/legacy/:
  model_inference_sequential.py
    → vlm_local_client.py (send_chat_request)
      → MODEL_MAPPING["Qwen25_VL_3B"] = "Qwen2.5-VL-3B-Instruct_4bit"
        → model_classes.py → create_model()
          → local_model/models/Qwen25_3B.py (custom wrapper)
            → model.predict(image, question) ← ONE IMAGE AT A TIME

  CURRENT - vLLM (Batch - BYPASSES local_model/models/):
  model_inference_vLLM.py
    → VLLM_MODEL_PATHS["Qwen25_VL_3B"] = "Qwen/Qwen2.5-VL-3B-Instruct"
      → vLLM.LLM(model=HuggingFace_path) ← LOADS DIRECTLY FROM HF
        → llm.generate([batch]) ← MULTIPLE IMAGES AT ONCE

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
| Category | Engines |
|----------|---------|
| Common (10) | Qwen25_VL_3B, Qwen25_VL_7B, Qwen2_VL_2B, LLAVA_1pt5_7B, LLAVA_v1pt6_Mistral_7B, InternVL3_1B, InternVL3_2B, InternVL25_4B, PaliGemma_VL_3B |
| vLLM-only (7) | MiniCPM_V_2pt5, DeepSeek_VL2_Small, H2OVL_Mississippi_2B, Phi35_Vision, Mono_InternVL_2B, MiniCPM_V_4, Molmo_7B |

**Test Checks:** Format "The answer is X", no errors, reasonable length

---
## Next Steps (A10 GPU)

1. Unit test validation: `python3 unit_test/test_inference_comparison.py --engine InternVL3_1B`
2. Full vLLM pipeline: `python scripts/model_inference_vLLM.py`



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Corrected Inference Calculation

ACTUAL INFERENCE COUNT (25 images, 411 questions):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Questions per image set:  411
Attack variants:          1 original + (9 attacks × 3 epsilon) = 28
Total question instances: 411 × 28 = 11,508
VLMs:                     16 local + 2 cloud = 18

TOTAL INFERENCES = 11,508 × 18 = 207,144 inferences
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This matches the plan's estimate of ~184,128 (16 VLMs × 11,508)
The difference is because plan used 16 VLMs, you have 18.

---
If Scaled to 80 Images

Assuming similar question distribution (~16.4 questions/image):

SCALED INFERENCE COUNT (80 images, ~1,312 questions):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Questions:               80 × 16.4 ≈ 1,312
Attack variants:         28
Total question instances: 1,312 × 28 = 36,736
VLMs:                    18

TOTAL INFERENCES = 36,736 × 18 = 661,248 inferences
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━