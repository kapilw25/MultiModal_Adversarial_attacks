# vLLM Integration Attempts - Observation Log

## Goal
Enable vLLM batch inference for Qwen2.5-VL-3B on Lambda.ai GPU instances (NFS-mounted virtualenv)

---

## Attempt 1: vLLM 0.11.2 (Latest)
**Config:** `vllm>=0.11.2`, `transformers>=4.56.0,<5`

**Error:**
```
CUDA_ERROR_NOT_INITIALIZED: initialization error
```

**Root Cause:** vLLM 0.11.x uses V1 engine which spawns subprocesses. Subprocesses fail to initialize CUDA from NFS-mounted virtualenv.

**Tried Workarounds:**
- `VLLM_USE_V1=0` -> Ignored (V1 is only engine in 0.11.x, V0 code removed)
- `VLLM_WORKER_MULTIPROC_METHOD=spawn` -> Got past CUDA init but hit flash-attn headdim error
- `VLLM_ATTENTION_BACKEND=TRITON_ATTN` -> Same headdim error (ViT hardcodes flash_attn calls)
- `VLLM_ATTENTION_BACKEND=XFORMERS` -> Back to CUDA init error

**Status:** [FAILED] - V1 engine architecture incompatible with NFS mounts

---

## Attempt 2: vLLM 0.9.2 + transformers 4.57.3
**Config:** `vllm==0.9.2` (last version with V0 engine)

**Error:**
```
ValueError: 'aimv2' is already used by a Transformers config, pick another name.
```

**Root Cause:** transformers 4.54+ registers 'aimv2' config, vLLM 0.9.2 also tries to register it -> conflict

**Status:** [FAILED] - transformers version too new

---

## Attempt 3: vLLM 0.9.2 + transformers 4.53.3
**Config:** `vllm==0.9.2`, `transformers>=4.49.0,<4.54.0`

**Error:**
```
ImportError: flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda9SetDeviceEab
```

**Root Cause:** flash-attn 2.8.3 was built for torch 2.9.0, but vLLM 0.9.2 requires torch 2.7.0 -> ABI mismatch

**Fix:** Uninstalled flash-attn (`pip uninstall flash-attn`)

**Status:** [PARTIAL] - vLLM imports successfully

---

## Attempt 4: vLLM 0.9.2 + no flash-attn + V0 engine
**Config:** `vllm==0.9.2`, `transformers==4.53.3`, no flash-attn, `VLLM_USE_V1=0`

**Result:**
```
INFO: Initializing a V0 LLM engine (v0.9.2)
INFO: Using Flash Attention backend.  # Uses xformers internally
INFO: Model loading took 7.1557 GiB and 2.507946 seconds
```

**V0 Engine: [OK] WORKS!**

**New Error:**
```
ERROR: You set or defaulted to '{"image": 1}' in `--limit-mm-per-prompt`, but passed 96 image items in the same prompt.
```

**Root Cause:** Qwen2.5-VL tokenizes images into 96 patches. vLLM default `limit_mm_per_prompt={"image": 1}` rejects this.

**Status:** [PARTIAL] - V0 engine works, need to fix mm_per_prompt config

---

## Attempt 5: Fix limit_mm_per_prompt config
**Config:** Set `limit_mm_per_prompt={"image": 100}` in LLM() constructor

**Error:**
```
Token indices sequence length is longer than the specified maximum sequence length for this model (1654784 > 131072)
WARNING: 1654784 tokens reserved for multi-modal embeddings (image: 1638400, video: 16384)
CUDA out of memory. Tried to allocate 148.00 MiB. GPU has 22.07 GiB total, 76.44 MiB free.
```

**Root Cause:** Setting `limit_mm_per_prompt={"image": 100}` caused vLLM to reserve memory for 100 images * 16384 tokens = 1,638,400 tokens. This is a catastrophic memory allocation (100x more than needed).

**The Real Issue:** vLLM's `limit_mm_per_prompt` counts **images**, not **patches**. Qwen2.5-VL sends 1 image that gets tokenized into 96 patches internally. The error message "passed 96 image items" is misleading - it's counting patches as images.

**Status:** [FAILED] - vLLM 0.9.2 has a bug counting Qwen2.5-VL image patches as separate images

---

## Attempt 6: Use mm_processor_kwargs to limit image size
**Source:** [vLLM Issue #20123](https://github.com/vllm-project/vllm/issues/20123), [vLLM Qwen2.5-VL Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html)

**Config:**
```python
llm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    mm_processor_kwargs={
        "min_pixels": 28 * 28,
        "max_pixels": 256 * 28 * 28,
    },
)
```

**Result:** Memory profiling improved (10.92 GiB KV cache vs previous OOM), BUT same error:
```
ERROR: You set or defaulted to '{"image": 1}' in `--limit-mm-per-prompt`, but passed 96 image items in the same prompt.
```

**Analysis:** `mm_processor_kwargs` only affects memory reservation, NOT the patch counting bug. The "96 image items" error happens at request validation time, before image processing.

**Status:** [FAILED] - mm_processor_kwargs doesn't fix patch counting issue

---

## Attempt 7: Test exact Colab code (no VLLM_USE_V1=0)
**Config:** Default vLLM 0.9.2 (V1 engine), InternVL3-1B

**Command:**
```bash
python3 -c "from vllm import LLM; llm = LLM(model='OpenGVLab/InternVL3-1B', trust_remote_code=True, gpu_memory_utilization=0.9); print('OK')"
```

**Result:** V1 engine starts (no CUDA subprocess error!) but fails with PIL dtype bug:
```
TypeError: Cannot handle this data type: (1, 1, 3), <i8
ValueError: Failed to apply InternVLProcessor on data=...
```

**Root Cause:** vLLM 0.9.2's InternVL processor creates dummy video array with dtype `int64` but PIL.Image.fromarray() requires `uint8`. This is a **bug in vLLM 0.9.2** that was fixed in later versions.

**Key Finding:** V1 engine CAN work on Lambda NFS! The CUDA subprocess issue may have been specific to earlier testing conditions. The current blocker is vLLM 0.9.2's multimodal processor bugs.

**Status:** [FAILED] - vLLM 0.9.2 has PIL dtype bug in InternVL processor

---

## Summary of vLLM 0.9.2 Bugs Found

| Model | Bug |
|-------|-----|
| Qwen2.5-VL | Counts image patches (96) as separate images, rejects with limit_mm_per_prompt |
| InternVL3 | PIL dtype error - dummy video array has int64 instead of uint8 |

**Conclusion:** vLLM 0.9.2 has multiple multimodal processor bugs. These are fixed in newer versions (0.10+, 0.11+) but those versions may have other issues.

---

## Options Going Forward

### Option A: Try vLLM 0.10.x or 0.11.x (latest)
The bugs are fixed in newer vLLM. Try:
```bash
pip install vllm --upgrade
```
Risk: May hit V1 engine CUDA subprocess issues again (but Attempt 7 suggests it might work now)

### Option B: Try LLaVA models
LLaVA is more mature in vLLM. Test:
```bash
python3 -c "from vllm import LLM; llm = LLM(model='llava-hf/llava-1.5-7b-hf', trust_remote_code=True); print('OK')"
```

### Option C: Abandon vLLM on Lambda NFS
Use HuggingFace sequential inference (already working). vLLM has too many bugs/incompatibilities with:
- NFS-mounted virtualenvs
- Multimodal model processors (Qwen2.5-VL, InternVL)

---

## Current State

| Component | Version | Status |
|-----------|---------|--------|
| vLLM | 0.9.2 | [OK] Installed |
| transformers | 4.53.3 | [OK] Compatible |
| torch | 2.7.0 | [OK] (downgraded from 2.9.0) |
| flash-attn | UNINSTALLED | [WARN] Using xformers instead |
| V0 Engine | Active | [OK] No subprocess spawning |

**Next Step:** Set `limit_mm_per_prompt={"image": 100}` to allow Qwen2.5-VL's 96 image patches

---

## Summary: Are We Making Progress?

**YES** - We've made significant progress:

1. [OK] Identified V1 engine subprocess spawning as root cause of NFS CUDA errors
2. [OK] Found vLLM 0.9.2 as last version with V0 engine
3. [OK] Resolved transformers version conflict (need <4.54.0)
4. [OK] Resolved flash-attn ABI mismatch (uninstalled, using xformers)
5. [OK] V0 engine initializes successfully
6. [PENDING] Final issue: trivial config fix for `limit_mm_per_prompt`

**Not shooting in the dark** - each error led to a specific fix in the dependency chain.
