#!/usr/bin/env python3
"""
Quick test: Which VLMs can initialize with vLLM V0 engine?
Tests model loading only, not inference.
"""
import os
os.environ["VLLM_USE_V1"] = "0"  # Force V0 engine

import sys
import gc
import torch

# Test models (smaller ones first to save time)
VLLM_MODEL_PATHS = {
    # Small models first
    "InternVL3_1B": "OpenGVLab/InternVL3-1B",
    "Qwen2_VL_2B": "Qwen/Qwen2-VL-2B-Instruct",
    "InternVL3_2B": "OpenGVLab/InternVL3-2B",
    "Qwen25_VL_3B": "Qwen/Qwen2.5-VL-3B-Instruct",
    "PaliGemma_VL_3B": "google/paligemma-3b-mix-224",
    "InternVL25_4B": "OpenGVLab/InternVL2.5-4B",

    # Larger models
    "LLAVA_1pt5_7B": "llava-hf/llava-1.5-7b-hf",
    "LLAVA_v1pt6_Mistral_7B": "llava-hf/llava-v1.6-mistral-7b-hf",
    "Qwen25_VL_7B": "Qwen/Qwen2.5-VL-7B-Instruct",
}

def cleanup_gpu():
    """Clean up GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def test_model_init(name: str, path: str) -> dict:
    """Test if a model can initialize with vLLM"""
    result = {"name": name, "path": path, "init_ok": False, "error": None}

    try:
        from vllm import LLM

        print(f"\n{'='*60}")
        print(f"Testing: {name} ({path})")
        print(f"{'='*60}")

        llm = LLM(
            model=path,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            dtype="half",
            enforce_eager=True,
        )

        print(f"[OK] {name} initialized successfully!")
        result["init_ok"] = True

        # Cleanup
        del llm
        cleanup_gpu()

    except Exception as e:
        error_msg = str(e)
        # Truncate long errors
        if len(error_msg) > 200:
            error_msg = error_msg[:200] + "..."
        result["error"] = error_msg
        print(f"[FAILED] {name}: {error_msg}")
        cleanup_gpu()

    return result

def main():
    print("="*70)
    print("  vLLM V0 Engine Model Compatibility Test")
    print("="*70)

    # Check vLLM import
    try:
        from vllm import LLM
        print("[OK] vLLM imported successfully")
    except ImportError as e:
        print(f"[FAILED] Cannot import vLLM: {e}")
        sys.exit(1)

    # Test each model
    results = []

    # Allow testing specific model via command line
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        if model_name in VLLM_MODEL_PATHS:
            results.append(test_model_init(model_name, VLLM_MODEL_PATHS[model_name]))
        else:
            print(f"Unknown model: {model_name}")
            print(f"Available: {list(VLLM_MODEL_PATHS.keys())}")
            sys.exit(1)
    else:
        # Test all models
        for name, path in VLLM_MODEL_PATHS.items():
            result = test_model_init(name, path)
            results.append(result)

            # Stop on first success to save time (optional)
            # if result["init_ok"]:
            #     print("\nFound working model, stopping...")
            #     break

    # Summary
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)

    ok_models = [r for r in results if r["init_ok"]]
    failed_models = [r for r in results if not r["init_ok"]]

    print(f"\n[OK] Working models ({len(ok_models)}):")
    for r in ok_models:
        print(f"  - {r['name']}")

    print(f"\n[FAILED] Failed models ({len(failed_models)}):")
    for r in failed_models:
        print(f"  - {r['name']}: {r['error'][:80]}...")

if __name__ == "__main__":
    main()
