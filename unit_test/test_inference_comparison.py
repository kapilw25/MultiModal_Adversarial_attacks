#!/usr/bin/env python3
"""
Unit Test: Sequential vs vLLM Inference Comparison

Purpose: Compare VLM output consistency between:
- model_inference_sequential.py (HuggingFace local_model/ based)
- model_inference_vLLM.py (vLLM continuous batching based)

Tests:
1. Response format consistency (both should produce "The answer is X" format)
2. Text cleaner application verification
3. Prompt format differences (informational)

Usage:
    python unit_test/test_inference_comparison.py [--engine ENGINE_NAME]

    # Test single engine:
    python unit_test/test_inference_comparison.py --engine Qwen25_VL_3B

    # Test all common engines:
    python unit_test/test_inference_comparison.py --all

Note: This test requires GPU and will load models. Run one engine at a time to avoid OOM.
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts', 'utils'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'local_model'))

# Test configuration
TEST_IMAGE_PATH = os.path.join(PROJECT_ROOT, "data/clean/chart/20231106193645289568.png")
TEST_QUESTION = "What is the title of this chart?"

# Common engines available in BOTH sequential and vLLM scripts
COMMON_ENGINES = [
    "Qwen25_VL_3B",
    "Qwen25_VL_7B",
    "Qwen2_VL_2B",
    "LLAVA_1pt5_7B",
    "LLAVA_v1pt6_Mistral_7B",
    "InternVL3_1B",
    "InternVL3_2B",
    "InternVL25_4B",
    "PaliGemma_VL_3B",
]

# Sequential-only engines (not in vLLM)
SEQUENTIAL_ONLY = [
    "Gemma3_VL_4B", "DeepSeek1_VL_1pt3B", "DeepSeek1_VL_7B",
    "SmolVLM2_pt25B", "SmolVLM2_pt5B", "SmolVLM2_2pt2B", "Moondream2_2B"
]

# vLLM-only engines (not in sequential)
VLLM_ONLY = [
    "MiniCPM_V_2pt5", "DeepSeek_VL2_Small", "H2OVL_Mississippi_2B",
    "Phi35_Vision", "Mono_InternVL_2B", "MiniCPM_V_4", "Molmo_7B"
]


def check_dependencies():
    """Check if required dependencies are available"""
    errors = []

    # Check test image exists
    if not os.path.exists(TEST_IMAGE_PATH):
        errors.append(f"Test image not found: {TEST_IMAGE_PATH}")

    # Check vLLM availability
    try:
        from vllm import LLM, SamplingParams
        vllm_available = True
    except ImportError:
        vllm_available = False
        errors.append("vLLM not installed. Install with: pip install vllm>=0.6.0")

    # Check torch/CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA not available. GPU required for inference.")
    except ImportError:
        errors.append("PyTorch not installed")

    return errors, vllm_available


def run_sequential_inference(engine: str, image_path: str, question: str) -> dict:
    """
    Run inference using sequential script (vlm_local_client)

    Returns dict with: response, raw_response, inference_time, error
    """
    result = {
        "method": "sequential",
        "engine": engine,
        "response": None,
        "raw_response": None,
        "inference_time_seconds": 0.0,
        "error": None,
        "prompt_used": None
    }

    try:
        import base64
        from mimetypes import guess_type
        from utils.vlm_local_client import send_chat_request, unload_model

        # Convert image to data URL (same as model_inference_sequential.py)
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'
        with open(image_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode('utf-8')
        image_url = f"data:{mime_type};base64,{base64_data}"

        # Prepare message (same format as sequential script)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }]

        result["prompt_used"] = f"[Sequential] Message with text: '{question}' + image"

        # Run inference
        start_time = time.time()
        response, _, metrics = send_chat_request(
            message_text=messages,
            engine=engine,
            temp=0.2,
            max_new_token=512,
            sample_n=1,
            collect_metrics=True
        )
        result["inference_time_seconds"] = time.time() - start_time
        result["response"] = response

        # Note: vlm_local_client applies clean_vlm_response internally
        # So response is already cleaned

        # Unload model to free GPU memory
        unload_model(engine)

    except Exception as e:
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()

    return result


def run_vllm_inference(engine: str, image_path: str, question: str) -> dict:
    """
    Run inference using vLLM script

    Returns dict with: response, raw_response, inference_time, error
    """
    result = {
        "method": "vllm",
        "engine": engine,
        "response": None,
        "raw_response": None,
        "inference_time_seconds": 0.0,
        "error": None,
        "prompt_used": None
    }

    try:
        from vllm import LLM, SamplingParams
        from utils.text_cleaner import clean_vlm_response
        import torch
        import gc

        # Import vLLM script components
        sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts'))
        from model_inference_vLLM import (
            VLLM_MODEL_PATHS,
            format_prompt_for_vllm,
            get_sampling_params_for_engine
        )

        if engine not in VLLM_MODEL_PATHS:
            result["error"] = f"Engine {engine} not in vLLM registry"
            return result

        model_path = VLLM_MODEL_PATHS[engine]

        # Format prompt (model-specific)
        formatted_prompt = format_prompt_for_vllm(engine, question)
        result["prompt_used"] = f"[vLLM] {formatted_prompt}"

        # Prepare vLLM input
        prompts = [{
            "prompt": formatted_prompt,
            "multi_modal_data": {"image": image_path}
        }]

        # Get model-specific sampling params
        sampling_params = get_sampling_params_for_engine(engine)

        # Initialize vLLM
        print(f"    Loading vLLM model: {model_path}")
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.85,
            max_model_len=4096,
            dtype="half",
            limit_mm_per_prompt={"image": 1}
        )

        # Run inference
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        result["inference_time_seconds"] = time.time() - start_time

        # Extract response
        raw_response = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        result["raw_response"] = raw_response

        # Apply text cleaner (same as vLLM script line 477)
        result["response"] = clean_vlm_response(raw_response)

        # Cleanup
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

    except Exception as e:
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()

    return result


def compare_results(sequential_result: dict, vllm_result: dict) -> dict:
    """Compare sequential vs vLLM results"""
    comparison = {
        "engine": sequential_result["engine"],
        "sequential_response": sequential_result["response"],
        "vllm_response": vllm_result["response"],
        "sequential_error": sequential_result["error"],
        "vllm_error": vllm_result["error"],
        "checks": {}
    }

    # Check 1: Both responses start with "The answer is"
    seq_resp = sequential_result["response"] or ""
    vllm_resp = vllm_result["response"] or ""

    comparison["checks"]["sequential_format_correct"] = seq_resp.lower().startswith("the answer is")
    comparison["checks"]["vllm_format_correct"] = vllm_resp.lower().startswith("the answer is")
    comparison["checks"]["both_format_correct"] = (
        comparison["checks"]["sequential_format_correct"] and
        comparison["checks"]["vllm_format_correct"]
    )

    # Check 2: Neither has error
    comparison["checks"]["sequential_no_error"] = sequential_result["error"] is None
    comparison["checks"]["vllm_no_error"] = vllm_result["error"] is None
    comparison["checks"]["both_no_error"] = (
        comparison["checks"]["sequential_no_error"] and
        comparison["checks"]["vllm_no_error"]
    )

    # Check 3: Response length reasonable (not empty, not too long)
    comparison["checks"]["sequential_length_ok"] = 5 < len(seq_resp) < 500
    comparison["checks"]["vllm_length_ok"] = 5 < len(vllm_resp) < 500

    # Check 4: Response similarity (informational only)
    if seq_resp and vllm_resp:
        # Extract answer part after "The answer is"
        seq_answer = seq_resp.lower().replace("the answer is ", "").strip().rstrip(".")
        vllm_answer = vllm_resp.lower().replace("the answer is ", "").strip().rstrip(".")
        comparison["checks"]["answers_similar"] = seq_answer == vllm_answer
        comparison["sequential_answer_extracted"] = seq_answer
        comparison["vllm_answer_extracted"] = vllm_answer
    else:
        comparison["checks"]["answers_similar"] = False

    # Overall pass/fail
    comparison["passed"] = (
        comparison["checks"]["both_format_correct"] and
        comparison["checks"]["both_no_error"]
    )

    return comparison


def run_test_for_engine(engine: str, verbose: bool = True) -> dict:
    """Run comparison test for a single engine (common engines only)"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing: {engine}")
        print(f"{'='*60}")
        print(f"  Image: {TEST_IMAGE_PATH}")
        print(f"  Question: {TEST_QUESTION}")

    # Run sequential inference
    if verbose:
        print(f"\n  [1/2] Running Sequential inference...")
    seq_result = run_sequential_inference(engine, TEST_IMAGE_PATH, TEST_QUESTION)
    if verbose:
        if seq_result["error"]:
            print(f"    ERROR: {seq_result['error']}")
        else:
            print(f"    Response: {seq_result['response'][:80]}...")
            print(f"    Time: {seq_result['inference_time_seconds']:.2f}s")

    # Run vLLM inference
    if verbose:
        print(f"\n  [2/2] Running vLLM inference...")
    vllm_result = run_vllm_inference(engine, TEST_IMAGE_PATH, TEST_QUESTION)
    if verbose:
        if vllm_result["error"]:
            print(f"    ERROR: {vllm_result['error']}")
        else:
            print(f"    Response: {vllm_result['response'][:80]}...")
            print(f"    Time: {vllm_result['inference_time_seconds']:.2f}s")

    # Compare results
    comparison = compare_results(seq_result, vllm_result)

    if verbose:
        print(f"\n  Comparison Results:")
        print(f"    Format check (seq): {'PASS' if comparison['checks']['sequential_format_correct'] else 'FAIL'}")
        print(f"    Format check (vllm): {'PASS' if comparison['checks']['vllm_format_correct'] else 'FAIL'}")
        print(f"    No errors (seq): {'PASS' if comparison['checks']['sequential_no_error'] else 'FAIL'}")
        print(f"    No errors (vllm): {'PASS' if comparison['checks']['vllm_no_error'] else 'FAIL'}")
        print(f"    Answers similar: {'YES' if comparison['checks'].get('answers_similar') else 'NO'}")
        print(f"    OVERALL: {'PASS' if comparison['passed'] else 'FAIL'}")

    # Add full results for debugging
    comparison["sequential_full"] = seq_result
    comparison["vllm_full"] = vllm_result

    return comparison


def run_vllm_only_test(engine: str, verbose: bool = True) -> dict:
    """Run vLLM-only test (no sequential comparison available)"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing (vLLM-only): {engine}")
        print(f"{'='*60}")
        print(f"  Image: {TEST_IMAGE_PATH}")
        print(f"  Question: {TEST_QUESTION}")
        print(f"  Note: No sequential equivalent for comparison")

    # Run vLLM inference only
    if verbose:
        print(f"\n  Running vLLM inference...")
    vllm_result = run_vllm_inference(engine, TEST_IMAGE_PATH, TEST_QUESTION)

    response = vllm_result["response"] or ""

    result = {
        "engine": engine,
        "vllm_response": vllm_result["response"],
        "vllm_error": vllm_result["error"],
        "checks": {
            "vllm_format_correct": response.lower().startswith("the answer is"),
            "vllm_no_error": vllm_result["error"] is None,
            "vllm_length_ok": 5 < len(response) < 500
        },
        "vllm_full": vllm_result
    }

    result["passed"] = result["checks"]["vllm_format_correct"] and result["checks"]["vllm_no_error"]

    if verbose:
        if vllm_result["error"]:
            print(f"    ERROR: {vllm_result['error']}")
        else:
            print(f"    Response: {response[:80]}...")
            print(f"    Time: {vllm_result['inference_time_seconds']:.2f}s")
        print(f"\n  Results:")
        print(f"    Format check: {'PASS' if result['checks']['vllm_format_correct'] else 'FAIL'}")
        print(f"    No errors: {'PASS' if result['checks']['vllm_no_error'] else 'FAIL'}")
        print(f"    OVERALL: {'PASS' if result['passed'] else 'FAIL'}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Compare Sequential vs vLLM inference")
    parser.add_argument("--engine", type=str, help="Specific engine to test")
    parser.add_argument("--all", action="store_true", help="Test all common engines")
    parser.add_argument("--vllm-only", action="store_true", help="Test vLLM-only engines (no comparison)")
    parser.add_argument("--list", action="store_true", help="List available engines")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    print("="*70)
    print("  UNIT TEST: Sequential vs vLLM Inference Comparison")
    print("="*70)

    if args.list:
        print("\nCommon Engines (--engine or --all):")
        for e in COMMON_ENGINES:
            print(f"  - {e}")
        print("\nSequential-Only Engines (not testable here):")
        for e in SEQUENTIAL_ONLY:
            print(f"  - {e}")
        print("\nvLLM-Only Engines (--vllm-only):")
        for e in VLLM_ONLY:
            print(f"  - {e}")
        return

    # Check dependencies
    errors, vllm_available = check_dependencies()
    if errors:
        print("\nDependency Errors:")
        for err in errors:
            print(f"  - {err}")
        if not vllm_available:
            print("\nCannot run tests without vLLM.")
            return

    # Handle --vllm-only flag
    if args.vllm_only:
        if args.engine:
            if args.engine not in VLLM_ONLY:
                print(f"\nError: {args.engine} not in vLLM-only engines.")
                print(f"Available: {VLLM_ONLY}")
                return
            engines_to_test = [args.engine]
        else:
            engines_to_test = VLLM_ONLY

        print(f"\n  Mode: vLLM-only (no sequential comparison)")

        all_results = {}
        passed = 0
        failed = 0

        for engine in engines_to_test:
            result = run_vllm_only_test(engine)
            all_results[engine] = result
            if result["passed"]:
                passed += 1
            else:
                failed += 1

        # Summary
        print("\n" + "="*70)
        print("  TEST SUMMARY (vLLM-only)")
        print("="*70)
        print(f"  Total engines tested: {len(engines_to_test)}")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")

        for engine, result in all_results.items():
            status = "PASS" if result["passed"] else "FAIL"
            print(f"    {engine}: {status}")

        sys.exit(0 if failed == 0 else 1)

    # Regular mode: test common engines with comparison
    if args.engine:
        if args.engine not in COMMON_ENGINES:
            print(f"\nError: {args.engine} not in common engines.")
            print(f"Use --vllm-only --engine {args.engine} for vLLM-only engines.")
            print(f"Available common: {COMMON_ENGINES}")
            return
        engines_to_test = [args.engine]
    elif args.all:
        engines_to_test = COMMON_ENGINES
    else:
        # Default: test first common engine only
        engines_to_test = [COMMON_ENGINES[0]]
        print(f"\nNo engine specified. Testing default: {engines_to_test[0]}")
        print("Use --all to test all common engines, or --engine NAME for specific engine")
        print("Use --vllm-only to test vLLM-only engines")

    # Run tests
    all_results = {}
    passed = 0
    failed = 0

    for engine in engines_to_test:
        result = run_test_for_engine(engine)
        all_results[engine] = result
        if result["passed"]:
            passed += 1
        else:
            failed += 1

    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    print(f"  Total engines tested: {len(engines_to_test)}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    for engine, result in all_results.items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"    {engine}: {status}")

    # Save results if requested
    if args.output:
        output_path = os.path.join(PROJECT_ROOT, args.output)
        with open(output_path, "w") as f:
            # Convert to serializable format
            serializable_results = {
                "timestamp": datetime.now().isoformat(),
                "test_image": TEST_IMAGE_PATH,
                "test_question": TEST_QUESTION,
                "summary": {
                    "total": len(engines_to_test),
                    "passed": passed,
                    "failed": failed
                },
                "results": {}
            }
            for engine, result in all_results.items():
                serializable_results["results"][engine] = {
                    "passed": result["passed"],
                    "sequential_response": result["sequential_response"],
                    "vllm_response": result["vllm_response"],
                    "checks": result["checks"],
                    "errors": {
                        "sequential": result["sequential_error"],
                        "vllm": result["vllm_error"]
                    }
                }
            json.dump(serializable_results, f, indent=2)
        print(f"\n  Results saved to: {output_path}")

    # Exit code
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
