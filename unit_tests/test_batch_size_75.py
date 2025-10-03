#!/usr/bin/env python3
"""
Unit Test: Verify Batch Size Increased to 75 and True Multi-Epsilon Batching

Tests:
1. Batch size configuration is 75 (was 64)
2. Multi-epsilon batching mixes images AND epsilons (not separate per-epsilon batches)
3. TensorRT max_batch_size is 75
"""

import sys
import re

print("=" * 70)
print("BATCH SIZE 75 & TRUE MULTI-EPSILON BATCHING VERIFICATION")
print("=" * 70)
print()

# Test 1: Check get_optimal_batch_size returns 75
print("TEST 1: Optimal Batch Size = 75")
print("-" * 70)

with open('attack_models/utils.py', 'r') as f:
    utils_code = f.read()

# Find the get_optimal_batch_size function
batch_size_match = re.search(r'if free_memory_gb >= 20:\s+optimal_batch = (\d+)', utils_code)

if batch_size_match:
    batch_size = int(batch_size_match.group(1))
    print(f"   Found: optimal_batch = {batch_size} (when free_memory_gb >= 20)")

    if batch_size == 75:
        print("   ✅ Batch size is 75 (correct)")
        test1_pass = True
    else:
        print(f"   ❌ Batch size is {batch_size}, expected 75")
        test1_pass = False
else:
    print("   ❌ Could not find batch size configuration")
    test1_pass = False

print(f"\n{'✅ TEST 1 PASSED' if test1_pass else '❌ TEST 1 FAILED'}")
print()

# Test 2: Check TensorRT max_batch_size is 75
print("TEST 2: TensorRT max_batch_size = 75")
print("-" * 70)

tensorrt_match = re.search(r'def create_tensorrt_classifier.*?max_batch_size=(\d+)', utils_code, re.DOTALL)

if tensorrt_match:
    tensorrt_batch = int(tensorrt_match.group(1))
    print(f"   Found: max_batch_size={tensorrt_batch} in create_tensorrt_classifier()")

    if tensorrt_batch == 75:
        print("   ✅ TensorRT max_batch_size is 75 (correct)")
        test2_pass = True
    else:
        print(f"   ❌ TensorRT max_batch_size is {tensorrt_batch}, expected 75")
        test2_pass = False
else:
    print("   ❌ Could not find TensorRT max_batch_size configuration")
    test2_pass = False

print(f"\n{'✅ TEST 2 PASSED' if test2_pass else '❌ TEST 2 FAILED'}")
print()

# Test 3: Check multi-epsilon batching logic
print("TEST 3: True Multi-Epsilon Batching (Not Split by Epsilon)")
print("-" * 70)

with open('scripts/attack_runner.py', 'r') as f:
    runner_code = f.read()

# Check that we're NOT splitting by epsilon
has_old_split_logic = "for eps_val, img_paths in epsilon_to_images.items():" in runner_code
has_new_unified_logic = "unique_images = sorted(list(set([img for img, _ in operations_needed])))" in runner_code
has_unified_call = "batch_multi_epsilon_whitebox_attack(\n                    image_paths=unique_images," in runner_code

print(f"   Old split-by-epsilon logic removed: {'✅ YES' if not has_old_split_logic else '❌ NO (still present!)'}")
print(f"   New unified batching logic: {'✅ YES' if has_new_unified_logic else '❌ NO'}")
print(f"   Calls batch function with unique_images: {'✅ YES' if has_unified_call else '❌ NO'}")

test3_pass = (not has_old_split_logic) and has_new_unified_logic and has_unified_call

if test3_pass:
    print("\n   ✅ Multi-epsilon batching correctly mixes images AND epsilons")
else:
    print("\n   ❌ Multi-epsilon batching still splits by epsilon (wrong!)")

print(f"\n{'✅ TEST 3 PASSED' if test3_pass else '❌ TEST 3 FAILED'}")
print()

# Test 4: Verify all batch size tiers scaled
print("TEST 4: All Batch Size Tiers Scaled Proportionally")
print("-" * 70)

expected_tiers = {
    20: 75,  # 20GB+ → 75
    15: 56,  # 15GB+ → 56
    10: 38,  # 10GB+ → 38
    6: 19,   # 6GB+ → 19
}

all_tiers_correct = True
for memory_gb, expected_batch in expected_tiers.items():
    pattern = rf'if free_memory_gb >= {memory_gb}:\s+optimal_batch = (\d+)'
    match = re.search(pattern, utils_code)

    if match:
        actual_batch = int(match.group(1))
        is_correct = actual_batch == expected_batch
        status = "✅" if is_correct else "❌"
        print(f"   {status} {memory_gb}GB+ tier: {actual_batch} (expected {expected_batch})")

        if not is_correct:
            all_tiers_correct = False
    else:
        print(f"   ❌ {memory_gb}GB+ tier: NOT FOUND")
        all_tiers_correct = False

test4_pass = all_tiers_correct

print(f"\n{'✅ TEST 4 PASSED' if test4_pass else '❌ TEST 4 FAILED'}")
print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)

all_passed = test1_pass and test2_pass and test3_pass and test4_pass

if all_passed:
    print("✅ ALL TESTS PASSED")
    print()
    print("Changes Verified:")
    print("  1. ✅ Optimal batch size increased to 75")
    print("  2. ✅ TensorRT max_batch_size increased to 75")
    print("  3. ✅ Multi-epsilon batching mixes images AND epsilons (true batching)")
    print("  4. ✅ All memory tiers scaled proportionally")
    print()
    print("Expected Behavior:")
    print("  • 25 images × 3 epsilons = 75 operations")
    print("  • Will process in 1 batch of 75 (not 3 batches of 22+21+21)")
    print("  • Maximum GPU utilization with mixed operations")
    exit_code = 0
else:
    print("❌ SOME TESTS FAILED")
    print()
    if not test1_pass:
        print("  ❌ Batch size not 75")
    if not test2_pass:
        print("  ❌ TensorRT max_batch_size not 75")
    if not test3_pass:
        print("  ❌ Multi-epsilon batching still splits by epsilon")
    if not test4_pass:
        print("  ❌ Batch size tiers not scaled correctly")
    exit_code = 1

print()
sys.exit(exit_code)
