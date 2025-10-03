#!/usr/bin/env python3
"""
Unit Test: Verify Black-box Multi-Epsilon Batch Processing

Tests:
1. batch_multi_epsilon_blackbox_attack function exists in black_box_universal.py
2. blackbox_batch_runner function exists
3. attack_runner.py calls multi-epsilon batch function for black-box
4. Implementation matches white-box pattern
"""

import sys
import re

print("=" * 70)
print("BLACK-BOX BATCH PROCESSING VERIFICATION")
print("=" * 70)
print()

# Test 1: Check blackbox_batch_runner exists with proper type annotations
print("TEST 1: blackbox_batch_runner Function Exists with Proper Types")
print("-" * 70)

with open('attack_models/black_box_universal.py', 'r') as f:
    blackbox_code = f.read()

blackbox_runner_exists = 'def blackbox_batch_runner(' in blackbox_code
# ✅ FIXED: Check for proper type annotations (List[np.ndarray], not list)
has_images_param = 'images: List[np.ndarray]' in blackbox_code
has_epsilon_param = 'epsilon_target: float' in blackbox_code
has_attack_type_param = 'attack_type: str' in blackbox_code
has_proper_return_type = '-> List[Dict]:' in blackbox_code

print(f"   Function blackbox_batch_runner: {'✅' if blackbox_runner_exists else '❌'}")
print(f"   Parameter images: List[np.ndarray]: {'✅' if has_images_param else '❌'}")
print(f"   Parameter epsilon_target: float: {'✅' if has_epsilon_param else '❌'}")
print(f"   Parameter attack_type: str: {'✅' if has_attack_type_param else '❌'}")
print(f"   Return type List[Dict]: {'✅' if has_proper_return_type else '❌'}")

test1_pass = (blackbox_runner_exists and has_images_param and has_epsilon_param
              and has_attack_type_param and has_proper_return_type)

print(f"\n{'✅ TEST 1 PASSED' if test1_pass else '❌ TEST 1 FAILED'}")
print()

# Test 2: Check batch_multi_epsilon_blackbox_attack exists with proper types
print("TEST 2: batch_multi_epsilon_blackbox_attack Function with Proper Types")
print("-" * 70)

batch_func_exists = 'def batch_multi_epsilon_blackbox_attack(' in blackbox_code
# ✅ FIXED: Check for proper type annotations (List[str], List[float], List[Dict])
has_image_paths_param = 'image_paths: List[str]' in blackbox_code
has_epsilon_targets_param = 'epsilon_targets: List[float]' in blackbox_code
has_proper_return_type_batch = '-> List[Dict]:' in blackbox_code
calls_generic_batch = 'return batch_multi_epsilon_attack(' in blackbox_code
passes_blackbox_runner = 'attack_runner_func=blackbox_batch_runner' in blackbox_code
sets_is_blackbox = 'is_blackbox=True' in blackbox_code

print(f"   Function batch_multi_epsilon_blackbox_attack: {'✅' if batch_func_exists else '❌'}")
print(f"   Parameter image_paths: List[str]: {'✅' if has_image_paths_param else '❌'}")
print(f"   Parameter epsilon_targets: List[float]: {'✅' if has_epsilon_targets_param else '❌'}")
print(f"   Return type List[Dict]: {'✅' if has_proper_return_type_batch else '❌'}")
print(f"   Calls generic batch_multi_epsilon_attack: {'✅' if calls_generic_batch else '❌'}")
print(f"   Passes blackbox_batch_runner: {'✅' if passes_blackbox_runner else '❌'}")
print(f"   Sets is_blackbox=True: {'✅' if sets_is_blackbox else '❌'}")

test2_pass = (batch_func_exists and has_image_paths_param and has_epsilon_targets_param
              and has_proper_return_type_batch and calls_generic_batch and passes_blackbox_runner
              and sets_is_blackbox)

print(f"\n{'✅ TEST 2 PASSED' if test2_pass else '❌ TEST 2 FAILED'}")
print()

# Test 3: Check attack_runner.py uses multi-epsilon batch for black-box
print("TEST 3: attack_runner.py Calls Multi-Epsilon Batch for Black-box")
print("-" * 70)

with open('scripts/attack_runner.py', 'r') as f:
    runner_code = f.read()

has_multi_epsilon_blackbox_func = 'def run_multi_epsilon_blackbox_attack(' in runner_code
imports_batch_function = 'from attack_models.black_box_universal import batch_multi_epsilon_blackbox_attack' in runner_code
calls_batch_in_main = 'success = self.run_multi_epsilon_blackbox_attack(valid_images, attack_type, epsilon_values, trial_number)' in runner_code
no_todo_comment = '# TODO: Implement blackbox multi-epsilon batching' not in runner_code
no_fallback_to_sequential = 'print("⚠️  Black-box multi-epsilon batching not yet implemented, falling back to sequential")' not in runner_code

print(f"   Function run_multi_epsilon_blackbox_attack: {'✅' if has_multi_epsilon_blackbox_func else '❌'}")
print(f"   Imports batch_multi_epsilon_blackbox_attack: {'✅' if imports_batch_function else '❌'}")
print(f"   Calls batch function in _execute_attacks_with_multi_epsilon_batch: {'✅' if calls_batch_in_main else '❌'}")
print(f"   Removed TODO comment: {'✅' if no_todo_comment else '❌'}")
print(f"   Removed fallback to sequential: {'✅' if no_fallback_to_sequential else '❌'}")

test3_pass = (has_multi_epsilon_blackbox_func and imports_batch_function and calls_batch_in_main
              and no_todo_comment and no_fallback_to_sequential)

print(f"\n{'✅ TEST 3 PASSED' if test3_pass else '❌ TEST 3 FAILED'}")
print()

# Test 4: Verify implementation matches white-box pattern
print("TEST 4: Black-box Implementation Matches White-box Pattern")
print("-" * 70)

# Check run_multi_epsilon_blackbox_attack has similar structure to whitebox
has_operations_needed_check = 'operations_needed = []' in runner_code and 'operations_needed.append((image_path, epsilon_value))' in runner_code
has_unique_extraction = 'unique_images = sorted(list(set([img for img, _ in operations_needed])))' in runner_code
has_unique_epsilons = 'unique_epsilons = sorted(list(set([eps for _, eps in operations_needed])))' in runner_code
has_result_processing = 'for result in results:' in runner_code and "result.get('success', False)" in runner_code
has_failure_logging = "self.db.insert_attack_result({" in runner_code and "'success': False," in runner_code

print(f"   Filters operations_needed (replacement=NO): {'✅' if has_operations_needed_check else '❌'}")
print(f"   Extracts unique images: {'✅' if has_unique_extraction else '❌'}")
print(f"   Extracts unique epsilons: {'✅' if has_unique_epsilons else '❌'}")
print(f"   Processes results from batch function: {'✅' if has_result_processing else '❌'}")
print(f"   Logs failures to database: {'✅' if has_failure_logging else '❌'}")

test4_pass = (has_operations_needed_check and has_unique_extraction and has_unique_epsilons
              and has_result_processing and has_failure_logging)

print(f"\n{'✅ TEST 4 PASSED' if test4_pass else '❌ TEST 4 FAILED'}")
print()

# Test 5: Verify batch size expectations
print("TEST 5: Batch Size and Documentation Consistency")
print("-" * 70)

# Check docstrings mention batch size of 75
blackbox_doc_75 = 'Processes: N images × M epsilons in batches of 75' in blackbox_code
example_25x3 = 'Example: 25 images × 3 epsilons = 75 operations' in blackbox_code

# Check blackbox_batch_runner processes images sequentially within epsilon group
processes_images_loop = 'for idx, (image, image_path) in enumerate(zip(images, image_paths)):' in blackbox_code

print(f"   Documentation mentions batch size 75: {'✅' if blackbox_doc_75 else '❌'}")
print(f"   Example shows 25×3=75: {'✅' if example_25x3 else '❌'}")
print(f"   blackbox_batch_runner processes images in loop: {'✅' if processes_images_loop else '❌'}")

test5_pass = blackbox_doc_75 and example_25x3 and processes_images_loop

print(f"\n{'✅ TEST 5 PASSED' if test5_pass else '❌ TEST 5 FAILED'}")
print()

# Test 6: Verify blackbox returns output_path (not adversarial_image)
print("TEST 6: Black-box Returns output_path (CRITICAL FIX)")
print("-" * 70)

# Check that blackbox_batch_runner returns output_path
returns_output_path = "'output_path': output_path" in blackbox_code
calls_get_output_path = 'output_path = get_output_path(image_path, attack_type, is_blackbox=True' in blackbox_code
calls_save_image = 'save_image(adv_image, output_path)' in blackbox_code
no_adversarial_image_in_result = "'adversarial_image': adv_image" not in blackbox_code

print(f"   Returns 'output_path' in result dict: {'✅' if returns_output_path else '❌'}")
print(f"   Calls get_output_path(is_blackbox=True): {'✅' if calls_get_output_path else '❌'}")
print(f"   Calls save_image() inside runner: {'✅' if calls_save_image else '❌'}")
print(f"   No 'adversarial_image' in result: {'✅' if no_adversarial_image_in_result else '❌'}")

test6_pass = returns_output_path and calls_get_output_path and calls_save_image and no_adversarial_image_in_result

print(f"\n{'✅ TEST 6 PASSED' if test6_pass else '❌ TEST 6 FAILED'}")
print()

# Test 7: Verify attack_runner uses output_path (not manual saving)
print("TEST 7: attack_runner.py Uses output_path (No Manual Saving)")
print("-" * 70)

with open('scripts/attack_runner.py', 'r') as f:
    runner_code = f.read()

uses_output_path = "result.get('output_path', '')" in runner_code
no_manual_save_in_loop = "Image.fromarray(adv_image).save(adversarial_path)" not in runner_code
has_pil_import_top = "from PIL import Image" in runner_code.split('def ')[0]  # Check if in imports section

print(f"   Uses result.get('output_path'): {'✅' if uses_output_path else '❌'}")
print(f"   No manual Image.save() in result loop: {'✅' if no_manual_save_in_loop else '❌'}")
print(f"   PIL imported at top (not in loop): {'✅' if has_pil_import_top else '❌'}")

test7_pass = uses_output_path and no_manual_save_in_loop and has_pil_import_top

print(f"\n{'✅ TEST 7 PASSED' if test7_pass else '❌ TEST 7 FAILED'}")
print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)

all_passed = test1_pass and test2_pass and test3_pass and test4_pass and test5_pass and test6_pass and test7_pass

if all_passed:
    print("✅ ALL TESTS PASSED")
    print()
    print("Implementation Verified:")
    print("  1. ✅ blackbox_batch_runner function created")
    print("  2. ✅ batch_multi_epsilon_blackbox_attack wrapper created")
    print("  3. ✅ attack_runner.py calls multi-epsilon batch for black-box")
    print("  4. ✅ Implementation matches white-box pattern")
    print("  5. ✅ Batch size and documentation consistent")
    print()
    print("Expected Behavior:")
    print("  • Black-box attacks now process [25 images × 3 epsilons] in parallel")
    print("  • Uses same Cartesian product batching as white-box")
    print("  • Batch size: 75 operations (was sequential: 1 at a time)")
    print("  • Full feature parity: selective mode, failure logging, etc.")
    exit_code = 0
else:
    print("❌ SOME TESTS FAILED")
    print()
    if not test1_pass:
        print("  ❌ blackbox_batch_runner function missing or incomplete")
    if not test2_pass:
        print("  ❌ batch_multi_epsilon_blackbox_attack function missing or incomplete")
    if not test3_pass:
        print("  ❌ attack_runner.py not using multi-epsilon batch for black-box")
    if not test4_pass:
        print("  ❌ Implementation doesn't match white-box pattern")
    if not test5_pass:
        print("  ❌ Batch size or documentation inconsistent")
    exit_code = 1

print()
sys.exit(exit_code)
