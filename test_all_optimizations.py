#!/usr/bin/env python3
"""
Test All GPU Optimizations with Explicit Logging

This script will show explicit traces for all 5 optimizations:
1. Memory Pool Optimization
2. TensorRT Compilation (black-box only)
3. Mixed Precision (AMP)
4. INT8 Quantization
5. Dynamic Batch Sizing
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_white_box_optimizations():
    """Test white-box attack with explicit optimization logging"""
    print("🔬 TESTING WHITE-BOX OPTIMIZATIONS")
    print("=" * 50)

    from attack_models.white_box_universal import run_epsilon_attack

    # This will show:
    # - Memory Pool optimization
    # - Dynamic Batch Sizing
    # - Mixed Precision (AMP)
    # - TensorRT disabled (expected for white-box)

    result = run_epsilon_attack(
        image_path="data/clean/chart/20231107140031466140.png",
        attack_type="fgsm",
        epsilon_target=0.02,
        trial_number=1
    )

    print("✅ White-box test completed")
    return True

def test_black_box_optimizations():
    """Test black-box attack with TensorRT enabled"""
    print("\n🔒 TESTING BLACK-BOX OPTIMIZATIONS (TensorRT)")
    print("=" * 50)

    from attack_models.black_box_universal import UniversalEpsilonBlackBoxAttack

    # This will show:
    # - TensorRT compilation (enabled for black-box)
    # - Memory Pool optimization
    # - Mixed Precision (AMP)

    attack = UniversalEpsilonBlackBoxAttack(epsilon_target=0.02)

    result = attack.run_epsilon_attack(
        image_path="data/clean/chart/20231107140031466140.png",
        attack_type="square",
        attack_params=None
    )

    print("✅ Black-box test completed")
    return True

def test_quantization():
    """Test quantization explicitly"""
    print("\n🔢 TESTING INT8 QUANTIZATION")
    print("=" * 50)

    from attack_models.utils import create_classifier

    # This will show explicit quantization logging
    classifier = create_classifier(
        device='cuda:0',
        requires_grad=False,
        use_quantization=True,
        optimization_level='high'
    )

    print("✅ Quantization test completed")
    return True

def main():
    """Run all optimization tests"""
    print("🚀 COMPREHENSIVE GPU OPTIMIZATION TEST")
    print("=" * 70)

    try:
        # Test 1: White-box (shows 3 optimizations)
        test_white_box_optimizations()

        # Test 2: Black-box (shows TensorRT)
        test_black_box_optimizations()

        # Test 3: Quantization
        test_quantization()

        print("\n" + "=" * 70)
        print("🏆 ALL OPTIMIZATION TESTS COMPLETED")
        print("✅ Check the output above for explicit traces of each optimization")
        print("=" * 70)

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())