#!/usr/bin/env python3
"""
Quick TensorRT Fix Validation Test

Tests the specific fix for the RuntimeError:
"Expected compiled_engine->exec_ctx->setInputShape(name.c_str(), dims) to be true but got false"
"""

import sys
import os
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from attack_models.utils import (
        create_tensorrt_classifier,
        create_optimized_classifier,
        TENSORRT_AVAILABLE
    )
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

def test_tensorrt_dynamic_batch():
    """Test TensorRT with dynamic batch sizes"""
    print("🔧 Testing TensorRT Dynamic Batch Fix")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    if not TENSORRT_AVAILABLE:
        print("❌ TensorRT not available")
        return False

    try:
        # Create TensorRT classifier with dynamic batch support
        print("📦 Creating TensorRT classifier...")
        classifier = create_tensorrt_classifier(
            device='cuda:0',
            requires_grad=False,
            probabilistic=True,
            count_queries=False,
            max_batch_size=4  # Test with max_batch_size=4
        )
        print("✅ TensorRT classifier created successfully")

        # Test different batch sizes
        test_cases = [
            ("Batch size 1", torch.rand(1, 3, 224, 224, device='cuda:0')),
            ("Batch size 2", torch.rand(2, 3, 224, 224, device='cuda:0')),  # This was failing before
            ("Batch size 4", torch.rand(4, 3, 224, 224, device='cuda:0')),  # Max batch size
        ]

        for test_name, test_tensor in test_cases:
            print(f"\n🧪 Testing: {test_name} - Shape: {test_tensor.shape}")
            try:
                # Convert to numpy for ART classifier
                test_numpy = test_tensor.cpu().numpy()

                # Run inference
                predictions = classifier.predict(test_numpy)
                print(f"   ✅ Success! Output shape: {predictions.shape}")

                # Verify output shape matches input batch
                expected_batch = test_tensor.shape[0]
                actual_batch = predictions.shape[0]
                if expected_batch == actual_batch:
                    print(f"   ✅ Batch consistency: {expected_batch} == {actual_batch}")
                else:
                    print(f"   ❌ Batch mismatch: expected {expected_batch}, got {actual_batch}")
                    return False

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                return False

        # Test batch size exceeding max (should automatically chunk)
        print(f"\n🧪 Testing: Batch size 8 (exceeds max_batch_size=4)")
        try:
            large_tensor = torch.rand(8, 3, 224, 224, device='cuda:0')
            large_numpy = large_tensor.cpu().numpy()
            predictions = classifier.predict(large_numpy)
            print(f"   ✅ Large batch handled! Output shape: {predictions.shape}")

            if predictions.shape[0] == 8:
                print(f"   ✅ Chunking worked correctly: 8 inputs -> 8 outputs")
            else:
                print(f"   ❌ Chunking failed: expected 8 outputs, got {predictions.shape[0]}")
                return False

        except Exception as e:
            print(f"   ❌ Large batch failed: {e}")
            return False

        print(f"\n🎉 ALL TESTS PASSED!")
        print("✅ TensorRT dynamic batch fix is working correctly")
        return True

    except Exception as e:
        print(f"❌ TensorRT test failed: {e}")
        return False

def main():
    """Run the TensorRT fix validation"""
    success = test_tensorrt_dynamic_batch()

    if success:
        print("\n🏆 CONCLUSION: TensorRT fix successfully resolves the shape error!")
        print("   - Dynamic batch sizes (1-4) work correctly")
        print("   - Automatic chunking handles larger batches")
        print("   - No more 'setInputShape' runtime errors")
        return 0
    else:
        print("\n❌ CONCLUSION: TensorRT fix needs more work")
        return 1

if __name__ == "__main__":
    exit(main())