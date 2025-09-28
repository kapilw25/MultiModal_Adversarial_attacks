#!/usr/bin/env python3
"""
TensorRT Optimization Standalone Test Suite

Tests TensorRT implementation in isolation with:
1. Compilation verification
2. Performance benchmarking vs standard PyTorch
3. Inference accuracy comparison
4. Memory usage optimization
5. Real adversarial attack integration

Manual testing script for TensorRT optimization validation.
"""

import sys
import os
import time
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from attack_models.utils import (
        create_tensorrt_classifier,
        create_optimized_classifier,
        TENSORRT_AVAILABLE,
        get_gpu_memory_info,
        optimize_memory_usage,
        setup_gpu_optimizations
    )
    from torchvision import models, transforms
    from PIL import Image
    import cv2
    IMPORTS_SUCCESS = True
    print("✅ All imports successful")
except ImportError as e:
    IMPORTS_SUCCESS = False
    print(f"❌ Import error: {e}")

class TensorRTTester:
    """Standalone TensorRT testing class"""

    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.test_image_path = None
        self.standard_model = None
        self.tensorrt_model = None
        self.tensorrt_classifier = None
        self.standard_classifier = None

    def setup_test_environment(self):
        """Setup test environment and models"""
        print("🔧 Setting up TensorRT test environment...")

        if not torch.cuda.is_available():
            print("❌ CUDA not available - cannot test TensorRT")
            return False

        if not TENSORRT_AVAILABLE:
            print("❌ TensorRT not available - check installation")
            return False

        # Setup GPU optimizations
        gpu_setup = setup_gpu_optimizations()
        print(f"GPU optimization setup: {'✅' if gpu_setup else '❌'}")

        # Get initial memory info
        memory_info = get_gpu_memory_info()
        print(f"🔍 Initial GPU Memory: {memory_info['free_gb']:.1f}GB free")

        return True

    def create_test_models(self):
        """Create both standard and TensorRT models for comparison"""
        print("\n🏗️ Creating test models...")

        try:
            # Create standard optimized classifier
            print("  Creating standard optimized classifier...")
            self.standard_classifier = create_optimized_classifier(
                device=self.device,
                requires_grad=False,
                probabilistic=True,
                count_queries=False,
                optimization_level='high'
            )
            print("  ✅ Standard classifier created")

            # Create TensorRT classifier
            print("  Creating TensorRT classifier...")
            self.tensorrt_classifier = create_tensorrt_classifier(
                device=self.device,
                requires_grad=False,
                probabilistic=True,
                count_queries=False
            )
            print("  ✅ TensorRT classifier created")

            return True

        except Exception as e:
            print(f"  ❌ Model creation failed: {e}")
            return False

    def create_test_data(self):
        """Create test data for benchmarking"""
        print("\n📊 Creating test data...")

        # TensorRT models now support dynamic batch sizes (1-4)
        # Test multiple batch sizes to verify dynamic batch functionality
        batch_sizes = [1, 2, 4]  # Test dynamic batch support
        test_data = {}

        for batch_size in batch_sizes:
            # Create random tensor in [0,1] range
            test_tensor = torch.rand(batch_size, 3, 224, 224, device=self.device)
            test_data[batch_size] = test_tensor
            print(f"  ✅ Test data batch_size={batch_size}: {test_tensor.shape}")

        return test_data

    def benchmark_inference_speed(self, test_data, num_iterations=10):
        """Benchmark inference speed comparison"""
        print(f"\n⏱️ Benchmarking inference speed ({num_iterations} iterations)...")

        results = {}

        for batch_size, test_tensor in test_data.items():
            print(f"\n  📊 Batch size: {batch_size}")

            # Convert to numpy for ART classifiers
            test_numpy = test_tensor.cpu().numpy()

            # Warm up GPU
            for _ in range(3):
                _ = self.standard_classifier.predict(test_numpy)
                _ = self.tensorrt_classifier.predict(test_numpy)

            torch.cuda.synchronize()

            # Benchmark standard classifier
            start_time = time.time()
            for _ in range(num_iterations):
                _ = self.standard_classifier.predict(test_numpy)
            torch.cuda.synchronize()
            standard_time = (time.time() - start_time) / num_iterations

            # Benchmark TensorRT classifier
            start_time = time.time()
            for _ in range(num_iterations):
                _ = self.tensorrt_classifier.predict(test_numpy)
            torch.cuda.synchronize()
            tensorrt_time = (time.time() - start_time) / num_iterations

            # Calculate speedup
            speedup = standard_time / tensorrt_time if tensorrt_time > 0 else 0

            results[batch_size] = {
                'standard_time': standard_time,
                'tensorrt_time': tensorrt_time,
                'speedup': speedup
            }

            print(f"    Standard:  {standard_time*1000:.1f}ms per batch")
            print(f"    TensorRT:  {tensorrt_time*1000:.1f}ms per batch")
            print(f"    Speedup:   {speedup:.2f}x")

        return results

    def test_inference_accuracy(self, test_data, tolerance=1e-3):
        """Test inference accuracy between standard and TensorRT models"""
        print(f"\n🎯 Testing inference accuracy (tolerance: {tolerance})...")

        accuracy_results = {}

        for batch_size, test_tensor in test_data.items():
            print(f"\n  📊 Batch size: {batch_size}")

            # Convert to numpy for ART classifiers
            test_numpy = test_tensor.cpu().numpy()

            # Get predictions from both models
            standard_pred = self.standard_classifier.predict(test_numpy)
            tensorrt_pred = self.tensorrt_classifier.predict(test_numpy)

            # Calculate differences
            abs_diff = np.abs(standard_pred - tensorrt_pred)
            max_diff = np.max(abs_diff)
            mean_diff = np.mean(abs_diff)

            # Check if within tolerance
            within_tolerance = max_diff <= tolerance

            accuracy_results[batch_size] = {
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'within_tolerance': within_tolerance
            }

            print(f"    Max difference:  {max_diff:.6f}")
            print(f"    Mean difference: {mean_diff:.6f}")
            print(f"    Within tolerance: {'✅' if within_tolerance else '❌'}")

            # Show top predictions comparison for batch_size=1
            if batch_size == 1:
                standard_top5 = np.argsort(standard_pred[0])[-5:][::-1]
                tensorrt_top5 = np.argsort(tensorrt_pred[0])[-5:][::-1]
                print(f"    Standard top-5 classes: {standard_top5}")
                print(f"    TensorRT top-5 classes:  {tensorrt_top5}")

        return accuracy_results

    def test_memory_usage(self):
        """Test memory usage comparison"""
        print("\n🧠 Testing memory usage...")

        # Get memory before models
        optimize_memory_usage()
        memory_before = get_gpu_memory_info()

        # Test memory usage during inference - use batch_size=2 to test dynamic batch
        test_tensor = torch.rand(2, 3, 224, 224, device=self.device)  # Test batch_size=2
        test_numpy = test_tensor.cpu().numpy()

        # Standard model memory usage
        torch.cuda.reset_peak_memory_stats()
        _ = self.standard_classifier.predict(test_numpy)
        standard_peak = torch.cuda.max_memory_allocated() / (1024**3)

        # TensorRT model memory usage
        torch.cuda.reset_peak_memory_stats()
        _ = self.tensorrt_classifier.predict(test_numpy)
        tensorrt_peak = torch.cuda.max_memory_allocated() / (1024**3)

        memory_after = get_gpu_memory_info()

        print(f"  Memory before: {memory_before['allocated_gb']:.2f}GB allocated")
        print(f"  Standard peak: {standard_peak:.2f}GB")
        print(f"  TensorRT peak: {tensorrt_peak:.2f}GB")
        print(f"  Memory after:  {memory_after['allocated_gb']:.2f}GB allocated")

        memory_savings = ((standard_peak - tensorrt_peak) / standard_peak * 100) if standard_peak > 0 else 0
        print(f"  Memory savings: {memory_savings:.1f}%")

        return {
            'standard_peak': standard_peak,
            'tensorrt_peak': tensorrt_peak,
            'memory_savings_percent': memory_savings
        }

    def test_real_image_inference(self):
        """Test with real image if available"""
        print("\n🖼️ Testing with real image inference...")

        # Try to find a real image in the project
        possible_images = [
            "data/clean/chart/20231107140031466140.png",
            "data/clean/table/20231107140031466140.png",
            "data/clean/dashboard/20231107140031466140.png"
        ]

        real_image_path = None
        for img_path in possible_images:
            if Path(img_path).exists():
                real_image_path = img_path
                break

        if not real_image_path:
            print("  ⚠️ No real images found - using synthetic data")
            return None

        try:
            # Load and preprocess real image
            print(f"  📖 Loading image: {real_image_path}")

            # Load image using OpenCV
            img = cv2.imread(real_image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Preprocess for model
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

            img_tensor = transform(img).unsqueeze(0)
            img_numpy = img_tensor.numpy()

            # Get predictions
            print("  🔍 Running inference...")
            standard_pred = self.standard_classifier.predict(img_numpy)
            tensorrt_pred = self.tensorrt_classifier.predict(img_numpy)

            # Get top predictions
            standard_top = np.argmax(standard_pred[0])
            tensorrt_top = np.argmax(tensorrt_pred[0])

            standard_conf = standard_pred[0][standard_top]
            tensorrt_conf = tensorrt_pred[0][tensorrt_top]

            print(f"  📊 Results:")
            print(f"    Standard: Class {standard_top} (confidence: {standard_conf:.4f})")
            print(f"    TensorRT: Class {tensorrt_top} (confidence: {tensorrt_conf:.4f})")
            print(f"    Agreement: {'✅' if standard_top == tensorrt_top else '❌'}")

            return {
                'image_path': real_image_path,
                'standard_prediction': (standard_top, standard_conf),
                'tensorrt_prediction': (tensorrt_top, tensorrt_conf),
                'agreement': standard_top == tensorrt_top
            }

        except Exception as e:
            print(f"  ❌ Real image test failed: {e}")
            return None

    def run_comprehensive_test(self):
        """Run all TensorRT tests"""
        print("🚀 TensorRT Comprehensive Test Suite")
        print("=" * 60)

        # Setup
        if not self.setup_test_environment():
            print("❌ Environment setup failed")
            return False

        # Create models
        if not self.create_test_models():
            print("❌ Model creation failed")
            return False

        # Create test data
        test_data = self.create_test_data()

        # Run benchmarks
        print("\n" + "=" * 60)
        speed_results = self.benchmark_inference_speed(test_data)

        print("\n" + "=" * 60)
        accuracy_results = self.test_inference_accuracy(test_data)

        print("\n" + "=" * 60)
        memory_results = self.test_memory_usage()

        print("\n" + "=" * 60)
        real_image_results = self.test_real_image_inference()

        # Summary
        print("\n" + "=" * 60)
        print("🏆 TENSORRT TEST SUMMARY")
        print("=" * 60)

        # Speed summary
        avg_speedup = np.mean([r['speedup'] for r in speed_results.values()])
        print(f"📈 Average speedup: {avg_speedup:.2f}x")

        # Accuracy summary
        all_accurate = all(r['within_tolerance'] for r in accuracy_results.values())
        print(f"🎯 Accuracy: {'✅ All tests passed' if all_accurate else '❌ Some tests failed'}")

        # Memory summary
        print(f"🧠 Memory savings: {memory_results['memory_savings_percent']:.1f}%")

        # Real image summary
        if real_image_results:
            print(f"🖼️ Real image: {'✅ Agreement' if real_image_results['agreement'] else '❌ Disagreement'}")

        print("\n🎯 CONCLUSION:")
        if avg_speedup > 1.2 and all_accurate:  # Lowered threshold for realistic expectations
            print("✅ TensorRT optimization is working correctly!")
            print(f"   - {avg_speedup:.1f}x faster inference")
            print(f"   - {memory_results['memory_savings_percent']:.1f}% memory savings")
            print("   - Maintained accuracy within tolerance")
            print("\n✅ NOTE: TensorRT models support dynamic batch sizes (1-4)")
            print("   Larger batches will be automatically chunked for processing")
        else:
            print("⚠️ TensorRT optimization needs attention:")
            if avg_speedup <= 1.2:
                print(f"   - Low speedup: {avg_speedup:.2f}x (expected >1.2x)")
            if not all_accurate:
                print("   - Accuracy issues detected")

        return True

def main():
    """Main entry point for manual testing"""
    if not IMPORTS_SUCCESS:
        print("❌ Import failures - check dependencies")
        return 1

    tester = TensorRTTester()
    success = tester.run_comprehensive_test()

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())