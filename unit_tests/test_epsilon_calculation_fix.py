#!/usr/bin/env python3
"""
Unit test to verify epsilon calculation fix for the mismatch between epsilon_target and epsilon_actual.

Tests the improved calculate_epsilon function that handles mixed input ranges correctly.
"""

import sys
import os
import unittest
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from attack_models.utils import calculate_epsilon


class TestEpsilonCalculationFix(unittest.TestCase):
    """Test suite for epsilon calculation fix."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test images in different ranges
        np.random.seed(42)  # For reproducible tests

        # Original image in [0,1] range
        self.img_01_range = np.random.rand(224, 224, 3).astype(np.float32)

        # Same image in [0,255] range
        self.img_255_range = (self.img_01_range * 255).astype(np.uint8)

        # Create adversarial perturbations
        self.epsilon_target = 0.05

        # Perturbation in [0,1] space
        perturbation_01 = np.random.uniform(-self.epsilon_target, self.epsilon_target,
                                          self.img_01_range.shape).astype(np.float32)

        # Create adversarial images
        self.adv_img_01 = np.clip(self.img_01_range + perturbation_01, 0, 1)
        self.adv_img_255 = (self.adv_img_01 * 255).astype(np.uint8)

        print(f"🔧 Test setup:")
        print(f"   Original [0,1] range: {self.img_01_range.min():.3f} to {self.img_01_range.max():.3f}")
        print(f"   Original [0,255] range: {self.img_255_range.min()} to {self.img_255_range.max()}")
        print(f"   Target epsilon: {self.epsilon_target}")

    def test_same_range_01_space(self):
        """Test epsilon calculation when both images are in [0,1] range."""
        print("\n🧪 Testing [0,1] vs [0,1] range...")

        epsilon_calc = calculate_epsilon(self.img_01_range, self.adv_img_01)

        # Should be close to target epsilon
        self.assertLess(abs(epsilon_calc - self.epsilon_target), 0.01,
                       f"Epsilon should be close to {self.epsilon_target}, got {epsilon_calc}")

        # Should be reasonable (not 255.0!)
        self.assertLess(epsilon_calc, 1.0,
                       f"Epsilon should be < 1.0 for [0,1] images, got {epsilon_calc}")

        print(f"   ✅ Calculated epsilon: {epsilon_calc:.6f} (target: {self.epsilon_target})")

    def test_same_range_255_space(self):
        """Test epsilon calculation when both images are in [0,255] range."""
        print("\n🧪 Testing [0,255] vs [0,255] range...")

        epsilon_calc = calculate_epsilon(self.img_255_range, self.adv_img_255)

        # Should be close to target epsilon (normalized)
        self.assertLess(abs(epsilon_calc - self.epsilon_target), 0.01,
                       f"Epsilon should be close to {self.epsilon_target}, got {epsilon_calc}")

        # Should NOT be 255.0 (the old bug!)
        self.assertLess(epsilon_calc, 1.0,
                       f"Epsilon should be < 1.0 even for [0,255] images, got {epsilon_calc}")

        print(f"   ✅ Calculated epsilon: {epsilon_calc:.6f} (target: {self.epsilon_target})")

    def test_mixed_ranges_01_vs_255(self):
        """Test epsilon calculation with mixed ranges: [0,1] vs [0,255]."""
        print("\n🧪 Testing [0,1] vs [0,255] mixed ranges...")

        epsilon_calc = calculate_epsilon(self.img_01_range, self.adv_img_255)

        # Should be close to target epsilon
        self.assertLess(abs(epsilon_calc - self.epsilon_target), 0.01,
                       f"Epsilon should be close to {self.epsilon_target}, got {epsilon_calc}")

        # Should be reasonable
        self.assertLess(epsilon_calc, 1.0,
                       f"Epsilon should be < 1.0 for mixed ranges, got {epsilon_calc}")

        print(f"   ✅ Calculated epsilon: {epsilon_calc:.6f} (target: {self.epsilon_target})")

    def test_mixed_ranges_255_vs_01(self):
        """Test epsilon calculation with mixed ranges: [0,255] vs [0,1]."""
        print("\n🧪 Testing [0,255] vs [0,1] mixed ranges...")

        epsilon_calc = calculate_epsilon(self.img_255_range, self.adv_img_01)

        # Should be close to target epsilon
        self.assertLess(abs(epsilon_calc - self.epsilon_target), 0.01,
                       f"Epsilon should be close to {self.epsilon_target}, got {epsilon_calc}")

        # Should be reasonable
        self.assertLess(epsilon_calc, 1.0,
                       f"Epsilon should be < 1.0 for mixed ranges, got {epsilon_calc}")

        print(f"   ✅ Calculated epsilon: {epsilon_calc:.6f} (target: {self.epsilon_target})")

    def test_bug_reproduction_old_behavior(self):
        """Test that reproduces the old bug (epsilon = 255.0) for documentation."""
        print("\n🧪 Reproducing old bug behavior...")

        # Simulate old buggy calculation (just for demonstration)
        img1_255 = self.img_255_range.astype(np.float32)
        img2_255 = self.adv_img_255.astype(np.float32)

        # This would be the old buggy calculation (without normalization)
        old_buggy_epsilon = float(np.max(np.abs(img1_255 - img2_255)))

        print(f"   🐛 Old buggy epsilon: {old_buggy_epsilon:.1f} (this was the problem!)")

        # New fixed calculation
        new_fixed_epsilon = calculate_epsilon(self.img_255_range, self.adv_img_255)

        print(f"   ✅ New fixed epsilon: {new_fixed_epsilon:.6f}")

        # Verify the fix
        self.assertGreater(old_buggy_epsilon, 1.0, "Old calculation should give large values")
        self.assertLess(new_fixed_epsilon, 1.0, "New calculation should give reasonable values")

    def test_zero_perturbation(self):
        """Test epsilon calculation with identical images (zero perturbation)."""
        print("\n🧪 Testing zero perturbation...")

        # Test with [0,1] range
        epsilon_01 = calculate_epsilon(self.img_01_range, self.img_01_range)
        self.assertAlmostEqual(epsilon_01, 0.0, places=6)

        # Test with [0,255] range
        epsilon_255 = calculate_epsilon(self.img_255_range, self.img_255_range)
        self.assertAlmostEqual(epsilon_255, 0.0, places=6)

        print(f"   ✅ Zero perturbation epsilons: {epsilon_01:.6f}, {epsilon_255:.6f}")

    def test_known_perturbation(self):
        """Test epsilon calculation with a known perturbation."""
        print("\n🧪 Testing known perturbation...")

        # Create image with exactly known epsilon
        base_img = np.zeros((10, 10, 3), dtype=np.float32)  # All zeros in [0,1]
        known_epsilon = 0.1
        perturbed_img = np.full_like(base_img, known_epsilon)  # All values = epsilon

        calculated_epsilon = calculate_epsilon(base_img, perturbed_img)

        self.assertAlmostEqual(calculated_epsilon, known_epsilon, places=6,
                              msg=f"Should calculate exact epsilon {known_epsilon}, got {calculated_epsilon}")

        print(f"   ✅ Known epsilon test: expected {known_epsilon}, got {calculated_epsilon:.6f}")

    def test_database_scenario_simulation(self):
        """Simulate the exact scenario that was causing database mismatches."""
        print("\n🧪 Simulating database scenario...")

        # Test with standard research epsilon values
        target_epsilons = [4/255, 8/255, 16/255]  # ≈ [0.016, 0.031, 0.063]

        for eps_target in target_epsilons:
            # Create realistic adversarial scenario
            original = np.random.rand(224, 224, 3).astype(np.float32)

            # Add controlled perturbation
            perturbation = np.random.uniform(-eps_target, eps_target, original.shape)
            adversarial = np.clip(original + perturbation, 0, 1).astype(np.float32)

            # Test the fixed calculation
            eps_actual = calculate_epsilon(original, adversarial)

            # Should NOT be 255.0 (the old bug)
            self.assertLess(eps_actual, 1.0,
                           f"Epsilon should be < 1.0, got {eps_actual} for target {eps_target}")

            # Should be reasonably close to target
            self.assertLess(abs(eps_actual - eps_target), eps_target * 0.5,
                           f"Epsilon should be reasonably close to target {eps_target}, got {eps_actual}")

            print(f"   ✅ Target: {eps_target:.3f} → Actual: {eps_actual:.6f} ✓")


def run_epsilon_fix_test():
    """Run the epsilon calculation fix test."""
    print("=" * 80)
    print("🚀 TESTING EPSILON CALCULATION FIX")
    print("=" * 80)
    print("This test verifies the fix for epsilon_target vs epsilon_actual mismatch")
    print("where epsilon_actual was incorrectly calculated as 255.0 instead of ~0.05")
    print()

    # Run tests
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_epsilon_fix_test()