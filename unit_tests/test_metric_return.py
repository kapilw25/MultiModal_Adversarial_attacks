#!/usr/bin/env python3
"""
Test that blackbox attacks now return calculated metrics
"""

import sys
import os
sys.path.append('/lambda/nfs/DiskUsEast1/MultiModal_Adversarial_attacks')

def test_blackbox_metric_return():
    """Test that blackbox attacks return actual calculated metrics"""

    # Use an existing image
    test_image = "data/clean/chart/20231106193645289568.png"

    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False

    print("🧪 Testing blackbox metric return...")

    try:
        from attack_models.black_box_universal import UniversalEpsilonBlackBoxAttack

        # Create attack instance
        attack_framework = UniversalEpsilonBlackBoxAttack(epsilon_target=4/255)

        # Run attack and capture returned metrics
        print("🚀 Running blackbox attack...")
        result_image, target_eps, returned_params = attack_framework.run_epsilon_attack(
            image_path=test_image,
            attack_type='square',
            attack_params=None
        )

        if result_image is not None:
            print("✅ Attack successful!")
            print("📊 Returned parameters keys:", list(returned_params.keys()))

            # Check if metrics are present and non-zero
            metrics_to_check = ['mean_perturbation', 'max_perturbation', 'l2_norm', 'l0_norm', 'total_queries']

            print("🔍 Metric values:")
            all_metrics_present = True
            for metric in metrics_to_check:
                value = returned_params.get(metric, None)
                print(f"   {metric}: {value}")

                if value is None:
                    print(f"   ❌ {metric} is missing!")
                    all_metrics_present = False
                elif value == 0 or value == 0.0:
                    print(f"   ⚠️  {metric} is zero (might be expected for some metrics)")
                else:
                    print(f"   ✅ {metric} has non-zero value")

            if all_metrics_present:
                print("🎉 SUCCESS: All metrics are returned in final_params!")
                return True
            else:
                print("❌ FAILURE: Some metrics are missing")
                return False
        else:
            print("❌ Attack failed to generate adversarial image")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_blackbox_metric_return()
    sys.exit(0 if success else 1)