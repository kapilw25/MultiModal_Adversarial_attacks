#!/usr/bin/env python3
"""
Test script to verify perturbation metrics are correctly saved to database
Run a single blackbox attack and verify metrics are saved (not zeros)
"""

import sqlite3
import subprocess
import sys
import os

def test_metric_fix():
    """Test that blackbox attacks save actual metrics instead of zeros"""

    print("🧪 Testing metric fix for blackbox attacks...")

    # Check if test image exists
    test_image = "data/images/chart/20231107140031466140.png"
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False

    # Clear database for clean test
    if os.path.exists('results/centralized_pipeline.db'):
        os.remove('results/centralized_pipeline.db')
        print("✅ Cleared existing database")

    # Create fresh schema
    from scripts.utils.centralized_database import create_centralized_schema
    create_centralized_schema()
    print("✅ Created fresh database schema")

    # Run attack_runner.py with automation for single attack
    print("🚀 Running attack_runner.py with automation...")

    # Create input simulation for attack_runner.py
    # Chart task (1) -> Blackbox (2) -> Square (1) -> Minimal epsilon (1) -> Selective overwrite (N)
    automation_input = "1\n2\n1\n1\nN\n"

    try:
        # Run attack_runner.py with automated input
        result = subprocess.run(
            [sys.executable, "scripts/attack_runner.py"],
            input=automation_input,
            text=True,
            capture_output=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            print("✅ Attack execution completed")

            # Check database for metrics
            conn = sqlite3.connect('results/centralized_pipeline.db')
            cursor = conn.cursor()

            cursor.execute('''
                SELECT attack_name, mean_perturbation, max_perturbation, l2_norm, l0_norm, total_queries
                FROM attack_executions
                ORDER BY timestamp DESC LIMIT 1
            ''')
            db_result = cursor.fetchone()
            conn.close()

            if db_result:
                print(f"\n🔍 Database verification:")
                print(f"   Attack: {db_result[0]}")
                print(f"   Mean perturbation: {db_result[1]} (should be > 0)")
                print(f"   Max perturbation: {db_result[2]} (should be > 0)")
                print(f"   L2 norm: {db_result[3]} (should be > 0)")
                print(f"   L0 norm: {db_result[4]} (should be > 0)")
                print(f"   Total queries: {db_result[5]} (should be > 0)")

                # Check if metrics are non-zero
                non_zero_metrics = []
                metrics = ['mean_perturbation', 'max_perturbation', 'l2_norm', 'l0_norm', 'total_queries']
                for i, value in enumerate(db_result[1:], 0):
                    if value > 0:
                        non_zero_metrics.append(metrics[i])

                print(f"\n✅ Non-zero metrics: {non_zero_metrics}")

                if len(non_zero_metrics) >= 4:
                    print("🎉 SUCCESS: Metric fix works! All perturbation metrics are correctly saved!")
                    return True
                else:
                    print("❌ FAILURE: Metrics are still zero or missing!")
                    return False
            else:
                print("❌ No database entry found!")
                return False
        else:
            print(f"❌ Attack execution failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_metric_fix()
    sys.exit(0 if success else 1)