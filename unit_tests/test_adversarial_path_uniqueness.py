#!/usr/bin/env python3
"""
Unit Test: Verify adversarial_image_path uniqueness in database

Tests:
1. execution_id is derived from adversarial_image_path
2. Re-inserting same adversarial_image_path replaces existing entry
3. Database has no duplicate adversarial_image_path values
"""

import sys
import os
import sqlite3
import tempfile
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils.centralized_database import CentralizedDB, create_centralized_schema

def test_execution_id_generation():
    """Test 1: execution_id is generated from adversarial_image_path"""
    print("=" * 70)
    print("TEST 1: Execution ID Generation from adversarial_image_path")
    print("=" * 70)

    # Create temp database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db_path = temp_db.name
    temp_db.close()

    # Override DB_PATH
    import scripts.utils.centralized_database as db_module
    original_db_path = db_module.DB_PATH
    db_module.DB_PATH = temp_db_path

    try:
        db = CentralizedDB()

        # Insert attack result
        result_data = {
            'image_path': 'data/clean/chart/20231107140031466140.png',
            'adversarial_image_path': 'data/adversarial/whitebox/fgsm/eps_0016/chart/20231107140031466140.png',
            'attack_type': 'fgsm',
            'attack_category': 'whitebox',
            'task_type': 'chart',
            'epsilon_level': 'minimal',
            'epsilon_target': 0.0157,
            'epsilon_achieved': 0.0156,
            'success': True,
            'execution_time': 1.5
        }

        db.insert_attack_result(result_data)

        # Verify execution_id is derived from adversarial_image_path
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT execution_id, adversarial_image_path FROM attack_executions")
        rows = cursor.fetchall()
        conn.close()

        assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"

        execution_id, adv_path = rows[0]
        expected_execution_id = 'data_adversarial_whitebox_fgsm_eps_0016_chart_20231107140031466140'

        print(f"✅ Execution ID: {execution_id}")
        print(f"✅ Expected:     {expected_execution_id}")
        print(f"✅ Match: {execution_id == expected_execution_id}")

        assert execution_id == expected_execution_id, f"execution_id mismatch: {execution_id} != {expected_execution_id}"

        print("✅ TEST 1 PASSED: execution_id correctly derived from adversarial_image_path")
        return True

    finally:
        # Cleanup
        db_module.DB_PATH = original_db_path
        os.unlink(temp_db_path)
        print()


def test_duplicate_replacement():
    """Test 2: Re-inserting same adversarial_image_path replaces (not duplicates)"""
    print("=" * 70)
    print("TEST 2: Duplicate Replacement (INSERT OR REPLACE)")
    print("=" * 70)

    # Create temp database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db_path = temp_db.name
    temp_db.close()

    # Override DB_PATH
    import scripts.utils.centralized_database as db_module
    original_db_path = db_module.DB_PATH
    db_module.DB_PATH = temp_db_path

    try:
        db = CentralizedDB()

        # First insertion
        result_data_1 = {
            'image_path': 'data/clean/chart/20231107140031466140.png',
            'adversarial_image_path': 'data/adversarial/whitebox/fgsm/eps_0016/chart/20231107140031466140.png',
            'attack_type': 'fgsm',
            'attack_category': 'whitebox',
            'task_type': 'chart',
            'epsilon_level': 'minimal',
            'epsilon_target': 0.0157,
            'epsilon_achieved': 0.0156,
            'success': True,
            'execution_time': 1.5,
            'timestamp': '2025-01-01T10:00:00'
        }

        db.insert_attack_result(result_data_1)
        print("✅ First insertion completed")

        # Check count
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM attack_executions WHERE adversarial_image_path = ?",
                      (result_data_1['adversarial_image_path'],))
        count_1 = cursor.fetchone()[0]
        print(f"   Rows with path: {count_1}")

        # Second insertion (SAME adversarial_image_path, DIFFERENT timestamp and epsilon)
        result_data_2 = {
            'image_path': 'data/clean/chart/20231107140031466140.png',
            'adversarial_image_path': 'data/adversarial/whitebox/fgsm/eps_0016/chart/20231107140031466140.png',
            'attack_type': 'fgsm',
            'attack_category': 'whitebox',
            'task_type': 'chart',
            'epsilon_level': 'minimal',
            'epsilon_target': 0.0157,
            'epsilon_achieved': 0.0158,  # Different epsilon
            'success': True,
            'execution_time': 2.1,  # Different time
            'timestamp': '2025-01-02T15:30:00'  # Different timestamp
        }

        db.insert_attack_result(result_data_2)
        print("✅ Second insertion completed (same path, different data)")

        # Check count again - should still be 1 (replaced, not added)
        cursor.execute("SELECT COUNT(*) FROM attack_executions WHERE adversarial_image_path = ?",
                      (result_data_2['adversarial_image_path'],))
        count_2 = cursor.fetchone()[0]
        print(f"   Rows with path: {count_2}")

        # Verify latest values are from second insertion
        cursor.execute("""
            SELECT epsilon_target, epsilon_l_inf, timestamp, execution_time_seconds
            FROM attack_executions
            WHERE adversarial_image_path = ?
        """, (result_data_2['adversarial_image_path'],))
        row = cursor.fetchone()
        conn.close()

        epsilon_target, epsilon_l_inf, timestamp, exec_time = row

        print(f"   Latest epsilon_target: {epsilon_target} (expected: {result_data_2['epsilon_target']})")
        print(f"   Latest epsilon_l_inf: {epsilon_l_inf} (expected: {result_data_2['epsilon_achieved']})")
        print(f"   Latest timestamp: {timestamp} (expected: {result_data_2['timestamp']})")
        print(f"   Latest exec_time: {exec_time} (expected: {result_data_2['execution_time']})")

        assert count_1 == 1, f"First insertion should create 1 row, got {count_1}"
        assert count_2 == 1, f"Second insertion should REPLACE (still 1 row), got {count_2}"
        assert epsilon_l_inf == result_data_2['epsilon_achieved'], f"epsilon_l_inf not updated"
        assert timestamp == result_data_2['timestamp'], f"timestamp not updated"

        print("✅ TEST 2 PASSED: Same adversarial_image_path correctly replaces existing entry")
        return True

    finally:
        # Cleanup
        db_module.DB_PATH = original_db_path
        os.unlink(temp_db_path)
        print()


def test_database_uniqueness():
    """Test 3: Verify actual database has no duplicate adversarial_image_path"""
    print("=" * 70)
    print("TEST 3: Database Uniqueness Check")
    print("=" * 70)

    db_path = "results/centralized_pipeline.db"

    if not os.path.exists(db_path):
        print("⚠️  Database not found, skipping test")
        print()
        return True

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Count total vs unique paths
    cursor.execute("SELECT COUNT(*) as total_rows FROM attack_executions")
    total_rows = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT adversarial_image_path) as unique_paths FROM attack_executions")
    unique_paths = cursor.fetchone()[0]

    # Find duplicates
    cursor.execute("""
        SELECT adversarial_image_path, COUNT(*) as count
        FROM attack_executions
        GROUP BY adversarial_image_path
        HAVING COUNT(*) > 1
        ORDER BY count DESC
        LIMIT 10
    """)
    duplicates = cursor.fetchall()

    conn.close()

    print(f"   Total rows: {total_rows}")
    print(f"   Unique adversarial_image_path: {unique_paths}")
    print(f"   Duplicates: {total_rows - unique_paths}")
    print()

    if duplicates:
        print("❌ DUPLICATES FOUND:")
        for path, count in duplicates:
            print(f"   {count}× {path}")
        print()
        print("❌ TEST 3 FAILED: Database contains duplicate adversarial_image_path entries")
        print("💡 Run attack_runner.py again with fix to clean up duplicates")
        return False
    else:
        print("✅ TEST 3 PASSED: No duplicate adversarial_image_path in database")
        return True

    print()


def test_multiple_attacks_same_image():
    """Test 4: Different attacks on same image create different paths"""
    print("=" * 70)
    print("TEST 4: Multiple Attacks on Same Image")
    print("=" * 70)

    # Create temp database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db_path = temp_db.name
    temp_db.close()

    # Override DB_PATH
    import scripts.utils.centralized_database as db_module
    original_db_path = db_module.DB_PATH
    db_module.DB_PATH = temp_db_path

    try:
        db = CentralizedDB()

        # Insert 3 different attacks on same clean image
        attacks = [
            {
                'attack_type': 'fgsm',
                'adversarial_image_path': 'data/adversarial/whitebox/fgsm/eps_0016/chart/image1.png',
            },
            {
                'attack_type': 'pgd',
                'adversarial_image_path': 'data/adversarial/whitebox/pgd/eps_0016/chart/image1.png',
            },
            {
                'attack_type': 'auto_pgd',
                'adversarial_image_path': 'data/adversarial/whitebox/auto_pgd/eps_0016/chart/image1.png',
            }
        ]

        for attack in attacks:
            result_data = {
                'image_path': 'data/clean/chart/image1.png',
                'adversarial_image_path': attack['adversarial_image_path'],
                'attack_type': attack['attack_type'],
                'attack_category': 'whitebox',
                'task_type': 'chart',
                'epsilon_level': 'minimal',
                'epsilon_target': 0.0157,
                'epsilon_achieved': 0.0156,
                'success': True,
                'execution_time': 1.5
            }
            db.insert_attack_result(result_data)

        # Check total rows
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM attack_executions")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT adversarial_image_path) FROM attack_executions")
        unique = cursor.fetchone()[0]

        cursor.execute("SELECT adversarial_image_path FROM attack_executions ORDER BY adversarial_image_path")
        paths = [row[0] for row in cursor.fetchall()]
        conn.close()

        print(f"   Total rows: {total}")
        print(f"   Unique paths: {unique}")
        print(f"   Paths:")
        for path in paths:
            print(f"      {path}")

        assert total == 3, f"Expected 3 rows, got {total}"
        assert unique == 3, f"Expected 3 unique paths, got {unique}"

        print("✅ TEST 4 PASSED: Different attacks create different adversarial paths")
        return True

    finally:
        # Cleanup
        db_module.DB_PATH = original_db_path
        os.unlink(temp_db_path)
        print()


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("ADVERSARIAL_IMAGE_PATH UNIQUENESS TEST SUITE")
    print("=" * 70)
    print()

    tests = [
        ("Execution ID Generation", test_execution_id_generation),
        ("Duplicate Replacement", test_duplicate_replacement),
        ("Multiple Attacks Same Image", test_multiple_attacks_same_image),
        ("Database Uniqueness", test_database_uniqueness),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ TEST FAILED: {test_name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("✅ ALL TESTS PASSED")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
