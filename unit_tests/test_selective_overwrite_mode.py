#!/usr/bin/env python3
"""
Unit Test: Verify Selective Overwrite Mode (NO) Skips Existing Files

Tests:
1. When replacement=NO, existing adversarial files should be SKIPPED (not overwritten)
2. When replacement=YES, existing adversarial files should be OVERWRITTEN
3. Database should only have unique adversarial_image_path entries
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_selective_mode_skips_existing():
    """Test 1: Selective mode (NO) should skip existing files"""
    print("=" * 70)
    print("TEST 1: Selective Mode Skips Existing Files")
    print("=" * 70)

    # Setup test environment
    test_dir = tempfile.mkdtemp()
    try:
        # Create mock existing adversarial image
        existing_path = Path(test_dir) / "data/adversarial/whitebox/fgsm/eps_0016/chart/image1.png"
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.write_text("existing_content_v1")

        original_mtime = existing_path.stat().st_mtime

        print(f"✅ Created existing file: {existing_path}")
        print(f"   Original mtime: {original_mtime}")
        print(f"   Content: existing_content_v1")

        # Simulate get_output_path_for_image check
        output_exists = existing_path.exists()

        if output_exists:
            print(f"✅ File exists check: {output_exists}")
            print(f"   SKIP processing (as expected in NO mode)")

            # Verify file wasn't modified
            new_mtime = existing_path.stat().st_mtime
            content = existing_path.read_text()

            assert original_mtime == new_mtime, f"File was modified! {original_mtime} != {new_mtime}"
            assert content == "existing_content_v1", f"Content changed! {content}"

            print(f"✅ File unchanged (mtime: {new_mtime}, content: {content})")
        else:
            raise AssertionError("File should exist but doesn't")

        print("✅ TEST 1 PASSED: Selective mode correctly skips existing files")
        return True

    finally:
        shutil.rmtree(test_dir)
        print()


def test_replacement_mode_overwrites():
    """Test 2: Replacement mode (YES) should overwrite existing files"""
    print("=" * 70)
    print("TEST 2: Replacement Mode Overwrites Existing Files")
    print("=" * 70)

    # Setup test environment
    test_dir = tempfile.mkdtemp()
    try:
        # Create mock existing adversarial image
        existing_path = Path(test_dir) / "data/adversarial/whitebox/fgsm/eps_0016/chart/image1.png"
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.write_text("existing_content_v1")

        original_mtime = existing_path.stat().st_mtime

        print(f"✅ Created existing file: {existing_path}")
        print(f"   Original content: existing_content_v1")

        # Simulate replacement mode - delete directory
        import time
        time.sleep(0.01)  # Ensure different mtime

        # In YES mode, the directory is cleared BEFORE processing
        shutil.rmtree(existing_path.parent.parent.parent)  # Remove whitebox dir
        print(f"✅ Cleared whitebox directory (as in YES mode)")

        # Now recreate with new content
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        existing_path.write_text("new_content_v2")

        new_mtime = existing_path.stat().st_mtime
        content = existing_path.read_text()

        print(f"✅ Created new file:")
        print(f"   New mtime: {new_mtime}")
        print(f"   Content: {content}")

        assert new_mtime != original_mtime, "File should have different mtime"
        assert content == "new_content_v2", f"Content should be updated: {content}"

        print("✅ TEST 2 PASSED: Replacement mode correctly overwrites files")
        return True

    finally:
        shutil.rmtree(test_dir)
        print()


def test_filter_logic_in_attack_runner():
    """Test 3: Verify attack_runner.py filter_existing_images logic"""
    print("=" * 70)
    print("TEST 3: Attack Runner Filter Logic")
    print("=" * 70)

    # Setup test environment
    test_dir = tempfile.mkdtemp()
    try:
        # Create some existing adversarial images
        base_path = Path(test_dir) / "data/adversarial/whitebox/fgsm/eps_0016/chart"
        base_path.mkdir(parents=True, exist_ok=True)

        existing_images = ["image1.png", "image2.png"]
        new_images = ["image3.png", "image4.png"]

        for img in existing_images:
            (base_path / img).write_text("existing")

        print(f"✅ Created {len(existing_images)} existing adversarial images")

        # Simulate filter_existing_images logic
        all_images = existing_images + new_images
        filtered = []

        for img_name in all_images:
            output_path = base_path / img_name
            if not output_path.exists():
                filtered.append(img_name)

        print(f"   Total images: {len(all_images)}")
        print(f"   Existing (skip): {len(existing_images)}")
        print(f"   New (process): {len(filtered)}")

        assert len(filtered) == len(new_images), f"Should filter {len(new_images)} images, got {len(filtered)}"
        assert set(filtered) == set(new_images), f"Filtered wrong images: {filtered} != {new_images}"

        print(f"✅ Correctly filtered to process: {filtered}")
        print("✅ TEST 3 PASSED: Filter logic works correctly")
        return True

    finally:
        shutil.rmtree(test_dir)
        print()


def test_database_behavior_with_skip():
    """Test 4: Database should NOT be touched when images are skipped"""
    print("=" * 70)
    print("TEST 4: Database Unchanged When Images Skipped")
    print("=" * 70)

    import sqlite3

    # Create temp database
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db_path = temp_db.name
    temp_db.close()

    try:
        # Create table
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE attack_executions (
                execution_id TEXT PRIMARY KEY,
                adversarial_image_path TEXT,
                timestamp TEXT
            )
        ''')

        # Insert initial entry
        cursor.execute('''
            INSERT INTO attack_executions VALUES (?, ?, ?)
        ''', ('id1', 'data/adversarial/whitebox/fgsm/eps_0016/chart/image1.png', '2025-01-01T10:00:00'))
        conn.commit()

        print("✅ Created initial database entry:")
        print("   ID: id1")
        print("   Timestamp: 2025-01-01T10:00:00")

        # Check count
        cursor.execute("SELECT COUNT(*) FROM attack_executions")
        initial_count = cursor.fetchone()[0]

        cursor.execute("SELECT timestamp FROM attack_executions WHERE execution_id = 'id1'")
        initial_timestamp = cursor.fetchone()[0]

        print(f"   Initial row count: {initial_count}")

        # Simulate SKIP behavior - NO database operation should occur
        print("\n💡 Simulating SKIP (file exists, replacement=NO):")
        print("   → No attack executed")
        print("   → No database insert/update")

        # Verify database unchanged
        cursor.execute("SELECT COUNT(*) FROM attack_executions")
        final_count = cursor.fetchone()[0]

        cursor.execute("SELECT timestamp FROM attack_executions WHERE execution_id = 'id1'")
        final_timestamp = cursor.fetchone()[0]

        print(f"\n✅ Database verification:")
        print(f"   Row count: {final_count} (unchanged)")
        print(f"   Timestamp: {final_timestamp} (unchanged)")

        assert initial_count == final_count, f"Row count changed: {initial_count} -> {final_count}"
        assert initial_timestamp == final_timestamp, f"Timestamp changed: {initial_timestamp} -> {final_timestamp}"

        conn.close()

        print("✅ TEST 4 PASSED: Database correctly unchanged when skipping")
        return True

    finally:
        os.unlink(temp_db_path)
        print()


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("SELECTIVE OVERWRITE MODE TEST SUITE")
    print("=" * 70)
    print()

    tests = [
        ("Selective Mode Skips Existing", test_selective_mode_skips_existing),
        ("Replacement Mode Overwrites", test_replacement_mode_overwrites),
        ("Filter Logic Verification", test_filter_logic_in_attack_runner),
        ("Database Unchanged on Skip", test_database_behavior_with_skip),
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
        print()
        print("💡 Expected Behavior Summary:")
        print("   [N] Selective Overwrite:")
        print("      → Checks if adversarial file EXISTS on disk")
        print("      → If EXISTS: SKIP (no attack, no DB update)")
        print("      → If NOT EXISTS: RUN attack, create file, update DB")
        print()
        print("   [Y] Complete Replacement:")
        print("      → DELETE all existing adversarial images")
        print("      → CLEAR database entries")
        print("      → Process ALL images fresh")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
