#!/usr/bin/env python3
"""
Cleanup Script: Remove duplicate adversarial_image_path entries from database

Strategy:
- For each duplicate adversarial_image_path, keep ONLY the latest entry (by timestamp)
- Delete older entries
- Verify uniqueness after cleanup
"""

import sys
import os
import sqlite3
from datetime import datetime

def analyze_duplicates(db_path):
    """Analyze duplicates before cleanup"""
    print("=" * 70)
    print("ANALYZING DUPLICATES")
    print("=" * 70)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get total and unique counts
    cursor.execute("SELECT COUNT(*) FROM attack_executions")
    total_before = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT adversarial_image_path) FROM attack_executions")
    unique_before = cursor.fetchone()[0]

    # Find all duplicate paths
    cursor.execute("""
        SELECT adversarial_image_path, COUNT(*) as count
        FROM attack_executions
        GROUP BY adversarial_image_path
        HAVING COUNT(*) > 1
        ORDER BY count DESC
    """)
    duplicates = cursor.fetchall()

    conn.close()

    print(f"   Total rows: {total_before}")
    print(f"   Unique paths: {unique_before}")
    print(f"   Duplicate paths: {len(duplicates)}")
    print(f"   Rows to delete: {total_before - unique_before}")
    print()

    if duplicates:
        print(f"   Top 10 duplicates:")
        for path, count in duplicates[:10]:
            print(f"      {count}× {path}")
        print()

    return total_before, unique_before, len(duplicates)


def cleanup_duplicates(db_path, dry_run=True):
    """Remove duplicate adversarial_image_path entries, keeping only latest"""
    print("=" * 70)
    print(f"CLEANUP MODE: {'DRY RUN (no changes)' if dry_run else 'EXECUTING (will modify database)'}")
    print("=" * 70)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Find all duplicate adversarial_image_path values
    cursor.execute("""
        SELECT adversarial_image_path
        FROM attack_executions
        GROUP BY adversarial_image_path
        HAVING COUNT(*) > 1
    """)
    duplicate_paths = [row[0] for row in cursor.fetchall()]

    total_deleted = 0

    for dup_path in duplicate_paths:
        # Get all entries for this path, ordered by timestamp (newest first)
        cursor.execute("""
            SELECT execution_id, timestamp, epsilon_l_inf, attack_name
            FROM attack_executions
            WHERE adversarial_image_path = ?
            ORDER BY timestamp DESC
        """, (dup_path,))

        entries = cursor.fetchall()

        if len(entries) <= 1:
            continue

        # Keep the first (latest) entry, delete the rest
        latest_entry = entries[0]
        old_entries = entries[1:]

        print(f"\n📂 Path: {dup_path}")
        print(f"   Total: {len(entries)} entries")
        print(f"   ✅ KEEP (latest): {latest_entry[1]} | ε={latest_entry[2]:.4f} | {latest_entry[3]}")

        for old_entry in old_entries:
            exec_id, timestamp, epsilon, attack = old_entry
            print(f"   ❌ DELETE (old):   {timestamp} | ε={epsilon:.4f} | {attack}")

            if not dry_run:
                cursor.execute("DELETE FROM attack_executions WHERE execution_id = ?", (exec_id,))
                total_deleted += 1

    if not dry_run:
        conn.commit()
        print(f"\n✅ Deleted {total_deleted} duplicate entries")
    else:
        print(f"\n💡 Would delete {len(duplicate_paths) - len([p for p in duplicate_paths if p])} duplicate entries")
        print(f"   (Run with --execute to apply changes)")

    conn.close()
    return total_deleted


def verify_cleanup(db_path):
    """Verify no duplicates remain"""
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM attack_executions")
    total_after = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT adversarial_image_path) FROM attack_executions")
    unique_after = cursor.fetchone()[0]

    cursor.execute("""
        SELECT adversarial_image_path, COUNT(*) as count
        FROM attack_executions
        GROUP BY adversarial_image_path
        HAVING COUNT(*) > 1
    """)
    duplicates_after = cursor.fetchall()

    conn.close()

    print(f"   Total rows: {total_after}")
    print(f"   Unique paths: {unique_after}")
    print(f"   Duplicates remaining: {len(duplicates_after)}")

    if duplicates_after:
        print("\n❌ DUPLICATES STILL EXIST:")
        for path, count in duplicates_after[:10]:
            print(f"   {count}× {path}")
        return False
    else:
        print("\n✅ NO DUPLICATES - Database is clean!")
        return True


def main():
    """Main cleanup workflow"""
    db_path = "results/centralized_pipeline.db"

    if not os.path.exists(db_path):
        print(f"❌ Database not found: {db_path}")
        return 1

    print("\n" + "=" * 70)
    print("DUPLICATE CLEANUP TOOL")
    print("=" * 70)
    print(f"Database: {db_path}")
    print()

    # Analyze
    total_before, unique_before, num_duplicates = analyze_duplicates(db_path)

    if num_duplicates == 0:
        print("✅ No duplicates found - database is already clean!")
        return 0

    # Determine mode
    import sys
    execute_mode = '--execute' in sys.argv

    # Cleanup (dry run or execute)
    cleanup_duplicates(db_path, dry_run=not execute_mode)

    # Verify
    if execute_mode:
        if verify_cleanup(db_path):
            print("\n✅ CLEANUP SUCCESSFUL")
            return 0
        else:
            print("\n❌ CLEANUP FAILED - duplicates remain")
            return 1
    else:
        print("\n💡 This was a dry run. To execute cleanup, run:")
        print(f"   python3 {sys.argv[0]} --execute")
        return 0


if __name__ == "__main__":
    exit(main())
