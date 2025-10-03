#!/usr/bin/env python3
"""
Verify Image Paths from attack_executions Table

Checks if all image paths (clean and adversarial) from attack_executions table
actually exist on disk.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.utils.centralized_database import CentralizedDB

def verify_image_paths():
    """Verify all image paths from attack_executions table exist on disk"""

    db = CentralizedDB()
    conn = db.get_connection()
    cursor = conn.cursor()

    # Get all unique clean image paths
    cursor.execute('SELECT DISTINCT image_path FROM attack_executions')
    clean_paths = [row[0] for row in cursor.fetchall()]

    # Get all unique adversarial image paths
    cursor.execute('SELECT DISTINCT adversarial_image_path FROM attack_executions')
    adv_paths = [row[0] for row in cursor.fetchall()]

    conn.close()

    print("="*80)
    print("IMAGE PATH VERIFICATION")
    print("="*80)

    # Verify clean images
    print(f"\n📂 Clean Images (image_path):")
    print(f"   Total unique paths: {len(clean_paths)}")

    missing_clean = []
    for path in clean_paths:
        if not os.path.exists(path):
            missing_clean.append(path)

    if missing_clean:
        print(f"   ❌ Missing: {len(missing_clean)}")
        print(f"\n   Missing clean image paths:")
        for path in missing_clean:
            print(f"      {path}")
    else:
        print(f"   ✅ All clean images exist on disk")

    # Verify adversarial images
    print(f"\n🎯 Adversarial Images (adversarial_image_path):")
    print(f"   Total unique paths: {len(adv_paths)}")

    missing_adv = []
    for path in adv_paths:
        if not os.path.exists(path):
            missing_adv.append(path)

    if missing_adv:
        print(f"   ❌ Missing: {len(missing_adv)}")
        print(f"\n   Missing adversarial image paths:")
        for path in missing_adv[:10]:  # Show first 10
            print(f"      {path}")
        if len(missing_adv) > 10:
            print(f"      ... and {len(missing_adv) - 10} more")
    else:
        print(f"   ✅ All adversarial images exist on disk")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print(f"Clean images:       {len(clean_paths) - len(missing_clean)}/{len(clean_paths)} exist")
    print(f"Adversarial images: {len(adv_paths) - len(missing_adv)}/{len(adv_paths)} exist")

    total_missing = len(missing_clean) + len(missing_adv)
    total_paths = len(clean_paths) + len(adv_paths)

    if total_missing == 0:
        print(f"\n🎉 SUCCESS: All {total_paths} image paths exist on disk!")
        return True
    else:
        print(f"\n⚠️  WARNING: {total_missing}/{total_paths} image paths are missing from disk")
        return False

if __name__ == "__main__":
    success = verify_image_paths()
    sys.exit(0 if success else 1)
