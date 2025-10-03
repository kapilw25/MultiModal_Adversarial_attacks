#!/usr/bin/env python3
"""
Simple Sync: Disk → Database (No Calculation)

Scans disk for existing adversarial images and creates database entries with markers:
- epsilon_l_inf = -1 (post-sync marker)
- execution_time_seconds = -1 (post-sync marker)

This distinguishes synced entries from real attack runs.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.utils.centralized_database import CentralizedDB

# Epsilon mapping
EPSILON_MAP = {
    'eps_0016': {'value': 4/255, 'level': 'minimal'},    # ≈ 0.0157
    'eps_0031': {'value': 8/255, 'level': 'standard'},   # ≈ 0.0314
    'eps_0063': {'value': 16/255, 'level': 'moderate'}   # ≈ 0.0627
}

def extract_metadata_from_path(adv_image_path: str):
    """
    Extract metadata from adversarial image path

    Example: data/adversarial/whitebox/fgsm/eps_0031/chart/20231107140031466140.png
    """
    parts = adv_image_path.split('/')

    # data/adversarial/{category}/{attack}/{epsilon}/{task}/{filename}
    if len(parts) >= 6:
        attack_category = parts[2]  # whitebox or blackbox
        attack_type = parts[3]      # fgsm, pgd, etc.
        epsilon_str = parts[4]      # eps_0016, eps_0031, eps_0063
        task_type = parts[5]        # chart, table, etc.
        filename = parts[6]         # image filename

        # Get epsilon metadata
        eps_data = EPSILON_MAP.get(epsilon_str, {'value': 0.0, 'level': 'unknown'})

        # Construct clean image path
        clean_image_path = f"data/clean/{task_type}/{filename}"

        return {
            'attack_category': attack_category,
            'attack_type': attack_type,
            'epsilon_str': epsilon_str,
            'epsilon_level': eps_data['level'],
            'epsilon_target': eps_data['value'],
            'task_type': task_type,
            'clean_image_path': clean_image_path,
            'adversarial_image_path': adv_image_path
        }

    return None

def simple_sync():
    """Scan disk and sync to database with markers (no calculation)"""

    db = CentralizedDB()
    adversarial_dir = Path("data/adversarial")

    if not adversarial_dir.exists():
        print(f"❌ Adversarial directory not found: {adversarial_dir}")
        return

    # Find all adversarial images
    all_images = list(adversarial_dir.glob("**/*.png"))
    print(f"📂 Found {len(all_images)} adversarial images on disk")

    # Get existing database entries
    conn = db.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT adversarial_image_path FROM attack_executions")
    existing_paths = set(row[0] for row in cursor.fetchall())
    conn.close()

    print(f"💾 Found {len(existing_paths)} existing database entries")

    # Find missing entries
    missing_images = []
    for img_path in all_images:
        rel_path = str(img_path)
        if rel_path not in existing_paths:
            missing_images.append(rel_path)

    print(f"⚠️  Missing from database: {len(missing_images)} images")

    if not missing_images:
        print("✅ Database is in sync with disk!")
        return

    # Show what will be created
    print(f"\n{'='*60}")
    print(f"Will create {len(missing_images)} database entries")
    print(f"Method: SIMPLE SYNC (no calculation)")
    print(f"Markers:")
    print(f"  • epsilon_l_inf = -1.0 (post-sync marker)")
    print(f"  • execution_time_seconds = -1 (post-sync marker)")
    print(f"Estimated time: <1 second")
    print(f"{'='*60}")

    response = input("Continue? [Y/n]: ").strip().lower()
    if response and response not in ['y', 'yes']:
        print("❌ Cancelled by user")
        return

    # Create database entries for missing images
    created_count = 0
    skipped_count = 0

    for adv_path in missing_images:
        metadata = extract_metadata_from_path(adv_path)

        if not metadata:
            print(f"⚠️  Skipped (invalid path format): {adv_path}")
            skipped_count += 1
            continue

        # Create database entry with markers
        db.insert_attack_result({
            'adversarial_image_path': metadata['adversarial_image_path'],
            'attack_type': metadata['attack_type'],
            'attack_category': metadata['attack_category'],
            'task_type': metadata['task_type'],
            'image_path': metadata['clean_image_path'],
            'epsilon_level': metadata['epsilon_level'],
            'epsilon_target': metadata['epsilon_target'],
            'epsilon_achieved': -1.0,  # POST-SYNC MARKER
            'execution_time': -1,      # POST-SYNC MARKER
            'success': True,
            'timestamp': datetime.now().isoformat()
        })

        created_count += 1

        if created_count % 50 == 0:
            print(f"  ✅ Created {created_count}/{len(missing_images)} entries...")

    print(f"\n{'='*60}")
    print(f"✅ Sync complete!")
    print(f"  Created: {created_count} entries")
    print(f"  Skipped: {skipped_count} entries")
    print(f"{'='*60}")

    # Verify final count
    conn = db.get_connection()
    cursor = conn.cursor()
    final_count = cursor.execute("SELECT COUNT(*) FROM attack_executions").fetchone()[0]
    unique_count = cursor.execute("SELECT COUNT(DISTINCT adversarial_image_path) FROM attack_executions").fetchone()[0]

    # Count synced vs real entries
    synced_count = cursor.execute("SELECT COUNT(*) FROM attack_executions WHERE execution_time_seconds = -1").fetchone()[0]
    real_count = cursor.execute("SELECT COUNT(*) FROM attack_executions WHERE execution_time_seconds > 0").fetchone()[0]

    conn.close()

    print(f"\n📊 Database statistics:")
    print(f"  Total entries: {final_count}")
    print(f"  Unique paths: {unique_count}")
    print(f"  Images on disk: {len(all_images)}")
    print()
    print(f"  Real attack runs (exec_time > 0):  {real_count}")
    print(f"  Post-synced (exec_time = -1):      {synced_count}")

    if unique_count == len(all_images):
        print(f"\n🎉 Database is now in sync with disk!")
        print(f"✅ Post-synced entries have epsilon_l_inf=-1 and execution_time_seconds=-1")
    else:
        print(f"\n⚠️  Still {len(all_images) - unique_count} images missing from database")

if __name__ == "__main__":
    simple_sync()
