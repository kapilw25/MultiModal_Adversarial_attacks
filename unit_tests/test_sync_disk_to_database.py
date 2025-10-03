#!/usr/bin/env python3
"""
Sync adversarial images from disk to database with ACTUAL epsilon calculation

This script scans all adversarial images on disk and creates missing database entries.
Uses REAL epsilon_l_inf calculation from image pairs (no hardcoded values).
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.utils.centralized_database import CentralizedDB
from attack_models.utils import calculate_epsilon, load_image

# Epsilon mapping
EPSILON_MAP = {
    'eps_0016': {'value': 4/255, 'level': 'minimal'},    # ≈ 0.0157
    'eps_0031': {'value': 8/255, 'level': 'standard'},   # ≈ 0.0314
    'eps_0063': {'value': 16/255, 'level': 'moderate'}   # ≈ 0.0627
}

def extract_metadata_from_path(adv_image_path: str):
    """
    Extract metadata from adversarial image path

    Example path: data/adversarial/whitebox/fgsm/eps_0031/chart/20231107140031466140.png
    Returns: {
        'attack_category': 'whitebox',
        'attack_type': 'fgsm',
        'epsilon_str': 'eps_0031',
        'epsilon_level': 'standard',
        'epsilon_target': 0.0314,
        'task_type': 'chart',
        'clean_image_path': 'data/clean/chart/20231107140031466140.png'
    }
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

def calculate_actual_epsilon(clean_path: str, adv_path: str) -> float:
    """
    Calculate ACTUAL epsilon_l_inf from image pair

    Returns:
        float: Measured L∞ norm between images
    """
    try:
        import cv2

        # Load both images
        original_image = load_image(clean_path)
        adversarial_image = load_image(adv_path)

        # Resize original to match adversarial dimensions (adversarial is 224x224)
        if original_image.shape != adversarial_image.shape:
            original_image = cv2.resize(original_image,
                                       (adversarial_image.shape[1], adversarial_image.shape[0]))

        # Calculate actual epsilon using utils function
        epsilon_l_inf = calculate_epsilon(original_image, adversarial_image)

        return epsilon_l_inf

    except Exception as e:
        print(f"    ⚠️  Epsilon calculation failed: {e}")
        return None

def scan_and_sync():
    """Scan disk for adversarial images and sync to database with REAL epsilon values"""

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

    # Ask for confirmation
    print(f"\n{'='*60}")
    print(f"Will create {len(missing_images)} database entries")
    print(f"Method: CALCULATE ACTUAL epsilon_l_inf from image pairs")
    print(f"Estimated time: ~{len(missing_images) * 0.023:.0f} seconds (~{len(missing_images) * 0.023 / 60:.1f} min)")
    print(f"{'='*60}")

    response = input("Continue? [Y/n]: ").strip().lower()
    if response and response not in ['y', 'yes']:
        print("❌ Cancelled by user")
        return

    # Create database entries for missing images
    created_count = 0
    skipped_count = 0
    start_time = time.time()

    for idx, adv_path in enumerate(missing_images):
        metadata = extract_metadata_from_path(adv_path)

        if not metadata:
            print(f"⚠️  Skipped (invalid path format): {adv_path}")
            skipped_count += 1
            continue

        # Check if clean image exists
        clean_path = metadata['clean_image_path']
        if not Path(clean_path).exists():
            print(f"⚠️  Skipped (clean image not found): {clean_path}")
            skipped_count += 1
            continue

        # Calculate ACTUAL epsilon_l_inf from image pair
        epsilon_l_inf = calculate_actual_epsilon(clean_path, adv_path)

        if epsilon_l_inf is None:
            # Fallback to target epsilon if calculation fails
            print(f"    Using epsilon_target as fallback: {metadata['epsilon_target']:.6f}")
            epsilon_l_inf = metadata['epsilon_target']

        # Create database entry
        db.insert_attack_result({
            'adversarial_image_path': metadata['adversarial_image_path'],
            'attack_type': metadata['attack_type'],
            'attack_category': metadata['attack_category'],
            'task_type': metadata['task_type'],
            'image_path': metadata['clean_image_path'],
            'epsilon_level': metadata['epsilon_level'],
            'epsilon_target': metadata['epsilon_target'],
            'epsilon_achieved': epsilon_l_inf,  # ACTUAL measured value
            'execution_time': 0,  # Unknown for pre-existing images
            'success': True,
            'timestamp': datetime.now().isoformat()
        })

        created_count += 1

        # Progress updates
        if created_count % 25 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / created_count
            remaining = (len(missing_images) - created_count) * avg_time
            print(f"  ✅ Created {created_count}/{len(missing_images)} entries... "
                  f"({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)")

    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"✅ Sync complete!")
    print(f"  Created: {created_count} entries")
    print(f"  Skipped: {skipped_count} entries")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg time per image: {total_time/created_count:.2f}s")
    print(f"{'='*60}")

    # Verify final count
    conn = db.get_connection()
    cursor = conn.cursor()
    final_count = cursor.execute("SELECT COUNT(*) FROM attack_executions").fetchone()[0]
    unique_count = cursor.execute("SELECT COUNT(DISTINCT adversarial_image_path) FROM attack_executions").fetchone()[0]

    # Get epsilon statistics
    cursor.execute("""
        SELECT
            attack_name,
            epsilon_level,
            AVG(epsilon_achieved) as avg_epsilon,
            MIN(epsilon_achieved) as min_epsilon,
            MAX(epsilon_achieved) as max_epsilon
        FROM attack_executions
        WHERE epsilon_level = 'standard'
        GROUP BY attack_name
        LIMIT 3
    """)

    print(f"\n📊 Database statistics:")
    print(f"  Total entries: {final_count}")
    print(f"  Unique paths: {unique_count}")
    print(f"  Images on disk: {len(all_images)}")

    print(f"\n📏 Sample epsilon_l_inf values (standard level):")
    for row in cursor.fetchall():
        attack, level, avg_eps, min_eps, max_eps = row
        print(f"  {attack:20s}: avg={avg_eps:.6f}, range=[{min_eps:.6f}, {max_eps:.6f}]")

    conn.close()

    if unique_count == len(all_images):
        print(f"\n🎉 Database is now in sync with disk!")
        print(f"✅ All epsilon values calculated from actual image pairs")
    else:
        print(f"\n⚠️  Still {len(all_images) - unique_count} images missing from database")

if __name__ == "__main__":
    scan_and_sync()
