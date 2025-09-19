#!/usr/bin/env python3
"""
Ground Truth Questions Table Populator

Standalone script to create/overwrite ground_truth_questions table in centralized_pipeline.db
Reads from eval_all.json and filters by processed_images.json

Usage:
    python scripts/utils/ground_truth_populator.py
    python scripts/utils/ground_truth_populator.py --overwrite
"""

import os
import sys
import json

# Add parent directory to path to access centralized_database
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.utils.centralized_database import CentralizedDB

def load_processed_images():
    """Load processed images data from JSON file"""
    processed_images_file = "data/processed_images.json"

    if not os.path.exists(processed_images_file):
        print(f"Error: {processed_images_file} not found")
        return None

    try:
        with open(processed_images_file, 'r') as f:
            processed_images = json.load(f)
        print(f"✅ Loaded processed images data: {sum(len(imgs) for imgs in processed_images.values())} total images")
        return processed_images
    except Exception as e:
        print(f"Error loading processed images: {e}")
        return None

def clear_ground_truth_table(db):
    """Clear existing ground truth questions table"""
    conn = db.get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("DELETE FROM ground_truth_questions")
        conn.commit()
        print("🗑️  Cleared existing ground_truth_questions table")
    except Exception as e:
        print(f"Warning: Could not clear table: {e}")
    finally:
        conn.close()

def populate_ground_truth_questions(overwrite=False):
    """
    Populate ground truth questions table from eval_all.json, filtered by processed_images.json

    Args:
        overwrite (bool): If True, clear existing data before populating

    Returns:
        bool: True if successful, False if failed
    """
    print("🚀 Starting ground truth questions table population...")

    # Check if the main eval_all.json file exists
    all_data_file = 'data/clean/benchmark/eval_all.json'
    if not os.path.exists(all_data_file):
        print(f"❌ Error: Main data file not found at {all_data_file}")
        return False

    # Load processed images
    processed_images = load_processed_images()
    if not processed_images:
        print("❌ Error loading processed_images.json")
        return False

    # Initialize database
    print("🔧 Initializing centralized database...")
    db = CentralizedDB()

    # Clear existing data if overwrite is requested
    if overwrite:
        clear_ground_truth_table(db)

    # List of all task types
    task_types = [
        'chart', 'table', 'road_map', 'dashboard',
        'flowchart', 'relation_graph', 'planar_layout', 'visual_puzzle'
    ]

    try:
        # Read all data from eval_all.json
        print(f"📖 Reading data from {all_data_file}...")
        with open(all_data_file, 'r') as f:
            all_data = [json.loads(line) for line in f]

        print(f"✅ Loaded {len(all_data)} total questions from eval_all.json")

        # Group data by task type - FILTER BY PROCESSED IMAGES ONLY
        task_data = {task: [] for task in task_types}

        # Create set of processed image paths for efficient lookup
        processed_images_set = set()
        for task_images in processed_images.values():
            processed_images_set.update(task_images)

        print(f"🔍 Filtering {len(all_data)} questions to match {len(processed_images_set)} processed images")

        # Filter questions by processed images
        for item in all_data:
            task_type = item.get('type')
            image_path = item.get('image', '')

            # Only include questions for images that are in processed_images.json
            if task_type in task_types and image_path in processed_images_set:
                task_data[task_type].append(item)

        # Save ground truth data to database (FILTERED)
        total_saved = 0
        filtered_count = 0

        print("\n📝 Saving filtered questions to database...")

        for task, data in task_data.items():
            if not data:
                print(f"⚠️  Warning: No ground truth data found for task '{task}' after filtering by processed_images.json")
                continue

            # Save each question to database
            task_saved = 0
            for idx, item in enumerate(data):
                # Make question_id unique by appending index (original has duplicates per image)
                original_qid = item.get('question_id', item.get('id', f"{task}_{item.get('image', '').split('/')[-1]}"))
                unique_question_id = f"{original_qid}_{idx}"

                question_data = {
                    'question_id': unique_question_id,
                    'image': item.get('image', ''),
                    'text': item.get('text', ''),
                    'answer': item.get('answer', ''),
                    'type': item.get('type', task),
                    'markers': item.get('markers', [])
                }

                try:
                    db.save_ground_truth_question(question_data)
                    total_saved += 1
                    task_saved += 1
                except Exception as e:
                    print(f"⚠️  Warning: Failed to save question {unique_question_id}: {e}")

            filtered_count += len(data)
            print(f"   ✅ {task}: {task_saved} questions saved")

        # Print comprehensive summary
        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"   📁 Source file: {all_data_file}")
        print(f"   📄 Original questions in eval_all.json: {len(all_data)}")
        print(f"   🖼️  Target images in processed_images.json: {len(processed_images_set)}")
        print(f"   ✅ Questions matched to processed images: {total_saved}")
        print(f"   📈 Filtering ratio: {total_saved}/{len(all_data)} = {(total_saved/len(all_data)*100):.1f}%")
        print(f"   💾 Database: results/centralized_pipeline.db")
        print(f"   🎯 Table: ground_truth_questions")

        if overwrite:
            print(f"   🔄 Mode: OVERWRITE (cleared existing data)")
        else:
            print(f"   🔄 Mode: APPEND (existing data preserved)")

        print(f"\n🎉 SUCCESS: {total_saved} filtered ground truth questions saved to centralized database")
        return True

    except Exception as e:
        print(f"❌ Error preparing ground truth data: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_population():
    """Verify the population was successful by checking table contents"""
    print("\n🔍 Verifying population results...")

    try:
        db = CentralizedDB()
        conn = db.get_connection()
        cursor = conn.cursor()

        # Get total count
        cursor.execute("SELECT COUNT(*) FROM ground_truth_questions")
        total_count = cursor.fetchone()[0]

        # Get count by task type
        cursor.execute("""
        SELECT question_type, COUNT(*)
        FROM ground_truth_questions
        GROUP BY question_type
        ORDER BY question_type
        """)
        task_counts = cursor.fetchall()

        # Get sample questions
        cursor.execute("SELECT question_id, image_path, question_text, answer FROM ground_truth_questions LIMIT 3")
        sample_questions = cursor.fetchall()

        conn.close()

        print(f"✅ Verification Results:")
        print(f"   Total questions in database: {total_count}")
        print(f"   Questions by task type:")
        for task_type, count in task_counts:
            print(f"     - {task_type}: {count}")

        print(f"\n📋 Sample questions:")
        for i, (qid, img_path, question, answer) in enumerate(sample_questions, 1):
            print(f"   {i}. ID: {qid}")
            print(f"      Image: {img_path}")
            print(f"      Q: {question}")
            print(f"      A: {answer}")
            print()

        return total_count > 0

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Main function with command line argument support"""
    import argparse

    parser = argparse.ArgumentParser(description='Populate ground_truth_questions table in centralized database')
    parser.add_argument('--overwrite', action='store_true',
                       help='Clear existing data before populating (default: append)')
    parser.add_argument('--verify-only', action='store_true',
                       help='Only verify existing data without populating')

    args = parser.parse_args()

    print("="*80)
    print("🏗️  GROUND TRUTH QUESTIONS TABLE POPULATOR")
    print("="*80)

    if args.verify_only:
        print("🔍 Running verification only...")
        success = verify_population()
    else:
        # Populate the table
        success = populate_ground_truth_questions(overwrite=args.overwrite)

        # Verify if population was successful
        if success:
            verify_population()

    if success:
        print("\n🎉 Operation completed successfully!")
        return 0
    else:
        print("\n❌ Operation failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())