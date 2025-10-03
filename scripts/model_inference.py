#!/usr/bin/env python3
"""
Simplified Model Inference Script

Goal: Get VLM answers into inference_results table for:
- Each ground_truth_questions entry (clean images)
- Each attack_executions adversarial image (adversarial images)

Database Flow: ground_truth_questions + attack_executions → VLM → inference_results
"""

import os
import sys
import base64
from datetime import datetime
from tqdm import tqdm

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'local_model'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import required modules
from utils.centralized_database import CentralizedDB
from utils.ground_truth_populator import populate_ground_truth_questions

# GPU Memory management
import torch
import gc

def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def local_image_to_data_url(image_path):
    """Convert local image to data URL for VLM input"""
    from mimetypes import guess_type

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    return f"data:{mime_type};base64,{base64_encoded_data}"

# Engine selection
AVAILABLE_ENGINES = [
    "Qwen25_VL_3B", "Qwen25_VL_7B", "Qwen2_VL_2B",
    "Gemma3_VL_4B", "PaliGemma_VL_3B",
    "DeepSeek1_VL_1pt3B", "DeepSeek1_VL_7B",
    "SmolVLM2_pt25B", "SmolVLM2_pt5B", "SmolVLM2_2pt2B",
    "InternVL3_1B", "InternVL3_2B", "InternVL25_4B",
    "Moondream2_2B", "LLAVA_1pt5_7B", "LLAVA_v1pt6_Mistral_7B"
]

def select_engine():
    """Simple engine selection menu"""
    print("\n" + "="*60)
    print("🔧 ENGINE SELECTION")
    print("="*60)

    print("Select VLM engine(s):")
    for i, engine in enumerate(AVAILABLE_ENGINES, 1):
        print(f"  [{i:2d}] {engine}")
    print(f"  [{len(AVAILABLE_ENGINES)+1:2d}] ALL ENGINES")

    while True:
        try:
            choice = input(f"\nEnter choice (1-{len(AVAILABLE_ENGINES)+1}): ").strip()
            choice = int(choice)

            if 1 <= choice <= len(AVAILABLE_ENGINES):
                return [AVAILABLE_ENGINES[choice-1]]
            elif choice == len(AVAILABLE_ENGINES) + 1:
                return AVAILABLE_ENGINES.copy()
            else:
                print(f"❌ Invalid choice. Please enter 1-{len(AVAILABLE_ENGINES)+1}")
        except ValueError:
            print("❌ Please enter a valid number")

# Removed image type selection - always process both as per instructions

class SimpleInferenceEngine:
    """Simplified inference engine focused on database operations"""

    def __init__(self):
        self.db = CentralizedDB()
        self.vlm_client = None

    def setup_vlm_client(self, engine):
        """Setup VLM client for the selected engine"""
        if engine == "gpt4o":
            from utils.vlm_cloud_client import send_chat_request_azure
            self.vlm_client = send_chat_request_azure
        else:
            from utils.vlm_local_client import send_chat_request
            self.vlm_client = send_chat_request

        print(f"✅ VLM client setup complete for {engine}")

    def load_inference_tasks(self):
        """Load inference tasks from database - ALWAYS processes BOTH clean and adversarial"""
        print("📋 Loading inference tasks from database...")

        # Load ground truth questions
        questions = self.db.get_ground_truth_questions()
        print(f"   📝 Loaded {len(questions)} ground truth questions")

        tasks = []

        # Get list of clean images that have adversarial counterparts
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT image_path FROM attack_executions WHERE success = 1')
        attack_clean_paths = [row[0] for row in cursor.fetchall()]
        conn.close()

        # Convert to relative format for matching with ground truth
        attack_clean_relative = [path.replace("data/clean/", "") for path in attack_clean_paths]
        print(f"   📝 Clean images with adversarial counterparts: {len(attack_clean_relative)}")

        # CLEAN IMAGE TASKS: Only process questions for images that have adversarial counterparts
        print("   🖼️  Processing clean images (only those with adversarial counterparts)...")
        for question in questions:
            if question['image'] in attack_clean_relative:
                image_path = f"data/clean/{question['image']}"
                if os.path.exists(image_path):
                    tasks.append({
                        'type': 'clean',
                        'image_path': image_path,  # image_path for CLEAN
                        'question_id': question['question_id'],
                        'question_text': question['text'],
                        'ground_truth': question['answer'],
                        'question_type': question['type'],
                        'original_image_path': question['image'],
                        'adversarial': False,
                        'attack_type': 'original',        # For clean images
                        'attack_name': 'original',        # For clean images
                        'epsilon_target': 1.0,           # Target epsilon for clean images (no perturbation)
                        'epsilon_actual': 1.0,           # Actual epsilon for clean images (no perturbation)
                        'task_type': question['type']    # From ground truth questions
                    })

        # ADVERSARIAL IMAGE TASKS: Process all adversarial images with mapped questions
        print("   🎯 Processing adversarial images...")
        conn = self.db.get_connection()
        cursor = conn.cursor()

        # Get adversarial images with ALL required fields from attack_executions
        cursor.execute('''
            SELECT DISTINCT ae.adversarial_image_path, ae.image_path, ae.attack_name,
                           ae.epsilon_l_inf, ae.attack_category, ae.epsilon_target, ae.task_type
            FROM attack_executions ae
            WHERE ae.success = 1
        ''')
        adv_images = cursor.fetchall()
        conn.close()

        print(f"   🎯 Found {len(adv_images)} adversarial images")

        for adv_path, clean_path, attack_name, epsilon_l_inf, attack_category, epsilon_target, task_type in adv_images:
            if os.path.exists(adv_path):
                # Find questions for this image
                clean_image_rel = clean_path.replace("data/clean/", "")
                matching_questions = [q for q in questions if q['image'] == clean_image_rel]

                for question in matching_questions:
                    tasks.append({
                        'type': 'adversarial',
                        'image_path': adv_path,  # adversarial_image_path for ADVERSARIAL
                        'clean_image_path': clean_path,  # image_path for reference
                        'question_id': question['question_id'],
                        'question_text': question['text'],
                        'ground_truth': question['answer'],
                        'question_type': question['type'],
                        'original_image_path': question['image'],
                        'adversarial': True,
                        'attack_type': attack_category,  # attack_category from attack_executions
                        'attack_name': attack_name,      # attack_name from attack_executions
                        'epsilon_target': epsilon_target,   # epsilon_target from attack_executions (goal)
                        'epsilon_actual': epsilon_l_inf or 0.0,  # epsilon_l_inf from attack_executions (actual achieved)
                        'task_type': task_type           # task_type from attack_executions
                    })

        print(f"🎯 Total inference tasks: {len(tasks)} (clean + adversarial)")
        return tasks

    def run_inference(self, engine, tasks):
        """Run VLM inference on all tasks"""
        print(f"\n🚀 Starting inference with {engine}")
        print(f"📊 Processing {len(tasks)} tasks")

        self.setup_vlm_client(engine)
        results = []

        with tqdm(total=len(tasks), desc=f"Processing {engine}") as pbar:
            for i, task in enumerate(tasks):
                try:
                    # Check if result already exists
                    result_id = self.generate_result_id(engine, task)
                    if self.result_exists(result_id):
                        pbar.set_description(f"Skipping existing {engine}")
                        pbar.update(1)
                        continue

                    # Convert image to data URL
                    image_url = local_image_to_data_url(task['image_path'])

                    # Prepare VLM message
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": task['question_text']},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }]

                    # Call VLM with metrics collection
                    response, _, metrics = self.vlm_client(
                        message_text=messages,
                        engine=engine,
                        temp=0.2,
                        max_new_token=512,
                        sample_n=1,
                        collect_metrics=True
                    )

                    # Create result with performance metrics
                    result = {
                        'result_id': result_id,
                        'question_id': task['question_id'],
                        'prompt': task['question_text'],
                        'model_response': response,
                        'ground_truth': task['ground_truth'],
                        'question_type': task['question_type'],
                        'model_id': engine,
                        'adversarial': task['adversarial'],
                        'task': task['question_type'],
                        'attack_type': task['attack_type'],
                        'epsilon_target': task['epsilon_target'],    # Target epsilon (goal)
                        'epsilon_actual': task['epsilon_actual'],    # Actual achieved epsilon
                        'inference_image_path': task['image_path'],  # Image fed to model (clean or adversarial)
                        'clean_image_path': f"data/clean/{task['original_image_path']}",  # Original unperturbed reference
                        'timestamp': datetime.now().isoformat() + 'Z',
                        # Performance metrics
                        'inference_time_seconds': metrics.get('inference_time_seconds'),
                        'gpu_before_mb': metrics.get('gpu_memory', {}).get('before_inference_mb'),
                        'gpu_after_mb': metrics.get('gpu_memory', {}).get('after_inference_mb'),
                        'gpu_peak_mb': metrics.get('gpu_memory', {}).get('peak_mb'),
                        'gpu_reserved_mb': metrics.get('gpu_memory', {}).get('reserved_mb'),
                        'total_gpu_mb': metrics.get('gpu_memory', {}).get('total_gpu_mb'),
                        'cpu_before_mb': metrics.get('cpu_memory', {}).get('before_inference_mb'),
                        'cpu_after_mb': metrics.get('cpu_memory', {}).get('after_inference_mb'),
                        'was_cached': metrics.get('model_loading', {}).get('was_cached'),
                        'loading_time_seconds': metrics.get('model_loading', {}).get('loading_time_seconds'),
                        'batch_size': metrics.get('batch_info', {}).get('batch_size'),
                        'position_in_batch': metrics.get('batch_info', {}).get('position_in_batch'),
                        'total_batch_questions': metrics.get('batch_info', {}).get('total_batch_questions')
                    }

                    # Save to database
                    self.save_result(result)
                    results.append(result)

                    pbar.set_description(f"✅ {engine} ({len(results)} saved)")

                except Exception as e:
                    print(f"\n⚠️ Error processing task {i}: {e}")
                    pbar.set_description(f"⚠️ {engine} (error)")

                finally:
                    pbar.update(1)

                    # Cleanup every 10 tasks
                    if i % 10 == 0:
                        cleanup_gpu_memory()

        print(f"✅ {engine} completed: {len(results)} results saved")
        return results

    def generate_result_id(self, engine, task):
        """Generate deterministic result ID using your specification:
        engine + "clean/adversarial" + attack_type + attack_name + epsilon_target + task_type + image_path + question_id
        """
        import hashlib

        # Extract components as per your specification
        clean_or_adversarial = "adversarial" if task['adversarial'] else "clean"
        attack_type = task['attack_type']      # "attack_category" from attack_executions (or "original" for clean)
        attack_name = task['attack_name']      # "attack_name" from attack_executions (or "original" for clean)
        epsilon_target = str(task['epsilon_target']) # "epsilon_target" from attack_executions (goal value)
        task_type = task['task_type']          # "task_type" from attack_executions (or question type for clean)
        image_path = task['image_path']        # Full path - "image_path" for clean or "adversarial_image_path" for adversarial
        question_id = task['question_id']

        # Create comprehensive combination as per your specification
        content = f"{engine}_{clean_or_adversarial}_{attack_type}_{attack_name}_{epsilon_target}_{task_type}_{image_path}_{question_id}"

        # Generate deterministic hash
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def result_exists(self, result_id):
        """Check if result already exists in database"""
        conn = self.db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM inference_results WHERE result_id = ?", (result_id,))
        exists = cursor.fetchone() is not None
        conn.close()
        return exists

    def save_result(self, result):
        """Save inference result to database"""
        conn = self.db.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO inference_results (
                result_id, question_id, prompt, model_response, ground_truth, question_type,
                answer_id, markers, model_id, adversarial, task, attack_type,
                epsilon_target, epsilon_l_inf, inference_image_path, clean_image_path, timestamp,
                inference_time_seconds, gpu_before_mb, gpu_after_mb, gpu_peak_mb, gpu_reserved_mb,
                total_gpu_mb, cpu_before_mb, cpu_after_mb, was_cached, loading_time_seconds,
                batch_size, position_in_batch, total_batch_questions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['result_id'], result['question_id'], result['prompt'],
            result['model_response'], result['ground_truth'], result['question_type'],
            '', '[]', result['model_id'], result['adversarial'], result['task'],
            result['attack_type'], result['epsilon_target'], result['epsilon_actual'],
            result['inference_image_path'], result['clean_image_path'], result['timestamp'],
            result.get('inference_time_seconds'), result.get('gpu_before_mb'),
            result.get('gpu_after_mb'), result.get('gpu_peak_mb'), result.get('gpu_reserved_mb'),
            result.get('total_gpu_mb'), result.get('cpu_before_mb'), result.get('cpu_after_mb'),
            result.get('was_cached'), result.get('loading_time_seconds'),
            result.get('batch_size'), result.get('position_in_batch'), result.get('total_batch_questions')
        ))

        conn.commit()
        conn.close()

def main():
    """Main execution function"""
    print("="*80)
    print("🚀 SIMPLIFIED VLM INFERENCE ENGINE")
    print("="*80)
    print("Goal: Populate inference_results table with VLM answers")
    print("Source: ground_truth_questions + attack_executions tables")

    # Ensure ground truth data exists
    print("\n🔍 Checking ground truth data...")
    if not populate_ground_truth_questions(overwrite=False):
        print("❌ Failed to ensure ground truth data. Exiting.")
        return

    # User selections
    engines = select_engine()

    print(f"\n📋 Configuration:")
    print(f"   🔧 Engines: {', '.join(engines)}")
    print(f"   🖼️  Processing: Both clean and adversarial images (as per instructions)")

    # Initialize inference engine
    inference_engine = SimpleInferenceEngine()

    # Load tasks
    tasks = inference_engine.load_inference_tasks()
    if not tasks:
        print("❌ No inference tasks found. Exiting.")
        return

    # Run inference for each engine
    all_results = {}
    for engine in engines:
        print(f"\n{'='*60}")
        results = inference_engine.run_inference(engine, tasks)
        all_results[engine] = results
        cleanup_gpu_memory()

    # Final summary
    print(f"\n{'='*80}")
    print("🎉 INFERENCE COMPLETE")
    print("="*80)
    total_results = sum(len(results) for results in all_results.values())
    print(f"📊 Total results saved: {total_results}")
    for engine, results in all_results.items():
        print(f"   {engine}: {len(results)} results")
    print(f"💾 Database: results/centralized_pipeline.db → inference_results table")

if __name__ == "__main__":
    main()