#!/usr/bin/env python3
"""
vLLM Batch Model Inference Script

Goal: Get VLM answers into inference_results table for:
- Each ground_truth_questions entry (clean images)
- Each attack_executions adversarial image (adversarial images)

Key Difference from model_inference_sequential.py:
- Uses vLLM continuous batching for 90-95% GPU utilization
- Does NOT use local_model/ directory (vLLM loads directly from HuggingFace)
- Processes multiple images simultaneously instead of one-at-a-time

Database Flow: ground_truth_questions + attack_executions → vLLM → inference_results
"""

import os
import sys
import time
from datetime import datetime
from tqdm import tqdm

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import required modules
from utils.centralized_database import CentralizedDB
from utils.ground_truth_populator import populate_ground_truth_questions
from utils.text_cleaner import clean_vlm_response  # CRITICAL: Clean responses like sequential version

# GPU Memory management
import torch
import gc

# ============================================================================
# vLLM AVAILABILITY CHECK
# ============================================================================
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    print("⚠️  vLLM not installed. Install with: pip install vllm>=0.6.0")

# ============================================================================
# vLLM MODEL REGISTRY - 100% vLLM Compatible Models
# Maps engine names to HuggingFace model paths
# ============================================================================
VLLM_MODEL_PATHS = {
    # Qwen models (1-3)
    "Qwen25_VL_3B": "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen25_VL_7B": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen2_VL_2B": "Qwen/Qwen2-VL-2B-Instruct",

    # LLaVA models (4-5)
    "LLAVA_1pt5_7B": "llava-hf/llava-1.5-7b-hf",
    "LLAVA_v1pt6_Mistral_7B": "llava-hf/llava-v1.6-mistral-7b-hf",

    # InternVL models (6-8)
    "InternVL3_1B": "OpenGVLab/InternVL3-1B",
    "InternVL3_2B": "OpenGVLab/InternVL3-2B",
    "InternVL25_4B": "OpenGVLab/InternVL2.5-4B",

    # PaliGemma (9)
    "PaliGemma_VL_3B": "google/paligemma-3b-mix-224",

    # New vLLM-compatible replacements (10-16)
    "MiniCPM_V_2pt5": "openbmb/MiniCPM-V-2_5",              # Replaces Moondream2-2B
    "DeepSeek_VL2_Small": "deepseek-ai/deepseek-vl2-small", # Replaces DeepSeek-VL-7B
    "H2OVL_Mississippi_2B": "h2oai/h2ovl-mississippi-2b",   # Replaces SmolVLM2-500M
    "Phi35_Vision": "microsoft/Phi-3.5-vision-instruct",    # New addition
    "Mono_InternVL_2B": "OpenGVLab/Mono-InternVL-2B",       # Replaces SmolVLM2-2.2B
    "MiniCPM_V_4": "openbmb/MiniCPM-V-4",                   # Replaces DeepSeek-VL-1.3B
    "Molmo_7B": "allenai/Molmo-7B-D-0924",                  # Replaces SmolVLM2-256M
}

# Available engines for menu selection
AVAILABLE_ENGINES = list(VLLM_MODEL_PATHS.keys())

# ============================================================================
# vLLM PROMPT FORMATTERS - Model-specific prompt templates
# Based on: https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language.py
# CRITICAL: Must match answer format suffix from local_model/models/*.py for consistency
# ============================================================================

# Answer format suffix used by some models in sequential version
# This ensures consistent response format for evaluation
ANSWER_FORMAT_SUFFIX = " Answer format (do not generate any other content): The answer is <answer>."

# Models that use answer format suffix in sequential version (from local_model/models/*.py)
MODELS_WITH_ANSWER_FORMAT = {
    "Qwen25_VL_3B", "Qwen25_VL_7B", "Qwen2_VL_2B",  # Qwen family
    "PaliGemma_VL_3B",                               # PaliGemma
    "DeepSeek_VL2_Small",                            # DeepSeek (replacing DeepSeek1)
    "MiniCPM_V_2pt5", "MiniCPM_V_4",                 # MiniCPM (new, assume needs format)
    "Phi35_Vision",                                  # Phi (new, assume needs format)
}

# Models that do NOT use answer format suffix (let model respond naturally)
MODELS_WITHOUT_ANSWER_FORMAT = {
    "LLAVA_1pt5_7B", "LLAVA_v1pt6_Mistral_7B",       # LLaVA family
    "InternVL3_1B", "InternVL3_2B", "InternVL25_4B", # InternVL family
    "Mono_InternVL_2B", "H2OVL_Mississippi_2B",      # InternVL variants
    "Molmo_7B",                                      # Molmo
}


def format_prompt_for_vllm(engine: str, question: str) -> str:
    """
    Format prompt according to vLLM's model-specific requirements.
    Each model family has its own prompt template.

    CRITICAL: Adds answer format suffix for models that use it in sequential version
    to ensure consistent response format for evaluation.

    Args:
        engine: Engine name from VLLM_MODEL_PATHS
        question: The question text to ask about the image

    Returns:
        Formatted prompt string for vLLM
    """

    # Determine if this model needs answer format suffix
    needs_answer_format = engine in MODELS_WITH_ANSWER_FORMAT

    # Apply answer format suffix if needed
    if needs_answer_format:
        question_with_format = question + ANSWER_FORMAT_SUFFIX
    else:
        question_with_format = question

    # QWEN FAMILY (Qwen2-VL, Qwen2.5-VL) - HAS answer format
    if engine in ["Qwen25_VL_3B", "Qwen25_VL_7B", "Qwen2_VL_2B"]:
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{question_with_format}<|im_end|>\n<|im_start|>assistant\n"
        )

    # LLAVA-1.5 - NO answer format
    elif engine == "LLAVA_1pt5_7B":
        return f"USER: <image>\n{question_with_format}\nASSISTANT:"

    # LLAVA-1.6/NeXT (Mistral-based) - NO answer format
    elif engine == "LLAVA_v1pt6_Mistral_7B":
        return f"[INST] <image>\n{question_with_format} [/INST]"

    # INTERNVL FAMILY (InternVL3, InternVL2.5, Mono-InternVL, H2OVL) - NO answer format
    elif engine in ["InternVL3_1B", "InternVL3_2B", "InternVL25_4B",
                    "Mono_InternVL_2B", "H2OVL_Mississippi_2B"]:
        return f"<image>\n{question_with_format}"

    # PALIGEMMA (uses "vqa" task prefix) - HAS answer format
    elif engine == "PaliGemma_VL_3B":
        return f"vqa {question_with_format}"

    # PHI-3.5-VISION - HAS answer format (new model, using format for consistency)
    elif engine == "Phi35_Vision":
        return f"<|user|>\n<|image_1|>\n{question_with_format}<|end|>\n<|assistant|>\n"

    # MINICPM FAMILY - HAS answer format (new model, using format for consistency)
    elif engine in ["MiniCPM_V_2pt5", "MiniCPM_V_4"]:
        return f"(<image>./</image>)\n{question_with_format}"

    # DEEPSEEK-VL2 - HAS answer format (replacing DeepSeek1 which had it)
    elif engine == "DeepSeek_VL2_Small":
        return f"<|User|>: <image>\n{question_with_format}\n\n<|Assistant|>:"

    # MOLMO - NO answer format
    elif engine == "Molmo_7B":
        return f"<|im_start|>user <image>\n{question_with_format}<|im_end|><|im_start|>assistant\n"

    # FALLBACK (generic format) - NO answer format
    else:
        print(f"⚠️  Warning: No specific prompt format for {engine}, using generic <image>\\n format")
        return f"<image>\n{question_with_format}"


def get_sampling_params_for_engine(engine: str):
    """
    Get model-specific sampling parameters to match sequential version behavior.
    Based on generation configs in local_model/models/*.py files.

    Args:
        engine: Engine name from VLLM_MODEL_PATHS

    Returns:
        SamplingParams configured for the specific model
    """
    # Model-specific parameters from local_model/models/*.py
    # Format: (max_tokens, temperature, top_p, do_sample_equivalent)

    params_map = {
        # Qwen models: max_new_tokens=128, temperature=0.3, top_p=0.95, do_sample=True
        "Qwen25_VL_3B": (128, 0.3, 0.95),
        "Qwen25_VL_7B": (128, 0.3, 0.95),
        "Qwen2_VL_2B": (128, 0.3, 0.95),

        # LLaVA models: max_new_tokens=200, do_sample=False (temperature=0 for greedy)
        "LLAVA_1pt5_7B": (200, 0.0, 1.0),
        "LLAVA_v1pt6_Mistral_7B": (200, 0.0, 1.0),

        # InternVL models: max_new_tokens=1024, do_sample=True
        "InternVL3_1B": (1024, 0.2, 0.95),
        "InternVL3_2B": (1024, 0.2, 0.95),
        "InternVL25_4B": (1024, 0.2, 0.95),
        "Mono_InternVL_2B": (1024, 0.2, 0.95),
        "H2OVL_Mississippi_2B": (1024, 0.2, 0.95),

        # PaliGemma: max_new_tokens=100, do_sample=False (temperature=0 for greedy)
        "PaliGemma_VL_3B": (100, 0.0, 1.0),

        # New models - using reasonable defaults
        "MiniCPM_V_2pt5": (512, 0.2, 0.95),
        "MiniCPM_V_4": (512, 0.2, 0.95),
        "DeepSeek_VL2_Small": (512, 0.2, 0.95),
        "Phi35_Vision": (512, 0.2, 0.95),
        "Molmo_7B": (512, 0.2, 0.95),
    }

    # Get params or use defaults
    max_tokens, temperature, top_p = params_map.get(engine, (512, 0.2, 0.95))

    return SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )


def cleanup_gpu_memory():
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def select_engine():
    """Interactive engine selection menu"""
    print("\n" + "="*60)
    print("🔧 vLLM ENGINE SELECTION")
    print("="*60)
    print("⚡ All engines use vLLM batch inference (90-95% GPU utilization)")
    print()

    print("Select VLM engine(s):")
    for i, engine in enumerate(AVAILABLE_ENGINES, 1):
        hf_path = VLLM_MODEL_PATHS[engine]
        print(f"  [{i:2d}] {engine:<25} → {hf_path}")
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


class VLLMInferenceEngine:
    """vLLM-based inference engine with continuous batching"""

    def __init__(self):
        self.db = CentralizedDB()
        self.current_llm = None
        self.current_engine = None

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
                        'image_path': image_path,
                        'question_id': question['question_id'],
                        'question_text': question['text'],
                        'ground_truth': question['answer'],
                        'question_type': question['type'],
                        'original_image_path': question['image'],
                        'adversarial': False,
                        'attack_type': 'original',
                        'attack_name': 'original',
                        'epsilon_target': 1.0,
                        'epsilon_actual': 1.0,
                        'task_type': question['type']
                    })

        # ADVERSARIAL IMAGE TASKS: Process all adversarial images with mapped questions
        print("   🎯 Processing adversarial images...")
        conn = self.db.get_connection()
        cursor = conn.cursor()

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
                clean_image_rel = clean_path.replace("data/clean/", "")
                matching_questions = [q for q in questions if q['image'] == clean_image_rel]

                for question in matching_questions:
                    tasks.append({
                        'type': 'adversarial',
                        'image_path': adv_path,
                        'clean_image_path': clean_path,
                        'question_id': question['question_id'],
                        'question_text': question['text'],
                        'ground_truth': question['answer'],
                        'question_type': question['type'],
                        'original_image_path': question['image'],
                        'adversarial': True,
                        'attack_type': attack_category,
                        'attack_name': attack_name,
                        'epsilon_target': epsilon_target,
                        'epsilon_actual': epsilon_l_inf or 0.0,
                        'task_type': task_type
                    })

        print(f"🎯 Total inference tasks: {len(tasks)} (clean + adversarial)")
        return tasks

    def initialize_vllm_engine(self, engine, gpu_memory_utilization=0.85):
        """Initialize vLLM engine for a specific model"""
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not available. Install with: pip install vllm>=0.6.0")

        if engine not in VLLM_MODEL_PATHS:
            raise ValueError(f"Engine {engine} not in VLLM_MODEL_PATHS registry")

        # Unload previous model if different
        if self.current_llm is not None and self.current_engine != engine:
            print(f"🔄 Unloading previous model: {self.current_engine}")
            del self.current_llm
            self.current_llm = None
            cleanup_gpu_memory()

        if self.current_llm is None or self.current_engine != engine:
            model_path = VLLM_MODEL_PATHS[engine]
            print(f"🚀 Loading vLLM engine: {engine}")
            print(f"   HuggingFace path: {model_path}")
            print(f"   GPU memory utilization: {gpu_memory_utilization*100:.0f}%")

            load_start = time.time()

            self.current_llm = LLM(
                model=model_path,
                trust_remote_code=True,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=4096,
                dtype="half",
                limit_mm_per_prompt={"image": 1}  # Enable vision support
            )

            load_time = time.time() - load_start
            print(f"✅ Model loaded in {load_time:.2f}s")
            self.current_engine = engine

        return self.current_llm

    def run_inference(self, engine, tasks):
        """Run VLM inference on all tasks using vLLM batch processing"""
        print(f"\n🚀 Starting vLLM batch inference with {engine}")
        print(f"📊 Total tasks: {len(tasks)}")

        # Initialize vLLM engine
        llm = self.initialize_vllm_engine(engine)

        # Filter tasks that don't already exist
        new_tasks = []
        for task in tasks:
            result_id = self.generate_result_id(engine, task)
            if not self.result_exists(result_id):
                task['result_id'] = result_id
                new_tasks.append(task)

        skipped = len(tasks) - len(new_tasks)
        if skipped > 0:
            print(f"⏭️  Skipped {skipped} existing results")

        if not new_tasks:
            print("✅ All tasks already completed")
            return []

        print(f"🔄 Processing {len(new_tasks)} new tasks with vLLM batch inference")

        # Show prompt format being used
        sample_prompt = format_prompt_for_vllm(engine, "<QUESTION>")
        print(f"📝 Prompt format for {engine}:")
        print(f"   {sample_prompt[:80]}..." if len(sample_prompt) > 80 else f"   {sample_prompt}")

        # Prepare prompts for vLLM multimodal (model-specific formatting)
        prompts = []
        for task in new_tasks:
            formatted_prompt = format_prompt_for_vllm(engine, task['question_text'])
            prompts.append({
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": task['image_path']}
            })

        # Get model-specific sampling parameters (matches sequential version behavior)
        sampling_params = get_sampling_params_for_engine(engine)
        print(f"⚙️  Sampling params: max_tokens={sampling_params.max_tokens}, "
              f"temp={sampling_params.temperature}, top_p={sampling_params.top_p}")

        # Run batch inference
        print("⚡ Running vLLM continuous batching...")
        inference_start = time.time()

        try:
            outputs = llm.generate(prompts, sampling_params)
            total_inference_time = time.time() - inference_start
            avg_time_per_task = total_inference_time / len(new_tasks) if new_tasks else 0

            print(f"✅ Batch inference complete!")
            print(f"   Total time: {total_inference_time:.2f}s")
            print(f"   Avg per task: {avg_time_per_task:.3f}s")
            print(f"   Throughput: {len(new_tasks)/total_inference_time:.1f} tasks/sec")

        except Exception as e:
            print(f"❌ vLLM inference failed: {e}")
            import traceback
            traceback.print_exc()
            return []

        # Process and save results
        results = []
        for i, (task, output) in enumerate(zip(new_tasks, outputs)):
            # Extract raw response
            raw_response = output.outputs[0].text if output.outputs else ""

            # CRITICAL: Clean response using same cleaner as sequential version
            # This ensures consistent format for evaluation (removes User:/Assistant: prefixes,
            # standardizes to "The answer is X" format, etc.)
            response_text = clean_vlm_response(raw_response)

            result = {
                'result_id': task['result_id'],
                'question_id': task['question_id'],
                'prompt': task['question_text'],
                'model_response': response_text,
                'ground_truth': task['ground_truth'],
                'question_type': task['question_type'],
                'model_id': engine,
                'adversarial': task['adversarial'],
                'task': task['question_type'],
                'attack_type': task['attack_type'],
                'epsilon_target': task['epsilon_target'],
                'epsilon_actual': task['epsilon_actual'],
                'inference_image_path': task['image_path'],
                'clean_image_path': f"data/clean/{task['original_image_path']}",
                'timestamp': datetime.now().isoformat() + 'Z',
                # Performance metrics (vLLM batch mode)
                'inference_time_seconds': avg_time_per_task,
                'gpu_before_mb': None,  # vLLM manages GPU internally
                'gpu_after_mb': None,
                'gpu_peak_mb': None,
                'gpu_reserved_mb': None,
                'total_gpu_mb': None,
                'cpu_before_mb': None,
                'cpu_after_mb': None,
                'was_cached': True,  # Model stays loaded for entire batch
                'loading_time_seconds': 0.0,
                'batch_size': len(new_tasks),
                'position_in_batch': i + 1,
                'total_batch_questions': len(new_tasks)
            }

            self.save_result(result)
            results.append(result)

        print(f"💾 Saved {len(results)} results to database")
        return results

    def generate_result_id(self, engine, task):
        """Generate deterministic result ID (same as sequential version for compatibility)"""
        import hashlib

        clean_or_adversarial = "adversarial" if task['adversarial'] else "clean"
        attack_type = task['attack_type']
        attack_name = task['attack_name']
        epsilon_target = str(task['epsilon_target'])
        task_type = task['task_type']
        image_path = task['image_path']
        question_id = task['question_id']

        content = f"{engine}_{clean_or_adversarial}_{attack_type}_{attack_name}_{epsilon_target}_{task_type}_{image_path}_{question_id}"
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
        """Save inference result to database (same schema as sequential version)"""
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

    def cleanup(self):
        """Cleanup vLLM engine and GPU memory"""
        if self.current_llm is not None:
            print(f"🧹 Cleaning up vLLM engine: {self.current_engine}")
            del self.current_llm
            self.current_llm = None
            self.current_engine = None
            cleanup_gpu_memory()


def main():
    """Main execution function"""
    print("="*80)
    print("🚀 vLLM BATCH INFERENCE ENGINE")
    print("="*80)
    print("Goal: Populate inference_results table with VLM answers")
    print("Source: ground_truth_questions + attack_executions tables")
    print("Mode: vLLM continuous batching (90-95% GPU utilization)")

    # Check vLLM availability
    if not VLLM_AVAILABLE:
        print("\n❌ vLLM is required for this script.")
        print("   Install with: pip install vllm>=0.6.0")
        print("   Or use model_inference_sequential.py for HuggingFace inference")
        return

    # Ensure ground truth data exists
    print("\n🔍 Checking ground truth data...")
    if not populate_ground_truth_questions(overwrite=False):
        print("❌ Failed to ensure ground truth data. Exiting.")
        return

    # User selections
    engines = select_engine()

    print(f"\n📋 Configuration:")
    print(f"   🔧 Engines: {', '.join(engines)}")
    print(f"   🖼️  Processing: Both clean and adversarial images")
    print(f"   ⚡ Mode: vLLM continuous batching")

    # Initialize inference engine
    inference_engine = VLLMInferenceEngine()

    # Load tasks
    tasks = inference_engine.load_inference_tasks()
    if not tasks:
        print("❌ No inference tasks found. Exiting.")
        return

    # Run inference for each engine
    all_results = {}
    for engine in engines:
        print(f"\n{'='*60}")
        try:
            results = inference_engine.run_inference(engine, tasks)
            all_results[engine] = results
        except Exception as e:
            print(f"❌ Error with {engine}: {e}")
            import traceback
            traceback.print_exc()
            all_results[engine] = []

        # Cleanup between engines
        inference_engine.cleanup()

    # Final summary
    print(f"\n{'='*80}")
    print("🎉 vLLM BATCH INFERENCE COMPLETE")
    print("="*80)
    total_results = sum(len(results) for results in all_results.values())
    print(f"📊 Total results saved: {total_results}")
    for engine, results in all_results.items():
        print(f"   {engine}: {len(results)} results")
    print(f"💾 Database: results/centralized_pipeline.db → inference_results table")


if __name__ == "__main__":
    main()
