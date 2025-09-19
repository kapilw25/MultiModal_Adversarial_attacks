import random, json
import os
import base64
from mimetypes import guess_type
import sys
import torch
import gc
from datetime import datetime
from pathlib import Path
from utils.centralized_database import CentralizedDB

# Add proper paths for imports when running from different directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)  # scripts directory
sys.path.insert(0, os.path.join(parent_dir, 'local_model'))  # local_model directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from attack_selector import select_attack
from batch_processor import create_batch_processor

# Set memory configuration for PyTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Function to clean up GPU memory
def cleanup_gpu_memory():
    """Clean up GPU memory to prevent out-of-memory errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("GPU memory cleaned up")

# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def load_processed_images():
    """Load the list of processed images from JSON file"""
    try:
        with open('data/processed_images.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: data/processed_images.json not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in data/processed_images.json")
        return None


def load_attack_parameters():
    """Load attack parameters containing SSIM values for adversarial images from database"""
    try:
        db = CentralizedDB()
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Get attack executions directly from database
        query = '''
        SELECT image_path, adversarial_image_path, ssim, ssim_target, attack_name, attack_category
        FROM attack_executions
        WHERE success = 1 AND ssim IS NOT NULL
        '''
        
        results = cursor.execute(query).fetchall()
        conn.close()
        
        if not results:
            print("⚠️ Warning: No successful attack executions found in database. SSIM values will not be available for adversarial images.")
            return {}

        # Create mapping for SSIM lookup
        ssim_mapping = {}
        for row in results:
            orig_path, adv_path, ssim, ssim_target, attack_name, attack_category = row
            
            # Create composite key lookup
            composite_key = (orig_path, adv_path)
            ssim_mapping[composite_key] = {
                'ssim': ssim,
                'ssim_target': ssim_target,
                'adversarial_image_path': adv_path,
                'attack_type': attack_name,
                'attack_category': attack_category
            }

        print(f"📊 Loaded SSIM data for {len(results)} attack executions from database")
        return ssim_mapping

    except Exception as e:
        print(f"❌ Error loading attack parameters from database: {e}")
        return {}


def get_ssim_for_image(image_path, attack_name, ssim_mapping, img_dir):
    """
    Get SSIM value for an image based on attack type using database lookup

    Args:
        image_path: Path to image (original like "chart/file.png" OR adversarial full path)
        attack_name: Name of the attack ("Original (No Attack)" for clean images)
        ssim_mapping: Dictionary mapping composite keys to SSIM data
        img_dir: Image directory path to construct adversarial image path

    Returns:
        float: SSIM value (1.0 for clean images, actual SSIM for adversarial)
    """
    if attack_name == "Original (No Attack)":
        return 1.0  # Clean images have perfect SSIM

    # Handle case where image_path is already the full adversarial path
    if image_path.startswith('data/adversarial/'):
        # Extract original path from adversarial path
        # Example: data/adversarial/whitebox/fgsm/ssim_085/chart/file.png -> chart/file.png
        path_parts = image_path.split('/')
        if len(path_parts) >= 6:
            original_image_path = '/'.join(path_parts[-2:])  # task/filename
            adv_path = image_path
            orig_path = f"data/clean/{original_image_path}"
        else:
            print(f"⚠️ Warning: Invalid adversarial path format: {image_path}")
            return 0.0
    else:
        # Normal case: image_path is original path like "chart/file.png"
        orig_path = f"data/clean/{image_path}"
        adv_path = f"{img_dir.rstrip('/')}/{image_path}"

    # Direct lookup using composite key from attack_executions table
    composite_key = (orig_path, adv_path)
    if composite_key in ssim_mapping:
        return ssim_mapping[composite_key]['ssim']

    # Fallback if exact match not found
    print(f"⚠️ Warning: No SSIM data found for key: {composite_key}")
    return 0.0


def construct_inference_output_path(engine, task, attack_name, img_dir, ssim_value=0.85):
    """
    Construct output file path following the same structure as adversarial images.

    Examples:
    - Clean: results/inference/DeepSeek1_VL_7B/clean/chart/eval_DeepSeek1_VL_7B_chart_98.json
    - Adversarial: results/inference/DeepSeek1_VL_7B/whitebox/fgsm/ssim_085/chart/eval_DeepSeek1_VL_7B_chart_98.json

    Args:
        engine: Model engine name
        task: Task type (chart, table, etc.)
        attack_name: Attack name ("Original (No Attack)" for clean)
        img_dir: Image directory path
        ssim_value: SSIM threshold value for directory naming

    Returns:
        str: Constructed output file path
    """
    # Parse attack information from img_dir or attack_name
    if attack_name == "Original (No Attack)":
        # Clean images: results/inference/{engine}/clean/{task}/
        attack_path = "clean"
    else:
        # Adversarial images: extract attack info from img_dir
        # img_dir example: "data/adversarial/whitebox/fgsm/" or with ssim folder
        if "data/adversarial/" in img_dir:
            # Extract path components after "data/adversarial/"
            remaining_path = img_dir.replace("data/adversarial/", "").strip("/")
            parts = remaining_path.split("/")

            if len(parts) >= 2:
                box_type = parts[0]  # whitebox/blackbox
                attack_type = parts[1]  # fgsm/pgd/square etc.

                # Check if SSIM folder already exists in path
                if len(parts) >= 3 and parts[2].startswith("ssim_"):
                    ssim_dir = parts[2]  # Use existing ssim folder
                else:
                    # Create SSIM directory name from value
                    ssim_dir = f"ssim_{ssim_value:.2f}".replace(".", "")

                attack_path = f"{box_type}/{attack_type}/{ssim_dir}"
            else:
                attack_path = f"unknown_attack/{attack_name}"
        else:
            # Fallback
            attack_path = f"unknown/{attack_name.lower().replace(' ', '_')}"

    # Construct full path: results/inference/{engine}/{attack_path}/{task}/
    base_dir = f"results/inference/{engine}/{attack_path}/{task}"
    os.makedirs(base_dir, exist_ok=True)

    # Generate filename
    filename = f"eval_{engine}_{task}_98.json"
    return os.path.join(base_dir, filename)


def select_engine():
    """Interactive function to select the engine to use"""
    # Import the list_available_models function from local_model_utils
    try:
        from vlm_local_client import list_available_models
    except ImportError as e:
        print(f"Error importing vlm_local_client: {e}")
        return
    
    # Get all available local models
    local_models = list_available_models()
    
    # Create the menu options
    print("\nSelect the engine to use:")
    # print("  [1] OpenAI GPT-4o")  # COMMENTED OUT - NO PAID API USAGE
    
    # Add all local models to the menu (starting from 1 now)
    for i, model in enumerate(local_models):
        print(f"  [{i+1}] {model}")
    
    # Add ALL options
    all_option = len(local_models) + 1
    all_except_gemma_option = len(local_models) + 2
    print(f"  [{all_option}] ALL LOCAL MODELS")
    print(f"  [{all_except_gemma_option}] ALL MODELS EXCEPT Gemma3_VL_4B (24.28s avg - skip slowest model)")

    while True:
        choice = input(f"\nEnter your choice (1-{all_except_gemma_option}): ")
        
        try:
            choice_num = int(choice)
            
            # if choice_num == 1:  # COMMENTED OUT - NO GPT-4o USAGE
            #     print("Selected: OpenAI GPT-4o")
            #     # Import for OpenAI GPT-4o
            #     from vlm_cloud_client import send_chat_request_azure
            #     return [('gpt4o', send_chat_request_azure)]
            if 1 <= choice_num <= len(local_models):
                # Selected a local model (adjusted index since we removed GPT-4o)
                model_name = local_models[choice_num - 1]
                print(f"Selected: {model_name}")
                try:
                    from vlm_local_client import send_chat_request_azure
                except ImportError:
                    from vlm_local_client import send_chat_request_azure
                return [(model_name, send_chat_request_azure)]
            elif choice_num == all_option:
                print("Selected: ALL LOCAL MODELS")
                # Import only local client
                try:
                    from vlm_local_client import send_chat_request_azure as local_send_chat
                except ImportError:
                    from vlm_local_client import send_chat_request_azure as local_send_chat
                
                # Create a list with all local models only (no GPT-4o)
                engines = []
                for model in local_models:
                    engines.append((model, local_send_chat))
                
                return engines
            elif choice_num == all_except_gemma_option:
                print("Selected: ALL MODELS EXCEPT Gemma3_VL_4B (skipping slowest model)")
                # Import only local client
                try:
                    from vlm_local_client import send_chat_request_azure as local_send_chat
                except ImportError:
                    from vlm_local_client import send_chat_request_azure as local_send_chat

                # Create a list with all local models except Gemma3_VL_4B
                engines = []
                for model in local_models:
                    if model != "Gemma3_VL_4B":
                        engines.append((model, local_send_chat))

                print(f"Will process {len(engines)} models (excluding Gemma3_VL_4B)")
                return engines
            else:
                print(f"Invalid choice. Please enter a number between 1 and {all_except_gemma_option}.")
        except ValueError:
            print("Please enter a valid number.")


def select_task():
    """Interactive function to select the task to evaluate"""
    tasks = [
        ('chart', 'Chart interpretation (27 questions per image)'),
        ('table', 'Table data extraction (21-22 questions per image)'),
        ('road_map', 'Road map navigation (1 question per image)'),
        ('dashboard', 'Dashboard analysis (20 questions per image)'),
        ('flowchart', 'Flowchart understanding (20 questions per image)'),
        ('relation_graph', 'Relation graph analysis (18-19 questions per image)'),
        ('planar_layout', 'Planar layout interpretation (12-24 questions per image)'),
        ('visual_puzzle', 'Visual puzzle solving (6 questions per image)'),
        ('all', 'ALL tasks')
    ]
    
    print("\nSelect the task to evaluate:")
    for i, (task_id, task_desc) in enumerate(tasks):
        print(f"  [{i+1}] {task_desc}")
    
    while True:
        choice = input(f"\nEnter your choice (1-{len(tasks)}): ")
        
        try:
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(tasks):
                selected_task = tasks[choice_num-1][0]
                print(f"Selected task: {tasks[choice_num-1][1]}")
                
                if selected_task == 'all':
                    return [task[0] for task in tasks if task[0] != 'all']
                else:
                    return [selected_task]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(tasks)}.")
        except ValueError:
            print("Please enter a valid number.")


def select_attack_type():
    """Interactive function to select attack type for all evaluations"""
    attack_options = [
        ('clean', 'Original (No Attack)'),
        ('pgd', 'PGD'),
        ('fgsm', 'FGSM'),
        ('cw_linf', 'CW-Linf'),
        ('deepfool', 'DeepFool'),
        ('square', 'Square'),
        ('simba', 'SimBA'),
        ('boundary', 'Boundary'),
        ('pixel', 'Pixel'),
        ('spatial', 'Spatial'),
        ('whitebox', 'White-Box Attacks Only (PGD, FGSM, CW-Linf, DeepFool)'),
        ('blackbox', 'Black-Box Attacks Only (Square, SimBA, Boundary, Pixel, Spatial)'),
        ('all', 'ALL Attacks (Clean + White-Box + Black-Box)')
    ]

    print("\nSelect attack type for all evaluations:")
    for i, (attack_type, desc) in enumerate(attack_options):
        print(f"  [{i+1}] {desc}")

    while True:
        choice = input(f"\nEnter your choice (1-{len(attack_options)}): ")

        try:
            choice_num = int(choice)

            if 1 <= choice_num <= len(attack_options):
                selected_type = attack_options[choice_num-1][0]
                print(f"Selected: {attack_options[choice_num-1][1]}")
                return selected_type
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(attack_options)}.")
        except ValueError:
            print("Please enter a valid number.")


def select_ssim_threshold():
    """Interactive SSIM threshold selection"""
    print("\nSelect SSIM threshold(s) for adversarial attacks:")
    print("  [1] SSIM = 0.85 (Standard threshold)")
    print("  [2] SSIM = 0.90 (High similarity)")
    print("  [3] SSIM = 0.95 (Very high similarity)")
    print("  [4] ALL thresholds (0.85, 0.90, 0.95)")
    print("  [5] Custom threshold (enter manually)")
    
    while True:
        try:
            choice = int(input("Enter your choice (1-5): "))
            if choice == 1:
                print("✅ Selected SSIM threshold: 0.85")
                return [0.85]
            elif choice == 2:
                print("✅ Selected SSIM threshold: 0.90")
                return [0.90]
            elif choice == 3:
                print("✅ Selected SSIM threshold: 0.95")
                return [0.95]
            elif choice == 4:
                print("✅ Selected ALL SSIM thresholds: [0.85, 0.90, 0.95]")
                return [0.85, 0.90, 0.95]
            elif choice == 5:
                while True:
                    try:
                        custom_ssim = float(input("Enter custom SSIM threshold (0.0-1.0): "))
                        if 0.0 <= custom_ssim <= 1.0:
                            print(f"✅ Selected custom SSIM threshold: {custom_ssim}")
                            return [custom_ssim]
                        else:
                            print("❌ SSIM must be between 0.0 and 1.0")
                    except ValueError:
                        print("❌ Please enter a valid number")
            else:
                print("❌ Please enter a valid number (1-5).")
        except ValueError:
            print("❌ Please enter a valid number.")


def select_overwrite_policy():
    """Interactive function to select overwrite policy for existing files"""
    overwrite_options = [
        ('skip', 'Skip existing files (faster, resume previous runs)'),
        ('overwrite', 'Overwrite all existing files (complete fresh run)')
    ]

    print("\nSelect policy for existing result files:")
    for i, (policy, desc) in enumerate(overwrite_options):
        print(f"  [{i+1}] {desc}")

    while True:
        choice = input(f"\nEnter your choice (1-{len(overwrite_options)}): ")

        try:
            choice_num = int(choice)

            if 1 <= choice_num <= len(overwrite_options):
                selected_policy = overwrite_options[choice_num-1][0]
                print(f"Selected: {overwrite_options[choice_num-1][1]}")
                return selected_policy
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(overwrite_options)}.")
        except ValueError:
            print("Please enter a valid number.")


def generate_attack_configs_from_selection(engine, task, num_samples, attack_type_selection, overwrite_policy):
    """
    Generate attack configurations based on user's attack type selection.

    Args:
        engine (str): Model engine name
        task (str): Task type
        num_samples (int): Number of evaluation samples
        attack_type_selection (str): Selected attack type
        overwrite_policy (str): 'skip' or 'overwrite'

    Returns:
        list: List of tuples (output_file, img_dir, attack_name)
    """
    
    # For clean images, create config directly
    if attack_type_selection == 'clean':
        output_file = construct_inference_output_path(engine, task, "Original (No Attack)", "data/clean/")
        
        # Apply overwrite policy
        if overwrite_policy == 'skip' and os.path.exists(output_file):
            print(f"Skipping existing file: {output_file}")
            return []
        
        return [(output_file, "data/clean/", "Original (No Attack)")]
    
    # For adversarial attacks, use the attack selector but handle overwrite policy
    from attack_selector import select_attack

    # Map attack type selection to auto_choice values for select_attack function
    attack_mapping = {
        'pgd': 2,            # PGD
        'fgsm': 3,           # FGSM
        'cw_linf': 4,        # CW-Linf
        'deepfool': 5,       # DeepFool
        'square': 6,         # Square
        'simba': 7,          # SimBA
        'boundary': 8,       # Boundary
        'pixel': 9,          # Pixel
        'spatial': 10,       # Spatial
        'all': 11           # ALL ATTACKS (last option in attack_selector)
    }

    if attack_type_selection in ['whitebox', 'blackbox']:
        # For grouped selections, manually call select_attack for each attack in the group
        if attack_type_selection == 'whitebox':
            attack_choices = [1, 2, 3, 4, 5]  # clean, pgd, fgsm, cw_linf, deepfool
        else:  # blackbox
            attack_choices = [1, 6, 7, 8, 9, 10]  # clean, square, simba, boundary, pixel, spatial

        all_configs = []
        for choice in attack_choices:
            configs = select_attack(engine, task, num_samples, auto_choice=choice)
            if configs:
                all_configs.extend(configs)

        # Apply overwrite policy
        if overwrite_policy == 'overwrite':
            return all_configs
        else:
            # Filter out existing files for skip policy
            filtered_configs = []
            for output_file, img_dir, attack_name in all_configs:
                if not os.path.exists(output_file):
                    filtered_configs.append((output_file, img_dir, attack_name))
            return filtered_configs
    else:
        # Single attack or ALL attacks
        auto_choice = attack_mapping.get(attack_type_selection)
        if auto_choice:
            configs = select_attack(engine, task, num_samples, auto_choice=auto_choice)
            if configs:
                # Apply overwrite policy
                if overwrite_policy == 'overwrite':
                    return configs
                else:
                    # Filter out existing files for skip policy
                    filtered_configs = []
                    for output_file, img_dir, attack_name in configs:
                        if not os.path.exists(output_file):
                            filtered_configs.append((output_file, img_dir, attack_name))
                    return filtered_configs
            return []
        else:
            print(f"Unknown attack type selection: {attack_type_selection}")
            return []


def get_task_question_count(task):
    """Return the actual question count by reading from the centralized database"""
    try:
        db = CentralizedDB()
        questions = db.get_ground_truth_questions(task_type=task)
        return len(questions)
    except Exception as e:
        print(f"⚠️  Warning: Could not load ground truth from database: {e}")
        # Fallback to hardcoded estimates if database fails
        task_counts_per_image = {
            'chart': 27,
            'table': 22,
            'road_map': 1,
            'dashboard': 20,
            'flowchart': 20,
            'relation_graph': 19,
            'planar_layout': 24,
            'visual_puzzle': 6
        }
        questions_per_image = task_counts_per_image.get(task, 10)
        return questions_per_image * 3


def ensure_model_directories(engine):
    """Ensure that the model's results directory exists"""
    model_dir = f'results/models/{engine}'
    os.makedirs(model_dir, exist_ok=True)
    print(f"Ensured directory exists: {model_dir}")


def run_evaluation(engine, send_chat_request_azure, task, random_count, output_file, img_dir, attack_name):
    """Run evaluation for a specific attack type using intelligent batch processing"""
    print(f"\nRunning evaluation for {attack_name} on task: {task}")

    # Load SSIM mapping from attack_parameters.json
    ssim_mapping = load_attack_parameters()

    # Construct proper output file path following adversarial directory structure
    output_file = construct_inference_output_path(engine, task, attack_name, img_dir)
    print(f"Output file: {output_file}")
    print(f"Image directory: {img_dir}")

    # Create batch processor with 20% safety margin
    batch_processor = create_batch_processor(safety_margin_percent=20.0)
    
    # Show memory status before evaluation starts
    if engine != 'gpt4o':
        try:
            try:
                from vlm_local_client import get_gpu_memory_info
            except ImportError:
                from vlm_local_client import get_gpu_memory_info
            print(f"🔍 Memory status before evaluation: {get_gpu_memory_info()}")
        except ImportError:
            memory_info = batch_processor.get_memory_summary()
            print(f"🔍 Memory status: {memory_info['allocated']:.1f}GB allocated, {memory_info['available']:.1f}GB available")
    
    # Ensure model directory exists
    ensure_model_directories(engine)
    
    # Load processed images
    processed_images = load_processed_images()
    if not processed_images or task not in processed_images:
        print(f"Error: No processed images found for task '{task}'")
        return
    
    # Load ground truth data from centralized database
    try:
        db = CentralizedDB()
        eval_data = db.get_ground_truth_questions(task_type=task)

        if not eval_data:
            print(f"Error: No ground truth data found in database for task '{task}'")
            return

        # Filter eval_data to only include questions for images in processed_images.json
        filtered_eval_data = []
        processed_images_for_task = set(processed_images.get(task, []))

        print(f"🔍 Filtering evaluation data for task '{task}':")
        print(f"   Available processed images: {list(processed_images_for_task)}")

        unique_images_found = set()
        for data in eval_data:
            image_path = data.get('image', '')

            # Check if this exact image path is in our processed images for this task
            if image_path in processed_images_for_task:
                filtered_eval_data.append(data)
                unique_images_found.add(image_path)
            else:
                # Fallback: check if just the filename matches (for backward compatibility)
                image_filename = os.path.basename(image_path)
                for processed_path in processed_images_for_task:
                    if os.path.basename(processed_path) == image_filename:
                        filtered_eval_data.append(data)
                        unique_images_found.add(image_path)
                        break
            
            print(f"   ✅ Found questions for {len(unique_images_found)} images: {list(unique_images_found)}")
            
            if not filtered_eval_data:
                print(f"Error: No evaluation data found for task '{task}' after filtering by processed_images.json")
                return
                
            print(f"Found {len(filtered_eval_data)} evaluation items for task '{task}' after filtering")
            
            # Use up to random_count items
            human_select = filtered_eval_data[:random_count]
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            res_list = []
            try:
                # Open file in write mode initially to clear any existing content
                with open(output_file, 'w') as fout:
                    pass  # Just create/clear the file
                
                # Group data by image for batch processing
                image_to_data = {}
                for data in human_select:
                    img_path = data.get('image', '')
                    if img_path not in image_to_data:
                        image_to_data[img_path] = []
                    image_to_data[img_path].append(data)
                
                # Add metadata to each data item including SSIM BEFORE batch processing
                for img_path, data_items in image_to_data.items():
                    # Get SSIM value for this image
                    ssim_value = get_ssim_for_image(img_path, attack_name, ssim_mapping, img_dir)

                    # Determine the correct image path for metadata
                    if attack_name == "Original (No Attack)":
                        # Clean images: use original path
                        metadata_image_path = img_path
                    else:
                        # Adversarial images: find adversarial image path using composite keys
                        metadata_image_path = img_path  # Default fallback

                        # Look for matching adversarial image path in the ssim_mapping
                        for (orig_path, adv_path), ssim_data in ssim_mapping.items():
                            if orig_path == img_path:
                                # Check if the adversarial path matches the expected pattern for this img_dir
                                if img_dir in adv_path:
                                    metadata_image_path = adv_path
                                    break

                        if metadata_image_path == img_path:
                            print(f"⚠️ Warning: No adversarial image path found for {img_path}, using original path")

                    for data in data_items:
                        data['metadata'] = {
                            "adversarial": attack_name != "Original (No Attack)",
                            "task": task,
                            "attack_type": attack_name.lower().replace(" ", "_").replace("(", "").replace(")", ""),
                            "ssim": ssim_value,
                            "image_path": metadata_image_path,
                            "original_image_path": img_path,  # Always keep reference to original
                            "timestamp": datetime.now().isoformat() + "Z"
                        }

                # Convert to list for batch processing
                image_data_list = list(image_to_data.items())
                
                # Use the VLM batch orchestrator for clean processing
                print(f"🚀 Using VLM batch orchestrator for {engine}")
                res_list = batch_processor.process_vlm_requests_in_batches(
                    messages_data=image_data_list,
                    vlm_client_func=send_chat_request_azure,
                    engine=engine,
                    local_image_to_data_url_func=local_image_to_data_url,
                    img_dir=img_dir
                )
                
                # Save all results to database
                try:
                    db = CentralizedDB()
                    for result in res_list:
                        # Save to database
                        inference_data = {
                            'evaluation_id': f"{engine}_{task}_{attack_name}_{result.get('id', '')}",
                            'model_name': engine,
                            'task_type': task,
                            'image_path': result.get('metadata', {}).get('image_path', ''),
                            'question': result.get('user_query', ''),
                            'model_response': result.get('response', ''),
                            'ground_truth': result.get('correct_answer', ''),
                            'attack_type': attack_name,
                            'adversarial': result.get('metadata', {}).get('adversarial', False),
                            'ssim': result.get('metadata', {}).get('ssim', 0.0),
                            'timestamp': result.get('metadata', {}).get('timestamp', datetime.now().isoformat() + "Z")
                        }
                        db.save_simple_inference_result(inference_data)

                    print(f"\n✅ Inference completed! Results saved to database")
                    print(f"📊 Processed {len(res_list)} questions across {len(image_data_list)} images")

                    # Also save to file for backward compatibility (optional)
                    with open(output_file, 'a') as fout:
                        for result in res_list:
                            fout.write(json.dumps(result) + '\n')
                    print(f"📁 Results also saved to file: {output_file}")

                except Exception as db_error:
                    print(f"⚠️ Warning: Failed to save to database: {db_error}")
                    print("📁 Falling back to file-only save...")
                    # Fallback to file save
                    with open(output_file, 'a') as fout:
                        for result in res_list:
                            fout.write(json.dumps(result) + '\n')
                    print(f"✅ Results saved to file: {output_file}")
                
            except Exception as e:
                print(f"❌ Error during evaluation: {e}")
                # Save partial results to database and file
                try:
                    db = CentralizedDB()
                    for res in res_list:
                        # Save to database
                        inference_data = {
                            'evaluation_id': f"{engine}_{task}_{attack_name}_{res.get('id', '')}",
                            'model_name': engine,
                            'task_type': task,
                            'image_path': res.get('metadata', {}).get('image_path', ''),
                            'question': res.get('user_query', ''),
                            'model_response': res.get('response', ''),
                            'ground_truth': res.get('correct_answer', ''),
                            'attack_type': attack_name,
                            'adversarial': res.get('metadata', {}).get('adversarial', False),
                            'ssim': res.get('metadata', {}).get('ssim', 0.0),
                            'timestamp': res.get('metadata', {}).get('timestamp', datetime.now().isoformat() + "Z")
                        }
                        db.save_simple_inference_result(inference_data)
                    print(f"💾 Partial results saved to database")
                except Exception as db_error:
                    print(f"⚠️ Warning: Failed to save partial results to database: {db_error}")

                # Also save to file
                with open(output_file, 'w') as fout:
                    for res in res_list:
                        fout.write(json.dumps(res) + '\n')
                print(f"💾 Partial results saved to {output_file}")

    except Exception as e:
        print(f"Error: Failed to load ground truth data from database: {e}")
        print("Make sure the centralized database is properly initialized and contains ground truth data")


def prepare_ground_truth_files():
    """Prepare ground truth data in centralized database from eval_all.json, filtered by processed_images.json"""
    # Check if the main eval_all.json file exists
    all_data_file = 'data/clean/benchmark/eval_all.json'
    if not os.path.exists(all_data_file):
        print(f"Error: Main data file not found at {all_data_file}")
        return False

    # Load processed images
    processed_images = load_processed_images()
    if not processed_images:
        print("Error loading processed_images.json")
        return False

    # Initialize database
    db = CentralizedDB()

    # List of all task types
    task_types = [
        'chart', 'table', 'road_map', 'dashboard',
        'flowchart', 'relation_graph', 'planar_layout', 'visual_puzzle'
    ]

    try:
        # Read all data
        with open(all_data_file, 'r') as f:
            all_data = [json.loads(line) for line in f]

        # Group data by task type - FILTER BY PROCESSED IMAGES ONLY
        task_data = {task: [] for task in task_types}

        # Create set of processed image paths for efficient lookup
        processed_images_set = set()
        for task_images in processed_images.values():
            processed_images_set.update(task_images)

        print(f"🔍 Filtering {len(all_data)} questions to match {len(processed_images_set)} processed images")

        for item in all_data:
            task_type = item.get('type')
            image_path = item.get('image', '')

            # Only include questions for images that are in processed_images.json
            if task_type in task_types and image_path in processed_images_set:
                task_data[task_type].append(item)

        # Save ground truth data to database instead of JSON files (FILTERED)
        total_saved = 0
        filtered_count = 0
        for task, data in task_data.items():
            if not data:
                print(f"⚠️ Warning: No ground truth data found for task '{task}' after filtering by processed_images.json")
                continue

            # Save each question to database
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
                db.save_ground_truth_question(question_data)
                total_saved += 1

            filtered_count += len(data)
            print(f"✅ Saved ground truth data for {task} with {len(data)} items to database (filtered)")

        print(f"\n📊 FILTERING SUMMARY:")
        print(f"   Original questions in eval_all.json: {len(all_data)}")
        print(f"   Target images in processed_images.json: {len(processed_images_set)}")
        print(f"   Questions matched to processed images: {total_saved}")
        print(f"   Filtering ratio: {total_saved}/{len(all_data)} = {(total_saved/len(all_data)*100):.1f}%")
        print(f"✅ Total {total_saved} FILTERED ground truth questions saved to centralized database")
        return True

    except Exception as e:
        print(f"Error preparing ground truth data: {e}")
        return False


def run_cross_attack_evaluation(engine, send_chat_request_azure, task, random_count, attack_configs):
    """Run cross-attack evaluation using intelligent batch processing for maximum GPU utilization"""
    print(f"\n🚀 Cross-Attack Batch Processing for task: {task}")
    print(f"Processing {len(attack_configs)} attacks simultaneously")

    # Load SSIM mapping from attack_parameters.json
    ssim_mapping = load_attack_parameters()

    # Update attack_configs with proper output paths
    updated_attack_configs = []
    for output_file, img_dir, attack_name in attack_configs:
        new_output_file = construct_inference_output_path(engine, task, attack_name, img_dir)
        updated_attack_configs.append((new_output_file, img_dir, attack_name))
        print(f"📁 Updated output path for {attack_name}: {new_output_file}")

    attack_configs = updated_attack_configs

    # Create batch processor with 20% safety margin
    batch_processor = create_batch_processor(safety_margin_percent=20.0)
    
    # Show memory status before evaluation starts
    if engine != 'gpt4o':
        try:
            try:
                from vlm_local_client import get_gpu_memory_info
            except ImportError:
                from vlm_local_client import get_gpu_memory_info
            print(f"🔍 Memory status before evaluation: {get_gpu_memory_info()}")
        except ImportError:
            memory_info = batch_processor.get_memory_summary()
            print(f"🔍 Memory status: {memory_info['allocated']:.1f}GB allocated, {memory_info['available']:.1f}GB available")
    
    # Ensure model directories exist for all attacks
    for output_file, _, _ in attack_configs:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load processed images
    processed_images = load_processed_images()
    if not processed_images or task not in processed_images:
        print(f"Error: No processed images found for task '{task}'")
        return
    
    processed_images_for_task = set(processed_images.get(task, []))
    
    # Load ground truth data from centralized database
    try:
        db = CentralizedDB()
        eval_data = db.get_ground_truth_questions(task_type=task)

        if not eval_data:
            print(f"Error: No ground truth data found in database for task '{task}'")
            return

        # Filter eval_data to only include questions for images in processed_images.json
        filtered_eval_data = []

        print(f"🔍 Filtering evaluation data for task '{task}':")
        print(f"   Available processed images: {list(processed_images_for_task)}")

        unique_images_found = set()
        for data in eval_data:
            image_path = data.get('image', '')

            # Check if this exact image path is in our processed images for this task
            if image_path in processed_images_for_task:
                filtered_eval_data.append(data)
                unique_images_found.add(image_path)
            else:
                # Fallback: check if just the filename matches (for backward compatibility)
                image_filename = os.path.basename(image_path)
                for processed_path in processed_images_for_task:
                    if os.path.basename(processed_path) == image_filename:
                        filtered_eval_data.append(data)
                        unique_images_found.add(image_path)
                        break

        print(f"   ✅ Found questions for {len(unique_images_found)} images: {list(unique_images_found)}")

        if not filtered_eval_data:
            print(f"Error: No evaluation data found for task '{task}' after filtering by processed_images.json")
            return

        print(f"Found {len(filtered_eval_data)} evaluation items for task '{task}' after filtering")

        # Use up to random_count items
        human_select = filtered_eval_data[:random_count]

        try:
            # Use cross-attack batch processing
            print(f"🚀 Using cross-attack batch orchestrator for {engine}")
            attack_results = batch_processor.process_cross_attack_batches(
                    attack_configs=attack_configs,
                    task=task,
                    engine=engine,
                    vlm_client_func=send_chat_request_azure,
                    local_image_to_data_url_func=local_image_to_data_url,
                    eval_data=human_select,
                    processed_images=list(processed_images_for_task)
                )

            # Save results for each attack to database and files
            for output_file, img_dir, attack_name in attack_configs:
                if attack_name in attack_results:
                        results = attack_results[attack_name]

                        # Save to database
                        try:
                            db = CentralizedDB()
                            for result in results:
                                inference_data = {
                                    'evaluation_id': f"{engine}_{task}_{attack_name}_{result.get('id', '')}",
                                    'model_name': engine,
                                    'task_type': task,
                                    'image_path': result.get('metadata', {}).get('image_path', ''),
                                    'question': result.get('user_query', ''),
                                    'model_response': result.get('response', ''),
                                    'ground_truth': result.get('correct_answer', ''),
                                    'attack_type': attack_name,
                                    'adversarial': result.get('metadata', {}).get('adversarial', False),
                                    'ssim': result.get('metadata', {}).get('ssim', 0.0),
                                    'timestamp': result.get('metadata', {}).get('timestamp', datetime.now().isoformat() + "Z")
                                }
                                db.save_simple_inference_result(inference_data)
                            print(f"✅ {attack_name} results saved to database")
                        except Exception as db_error:
                            print(f"⚠️ Warning: Failed to save {attack_name} results to database: {db_error}")

                        # Also save to file for backward compatibility
                        with open(output_file, 'w') as fout:
                            for result in results:
                                fout.write(json.dumps(result) + '\n')

                        print(f"✅ {attack_name} results saved to {output_file}")
                        print(f"📊 Processed {len(results)} questions for {attack_name}")
                
            print(f"\n🎉 Cross-Attack Processing completed for task '{task}'!")
            print(f"📈 Performance gain: {len(attack_configs)}x GPU utilization vs sequential processing")
                
        except Exception as e:
            print(f"❌ Error during cross-attack evaluation: {e}")
            # Fallback to sequential processing
            print("🔄 Falling back to sequential processing...")
            for output_file, img_dir, attack_name in attack_configs:
                run_evaluation(engine, send_chat_request_azure, task, random_count, output_file, img_dir, attack_name)

    except Exception as e:
        print(f"Error: Failed to load ground truth data from database: {e}")
        print("Make sure the centralized database is properly initialized and contains ground truth data")


if __name__ == '__main__':
    # Prepare ground truth data in centralized database for all tasks
    if not prepare_ground_truth_files():
        print("Failed to prepare ground truth data. Please check the data directory.")
        sys.exit(1)

    # STREAMLINED INPUT COLLECTION - 5 inputs only
    print("\n" + "="*80)
    print("🚀 MULTIMODAL ADVERSARIAL ATTACKS - BATCH INFERENCE")
    print("="*80)
    print("📝 This script will collect 5 inputs upfront and process all combinations:")
    print("   1️⃣  Engine Selection (Model to evaluate)")
    print("   2️⃣  Task Selection (Image types to evaluate)")
    print("   3️⃣  Attack Type Selection (Individual/Grouped attacks)")
    print("   4️⃣  SSIM Threshold Selection (For adversarial attacks)")
    print("   5️⃣  Overwrite Policy (Skip existing vs Fresh run)")
    print("="*80)

    # Input 1: Select engine(s)
    print("\n1️⃣  ENGINE SELECTION:")
    engine_configs = select_engine()

    # Input 2: Select task(s)
    print("\n2️⃣  TASK SELECTION:")
    selected_tasks = select_task()

    # Input 3: Select attack type
    print("\n3️⃣  ATTACK TYPE SELECTION:")
    attack_type_selection = select_attack_type()

    # Input 4: Select SSIM threshold (only for adversarial attacks)
    selected_ssim_thresholds = [1.0]  # Default for clean images (perfect similarity)
    if attack_type_selection != 'clean':
        print("\n4️⃣  SSIM THRESHOLD SELECTION:")
        selected_ssim_thresholds = select_ssim_threshold()
    else:
        print("\n4️⃣  SSIM THRESHOLD SELECTION:")
        print("✅ Skipped (clean images have SSIM = 1.0)")

    # Input 5: Select overwrite policy
    print("\n5️⃣  OVERWRITE POLICY:")
    overwrite_policy = select_overwrite_policy()

    # Display final configuration
    print("\n" + "="*80)
    print("🎯 FINAL CONFIGURATION SUMMARY:")
    print("="*80)
    if engine_configs:
        print(f"📱 Engines: {len(engine_configs)} selected ({', '.join([e[0] for e in engine_configs])})")
    else:
        print("❌ No engines available - check import errors above")
        sys.exit(1)
    print(f"📋 Tasks: {len(selected_tasks)} selected ({', '.join(selected_tasks)})")
    print(f"⚔️  Attack Type: {attack_type_selection.upper()}")
    if attack_type_selection == 'clean':
        print(f"🎯 SSIM: 1.0 (clean images)")
    else:
        print(f"🎯 SSIM Thresholds: {selected_ssim_thresholds}")
    print(f"🔄 Overwrite Policy: {overwrite_policy.upper()}")
    print("="*80)

    # Confirm before starting
    proceed = input("\n🚀 Ready to start processing? (y/n): ").lower()
    if proceed != 'y':
        print("❌ Processing cancelled by user.")
        sys.exit(0)

    print("\n✅ Starting batch processing with collected configuration...")
    
    
    # Process each engine
    for engine_idx, (engine, send_chat_request_azure) in enumerate(engine_configs):
        print(f"\n{'='*20} Evaluating {engine} (Model {engine_idx+1}/{len(engine_configs)}) {'='*20}")
        
        # If this is a local model (not GPT-4o), unload previous models to free GPU memory
        if engine != 'gpt4o' and engine_idx > 0:
            # Import unload function
            try:
                from vlm_local_client import unload_all_models, get_gpu_memory_info
            except ImportError:
                from vlm_local_client import unload_all_models, get_gpu_memory_info
            
            print("🔧 Unloading previously loaded models to free GPU memory...")
            unload_all_models()
            print(f"Memory status after cleanup: {get_gpu_memory_info()}")
        
        # Process each selected task
        for task in selected_tasks:
            print(f"\n{'-'*20} Task: {task} {'-'*20}")
            
            # Get appropriate question count for this task
            random_count = get_task_question_count(task)

            # Generate attack configurations based on user's selection
            attack_configs = generate_attack_configs_from_selection(engine, task, random_count, attack_type_selection, overwrite_policy)
            
            if not attack_configs:
                print(f"No attacks selected or all selected attacks already have output files for {engine} on task {task}. Skipping.")
                continue
            
            # Check if we should use cross-attack batching optimization
            if len(attack_configs) > 1:
                print(f"\n🚀 Cross-Attack Batching: Processing {len(attack_configs)} attacks simultaneously for maximum GPU utilization!")
                run_cross_attack_evaluation(engine, send_chat_request_azure, task, random_count, attack_configs)
            else:
                # Single attack - use normal processing
                for output_file, img_dir, attack_name in attack_configs:
                    run_evaluation(engine, send_chat_request_azure, task, random_count, output_file, img_dir, attack_name)
    
    print("\n✅ All evaluations completed!")
    
    # Final cleanup after all evaluations
    try:
        try:
            from vlm_local_client import unload_all_models, get_gpu_memory_info
        except ImportError:
            from vlm_local_client import unload_all_models, get_gpu_memory_info
        print("🧹 Final cleanup: Unloading all models...")
        unload_all_models()
        print(f"Final memory status: {get_gpu_memory_info()}")
    except ImportError:
        print("Note: Local model cleanup not available (cloud-only run)")
