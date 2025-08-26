import random, json
import os
import base64
from mimetypes import guess_type
import sys
import torch
import gc

# Add proper paths for imports when running from different directories
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)  # scripts directory
sys.path.insert(0, os.path.join(parent_dir, 'local_model'))  # local_model directory
try:
    from scripts.attack_selector import select_attack
    from scripts.batch_processor import create_batch_processor
except ImportError:
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


def select_engine():
    """Interactive function to select the engine to use"""
    # Import the list_available_models function from local_model_utils
    try:
        from vlm_local_client import list_available_models
    except ImportError:
        from scripts.vlm_local_client import list_available_models
    
    # Get all available local models
    local_models = list_available_models()
    
    # Create the menu options
    print("\nSelect the engine to use:")
    # print("  [1] OpenAI GPT-4o")  # COMMENTED OUT - NO PAID API USAGE
    
    # Add all local models to the menu (starting from 1 now)
    for i, model in enumerate(local_models):
        print(f"  [{i+1}] {model}")
    
    # Add ALL option
    all_option = len(local_models) + 1
    print(f"  [{all_option}] ALL LOCAL MODELS")
    
    while True:
        choice = input(f"\nEnter your choice (1-{all_option}): ")
        
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
                    from scripts.vlm_local_client import send_chat_request_azure
                return [(model_name, send_chat_request_azure)]
            elif choice_num == all_option:
                print("Selected: ALL LOCAL MODELS")
                # Import only local client
                try:
                    from vlm_local_client import send_chat_request_azure as local_send_chat
                except ImportError:
                    from scripts.vlm_local_client import send_chat_request_azure as local_send_chat
                
                # Create a list with all local models only (no GPT-4o)
                engines = []
                for model in local_models:
                    engines.append((model, local_send_chat))
                
                return engines
            else:
                print(f"Invalid choice. Please enter a number between 1 and {all_option}.")
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


def get_task_question_count(task):
    """Return the actual question count by reading the ground truth file"""
    ground_truth_file = f'results/ground_truth/eval_{task}.json'
    try:
        with open(ground_truth_file, 'r') as f:
            count = sum(1 for line in f if line.strip())
        return count
    except FileNotFoundError:
        print(f"⚠️  Warning: Ground truth file not found: {ground_truth_file}")
        # Fallback to hardcoded estimates if ground truth file doesn't exist
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
                from scripts.vlm_local_client import get_gpu_memory_info
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
    
    # Define input file path - use centralized ground truth file
    ground_truth_file = f'results/ground_truth/eval_{task}.json'
    
    try:
        with open(ground_truth_file) as f:
            eval_data = []
            for line in f:
                eval_data.append(json.loads(line))

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
                
                # Convert to list for batch processing
                image_data_list = list(image_to_data.items())
                
                # Add metadata to each data item
                for _, data_items in image_data_list:
                    for data in data_items:
                        data['metadata'] = {
                            "adversarial": attack_name != "Original (No Attack)",
                            "task": task
                        }
                
                # Use the VLM batch orchestrator for clean processing
                print(f"🚀 Using VLM batch orchestrator for {engine}")
                res_list = batch_processor.process_vlm_requests_in_batches(
                    messages_data=image_data_list,
                    vlm_client_func=send_chat_request_azure,
                    engine=engine,
                    local_image_to_data_url_func=local_image_to_data_url,
                    img_dir=img_dir
                )
                
                # Save all results
                with open(output_file, 'a') as fout:
                    for result in res_list:
                        fout.write(json.dumps(result) + '\n')
                
                print(f"\n✅ Inference completed! Results saved to {output_file}")
                print(f"📊 Processed {len(res_list)} questions across {len(image_data_list)} images")
                
            except Exception as e:
                print(f"❌ Error during evaluation: {e}")
                # Save partial results
                with open(output_file, 'w') as fout:
                    for res in res_list:
                        fout.write(json.dumps(res) + '\n')
                print(f"💾 Partial results saved to {output_file}")
                
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {ground_truth_file}")
        print(f"Make sure the file exists at {os.path.abspath(ground_truth_file)}")
        print("Directory structure should be:")
        print(f"  results/ground_truth/eval_{task}.json")


def prepare_ground_truth_files():
    """Prepare ground truth files for all tasks from eval_all.json, filtered by processed_images.json"""
    # Ensure ground truth directory exists
    os.makedirs('results/ground_truth', exist_ok=True)
    
    # Check if the main eval_all.json file exists
    all_data_file = 'data/test_extracted/benchmark/eval_all.json'
    if not os.path.exists(all_data_file):
        print(f"Error: Main data file not found at {all_data_file}")
        return False
    
    # Load processed images
    processed_images = load_processed_images()
    if not processed_images:
        print("Error loading processed_images.json")
        return False
    
    # List of all task types
    task_types = [
        'chart', 'table', 'road_map', 'dashboard', 
        'flowchart', 'relation_graph', 'planar_layout', 'visual_puzzle'
    ]
    
    try:
        # Read all data
        with open(all_data_file, 'r') as f:
            all_data = [json.loads(line) for line in f]
        
        # Group data by task type and filter by processed images
        task_data = {task: [] for task in task_types}
        
        for item in all_data:
            task_type = item.get('type')
            if task_type in task_types:
                # Extract the image filename from the path
                image_path = item.get('image', '')
                
                # Check if this image is in our processed_images list
                # We need to check if the image filename (without path) is in the processed_images list
                image_filename = os.path.basename(image_path)
                
                # Check if this image is in our processed images for this task
                for processed_path in processed_images.get(task_type, []):
                    if os.path.basename(processed_path) == image_filename:
                        task_data[task_type].append(item)
                        break
        
        # Write separate ground truth files for each task
        for task, data in task_data.items():
            if not data:
                print(f"Warning: No ground truth data found for task '{task}' after filtering by processed_images.json")
                continue
                
            output_file = f'results/ground_truth/eval_{task}.json'
            with open(output_file, 'w') as f:
                for item in data:
                    f.write(json.dumps(item) + '\n')
            print(f"Created ground truth file for {task} with {len(data)} items: {output_file}")
        
        return True
    
    except Exception as e:
        print(f"Error preparing ground truth files: {e}")
        return False


def run_cross_attack_evaluation(engine, send_chat_request_azure, task, random_count, attack_configs):
    """Run cross-attack evaluation using intelligent batch processing for maximum GPU utilization"""
    print(f"\n🚀 Cross-Attack Batch Processing for task: {task}")
    print(f"Processing {len(attack_configs)} attacks simultaneously")
    
    # Create batch processor with 20% safety margin
    batch_processor = create_batch_processor(safety_margin_percent=20.0)
    
    # Show memory status before evaluation starts
    if engine != 'gpt4o':
        try:
            try:
                from vlm_local_client import get_gpu_memory_info
            except ImportError:
                from scripts.vlm_local_client import get_gpu_memory_info
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
    
    # Define input file path - use centralized ground truth file
    ground_truth_file = f'results/ground_truth/eval_{task}.json'
    
    try:
        with open(ground_truth_file) as f:
            eval_data = []
            for line in f:
                eval_data.append(json.loads(line))

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
                
                # Save results for each attack
                for output_file, img_dir, attack_name in attack_configs:
                    if attack_name in attack_results:
                        results = attack_results[attack_name]
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
                
    except FileNotFoundError:
        print(f"Error: Ground truth file not found at {ground_truth_file}")
        print(f"Make sure the file exists at {os.path.abspath(ground_truth_file)}")
        print("Directory structure should be:")
        print(f"  results/ground_truth/eval_{task}.json")


if __name__ == '__main__':
    # Prepare ground truth files for all tasks
    if not prepare_ground_truth_files():
        print("Failed to prepare ground truth files. Please check the data directory.")
        sys.exit(1)
    
    # Select engine(s) and get the appropriate send_chat_request_azure function(s)
    engine_configs = select_engine()
    
    # Select task(s) to evaluate
    selected_tasks = select_task()
    
    # Check if ALL tasks were selected
    is_all_tasks = len(selected_tasks) > 1 and all(task in ['chart', 'table', 'road_map', 'dashboard', 
                                                           'flowchart', 'relation_graph', 'planar_layout', 
                                                           'visual_puzzle'] for task in selected_tasks)
    
    # Check if ALL engines were selected
    is_all_engines = len(engine_configs) > 1
    
    # If ALL tasks were selected or ALL engines were selected, ask once for attack type
    all_attacks_choice = None
    if is_all_tasks or is_all_engines:
        print("\nYou selected", end=" ")
        if is_all_tasks:
            print("ALL tasks", end="")
        if is_all_tasks and is_all_engines:
            print(" and", end=" ")
        if is_all_engines:
            print("ALL engines", end="")
        print(". Do you want to use ALL ATTACKS for all combinations?")
        print("  [1] Yes, use ALL ATTACKS for all combinations")
        print("  [2] No, ask for each combination separately")
        
        while True:
            choice = input("\nEnter your choice (1-2): ")
            try:
                choice_num = int(choice)
                if choice_num == 1:
                    all_attacks_choice = 19  # This is the option number for ALL ATTACKS
                    print("Using ALL ATTACKS for all combinations")
                    break
                elif choice_num == 2:
                    print("Will ask for attack type for each combination separately")
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except ValueError:
                print("Please enter a valid number.")
    
    
    # Process each engine
    for engine_idx, (engine, send_chat_request_azure) in enumerate(engine_configs):
        print(f"\n{'='*20} Evaluating {engine} (Model {engine_idx+1}/{len(engine_configs)}) {'='*20}")
        
        # If this is a local model (not GPT-4o), unload previous models to free GPU memory
        if engine != 'gpt4o' and engine_idx > 0:
            # Import unload function
            try:
                from vlm_local_client import unload_all_models, get_gpu_memory_info
            except ImportError:
                from scripts.vlm_local_client import unload_all_models, get_gpu_memory_info
            
            print("🔧 Unloading previously loaded models to free GPU memory...")
            unload_all_models()
            print(f"Memory status after cleanup: {get_gpu_memory_info()}")
        
        # Process each selected task
        for task in selected_tasks:
            print(f"\n{'-'*20} Task: {task} {'-'*20}")
            
            # Get appropriate question count for this task
            random_count = get_task_question_count(task)
            
            # Select attack type(s) - if all_attacks_choice is set, use that instead of asking
            if all_attacks_choice is not None:
                # Directly call select_attack with the ALL ATTACKS choice
                from attack_selector import select_attack
                import sys
                
                # Temporarily redirect stdin to provide the automatic choice
                original_stdin = sys.stdin
                sys.stdin = open('/dev/null', 'r')
                
                # Call select_attack with the ALL ATTACKS choice
                attack_configs = select_attack(engine, task, random_count, auto_choice=all_attacks_choice)
                
                # Restore stdin
                sys.stdin = original_stdin
            else:
                # Normal interactive selection
                attack_configs = select_attack(engine, task, random_count)
            
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
            from scripts.vlm_local_client import unload_all_models, get_gpu_memory_info
        print("🧹 Final cleanup: Unloading all models...")
        unload_all_models()
        print(f"Final memory status: {get_gpu_memory_info()}")
    except ImportError:
        print("Note: Local model cleanup not available (cloud-only run)")
