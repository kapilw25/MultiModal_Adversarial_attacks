import time
import random, json
from tqdm import tqdm
import os
import base64
from mimetypes import guess_type
import sys
import torch
import gc
from adversarial_attack_config import select_attack

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
    from local_model_utils import list_available_models
    
    # Get all available local models
    local_models = list_available_models()
    
    # Create the menu options
    print("\nSelect the engine to use:")
    print("  [1] OpenAI GPT-4o")
    
    # Add all local models to the menu
    for i, model in enumerate(local_models):
        print(f"  [{i+2}] {model}")
    
    # Add ALL option
    all_option = len(local_models) + 2
    print(f"  [{all_option}] ALL")
    
    while True:
        choice = input(f"\nEnter your choice (1-{all_option}): ")
        
        try:
            choice_num = int(choice)
            
            if choice_num == 1:
                print("Selected: OpenAI GPT-4o")
                # Import for OpenAI GPT-4o
                from cloud_model_utils import send_chat_request_azure
                return [('gpt4o', send_chat_request_azure)]
            elif 2 <= choice_num <= len(local_models) + 1:
                # Selected a local model
                model_name = local_models[choice_num - 2]
                print(f"Selected: {model_name}")
                from local_model_utils import send_chat_request_azure
                return [(model_name, send_chat_request_azure)]
            elif choice_num == all_option:
                print("Selected: ALL engines")
                # Import both modules
                from cloud_model_utils import send_chat_request_azure as gpt4o_send_chat
                from local_model_utils import send_chat_request_azure as local_send_chat
                
                # Create a list with GPT-4o and all local models
                engines = [('gpt4o', gpt4o_send_chat)]
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
    """Return the appropriate question count for each task type"""
    task_counts = {
        'chart': 27,
        'table': 22,
        'road_map': 1,
        'dashboard': 20,
        'flowchart': 20,
        'relation_graph': 19,
        'planar_layout': 24,
        'visual_puzzle': 6
    }
    return task_counts.get(task, 10)  # Default to 10 if task not found


def ensure_model_directories(engine):
    """Ensure that the model's results directory exists"""
    model_dir = f'results/models/{engine}'
    os.makedirs(model_dir, exist_ok=True)
    print(f"Ensured directory exists: {model_dir}")


def run_evaluation(engine, send_chat_request_azure, task, random_count, output_file, img_dir, attack_name):
    """Run evaluation for a specific attack type using only images from processed_images.json"""
    print(f"\nRunning evaluation for {attack_name} on task: {task}")
    print(f"Output file: {output_file}")
    print(f"Image directory: {img_dir}")
    
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
            for data in eval_data:
                image_path = data.get('image', '')
                image_filename = os.path.basename(image_path)
                
                # Check if this image is in our processed images for this task
                for processed_path in processed_images.get(task, []):
                    if image_filename in processed_path:
                        filtered_eval_data.append(data)
                        break
            
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
                
                # Group data by image to process one image at a time
                image_to_data = {}
                for data in human_select:
                    img_path = data.get('image', '')
                    if img_path not in image_to_data:
                        image_to_data[img_path] = []
                    image_to_data[img_path].append(data)
                
                # Process one image at a time
                for img_path, data_items in image_to_data.items():
                    print(f"Processing image: {img_path}")
                    full_img_path = img_dir + img_path
                    
                    # Check if image exists
                    if not os.path.exists(full_img_path):
                        print(f"Warning: Image not found: {full_img_path}")
                        continue
                    
                    # Convert image to data URL once per image
                    url = local_image_to_data_url(full_img_path)
                    
                    # Process all questions for this image
                    for data in tqdm(data_items):
                        msgs = [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": data['text'] + " Answer format (do not generate any other content): The answer is <answer>."
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": url
                                        }
                                    }
                                ]
                            }
                        ]

                        res, _ = send_chat_request_azure(message_text=msgs, engine=engine, sample_n=1)

                        if 'markers' in data.keys():
                            markers = data['markers']
                        else:
                            # Initialize markers to avoid NoneType error
                            markers = []

                        res = {
                            "question_id": data['question_id'],
                            "prompt": data['text'],
                            "text": res,
                            "truth": data['answer'],
                            "type": data['type'],
                            "answer_id": "",
                            "markers": markers,
                            "model_id": engine,
                            "metadata": {
                                "adversarial": attack_name != "Original (No Attack)",
                                "task": task
                            }
                        }

                        res_list.append(res)

                        time.sleep(0.1)
                        
                        with open(output_file, 'a') as fout:
                            fout.write(json.dumps(res) + '\n')
                    
                    # Clean up memory after processing each image
                    cleanup_gpu_memory()
                
                print(f"\nInference completed! Results saved to {output_file}")
                
            except Exception as e:
                print(f"Error during evaluation: {e}")
                # Save partial results
                with open(output_file, 'w') as fout:
                    for res in res_list:
                        fout.write(json.dumps(res) + '\n')
                print(f"Partial results saved to {output_file}")
                
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
                task_folder = os.path.dirname(image_path)
                
                # Check if this image is in our processed images for this task
                for processed_path in processed_images.get(task_type, []):
                    if image_filename in processed_path:
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
    for engine, send_chat_request_azure in engine_configs:
        print(f"\n{'='*20} Evaluating {engine} {'='*20}")
        
        # Process each selected task
        for task in selected_tasks:
            print(f"\n{'-'*20} Task: {task} {'-'*20}")
            
            # Get appropriate question count for this task
            random_count = get_task_question_count(task)
            
            # Select attack type(s) - if all_attacks_choice is set, use that instead of asking
            if all_attacks_choice is not None:
                # Directly call select_attack with the ALL ATTACKS choice
                from adversarial_attack_config import select_attack
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
            
            # Run evaluation for each selected attack
            for output_file, img_dir, attack_name in attack_configs:
                run_evaluation(engine, send_chat_request_azure, task, random_count, output_file, img_dir, attack_name)
