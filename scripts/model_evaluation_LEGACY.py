import json
import re
import os
import sys
import glob
import subprocess
from rouge import Rouge
from tqdm import tqdm
import spacy
import nltk
import tabulate

# Add utils directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import ground truth populator for ensuring data exists
from ground_truth_populator import populate_ground_truth_questions

# Download wordnet if not already downloaded
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn

def ensure_ground_truth_data():
    """Ensure ground truth data exists in database before evaluation"""
    from centralized_database import CentralizedDB

    try:
        db = CentralizedDB()
        questions = db.get_ground_truth_questions()

        if not questions:
            print("⚠️  No ground truth data found in database. Populating from source files...")
            if populate_ground_truth_questions(overwrite=False):
                print("✅ Ground truth data populated successfully")
                return True
            else:
                print("❌ Failed to populate ground truth data")
                return False
        else:
            print(f"✅ Found {len(questions)} ground truth questions in database")
            return True
    except Exception as e:
        print(f"⚠️  Error checking ground truth data: {e}")
        # Try to populate anyway
        print("🔄 Attempting to populate ground truth data...")
        return populate_ground_truth_questions(overwrite=False)

def update_database_from_json(json_path):
    """
    Update the SQLite database by calling database_manager.main() directly
    
    Args:
        json_path (str): Path to the JSON file to process
    """
    try:
        # Import database manager functions
        import database_manager
        
        # Temporarily update the JSON_PATH in the database_manager module
        original_path = database_manager.JSON_PATH
        database_manager.JSON_PATH = json_path
        
        print(f"🔄 Updating database with {json_path}...")
        
        # Run the full database manager main() function
        # This will handle everything including:
        # - Directory setup and backups
        # - Database creation and population  
        # - Attack parameters integration
        # - View creation and verification
        database_manager.main()
            
        # Restore original path
        database_manager.JSON_PATH = original_path
        
    except ImportError:
        print("❌ Could not import database_manager - falling back to subprocess")
        # Fallback to subprocess if import fails
        try:
            result = subprocess.run([sys.executable, "scripts/utils/database_manager.py"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Database updated successfully (subprocess)")
            else:
                print(f"⚠️  Database update failed: {result.stderr}")
        except Exception as e:
            print(f"❌ Subprocess fallback failed: {e}")
            
    except Exception as e:
        print(f"❌ Error updating database: {e}")
        
        # Restore original path in case of error
        try:
            import database_manager
            database_manager.JSON_PATH = original_path
        except:
            pass

def are_synonyms(word1, word2):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    for s1 in synsets1:
        for s2 in synsets2:
            if s1 == s2:
                return True
    return False

def is_number(s):
    try:
        s = s.replace(',', '')
        float(s)
        return True
    except ValueError:
        return False


def str_to_num(s):
    s = s.replace(',', '')
    if is_number(s):
        return float(s)


def extract_number(s):
    pattern = r'[\d]+[\,\d]*[\.]{0,1}[\d]+'

    if re.search(pattern, s) is not None:
        result = []
        for catch in re.finditer(pattern, s):
            result.append(catch[0])
        return result
    else:
        return []


def relaxed_accuracy(pr, gt):
    return abs(float(pr) - float(gt)) <= 0.05 * abs(float(gt))


nlp = spacy.load('en_core_web_sm')

def remove_units(text):
    doc = nlp(text)
    new_text = []
    i = 0

    while i < len(doc):
        token = doc[i]
        if token.pos_ == 'NUM':
            j = i + 1

            possible_unit_parts = []
            while j < len(doc) and (doc[j].pos_ == 'NOUN' or doc[j].pos_ == 'ADP' or doc[j].tag_ in ['NN', 'IN']):
                possible_unit_parts.append(doc[j].text)
                j += 1
            if possible_unit_parts:
                new_text.append(token.text)  
                i = j 
                continue
        new_text.append(token.text)
        i += 1

    return ' '.join(new_text)

# For evalution except map
def evaluator(path):
    print(f"Evaluating {path}...")
    
    # Check if file exists
    if not os.path.exists(path):
        print(f"Error: File {path} not found.")
        return None, None, 0
        
    eval_file = []
    with open(path) as f:
        for line in f:
            eval_file.append(json.loads(line))

    ok_results = []
    bad_results = []
    structural_cnt = 0
    data_extraction_cnt = 0
    math_reasoning_cnt = 0
    color_cnt = 0
    caption_cnt = 0
    summary_cnt = 0

    rouge = Rouge()
    summary_score = 0.0

    for result in tqdm(eval_file):
        pr = result['text'] # predicted response
        gt = result['truth'] # ground truth

        pr = pr.strip().lower()
        gt = gt.strip().lower()

        pattern = r'the answer is (.*?)(?:\.\s|$)'
        match = re.search(pattern, pr)
        if match:
            pr = match.group(1)

        match = re.search(pattern, gt)
        if match:
            gt = match.group(1)

        if len(pr) > 0:
            if pr[-1] == '.':
                pr = pr[:-1]
                if len(pr) >= 1 and pr[-1] == '.':
                    pr = pr[:-1]
            if len(pr) >= 1 and pr[-1] == '%':
                pr = pr[:-1]
            if pr.endswith("\u00b0c"):
                pr = pr[:-2]

        if len(gt) > 0:
            if gt[-1] == '.':
                gt = gt[:-1]
            if gt[-1] == '%':
                gt = gt[:-1]
            if gt.endswith("\u00b0c"):
                gt = gt[:-2]

        pr = remove_units(pr)
        gt = remove_units(gt)

        numeric_values = extract_number(pr)

        if result['type'] == 'STRUCTURAL':
            structural_cnt += 1
        elif result['type'] == 'DATA_EXTRACTION':
            data_extraction_cnt += 1
        elif result['type'] == 'MATH_REASONING':
            math_reasoning_cnt += 1
        elif result['type'] == 'COLOR':
            color_cnt += 1
        elif result['type'] == 'CAPTION':
            caption_cnt += 1
        elif result['type'] == 'SUMMARY':
            summary_cnt += 1

        if result['type'] == 'SUMMARY':
            if pr != '':
                summary_score += rouge.get_scores(gt, pr, avg=True)['rouge-l']['f']
            continue

        if is_number(pr) and is_number(gt) and relaxed_accuracy(str_to_num(pr), str_to_num(gt)):
            ok_results.append(result)
        elif is_number(gt):
            flag = False
            for v in numeric_values:
                if relaxed_accuracy(str_to_num(v), str_to_num(gt)):
                    ok_results.append(result)
                    flag = True
                    break
            if not flag:
                bad_results.append(result)
        elif pr in ['a', 'b', 'c', 'd'] or gt in ['a', 'b', 'c', 'd']:
            if pr == gt:
                ok_results.append(result)
            else:
                bad_results.append(result)
        elif len(gt) >= 2 and gt[0] == '[' and gt[-1] == ']':
            if pr == gt:
                ok_results.append(result)
            else:
                bad_results.append(result)
        elif len(gt) >= 2 and gt[0] == '(' and gt[-1] == ')':
            first = gt[1]
            second = gt[-2]
            pr_values = extract_number(pr)
            if len(pr_values) == 2 and pr_values[0] == first and pr_values[1] == second:
                ok_results.append(result)
            else:
                bad_results.append(result)
        elif pr != "" and pr in gt or gt in pr:
            ok_results.append(result)
        elif pr != "" and are_synonyms(pr, gt):
            ok_results.append(result)
        else:
            bad_results.append(result)
    
    # Determine file type based on file path structure (NEW approach)
    if "/adversarial/whitebox/pgd/" in path:
        file_type = "PGD Adversarial"
    elif "/adversarial/whitebox/fgsm/" in path:
        file_type = "FGSM Adversarial"
    elif "/adversarial/whitebox/cw_linf/" in path:
        file_type = "CW-L∞ Adversarial"
    elif "/adversarial/whitebox/deepfool/" in path:
        file_type = "DeepFool Adversarial"
    elif "/adversarial/blackbox/square/" in path:
        file_type = "Square Adversarial"
    elif "/adversarial/blackbox/simba/" in path:
        file_type = "SimBA Adversarial"
    elif "/adversarial/blackbox/boundary/" in path:
        file_type = "Boundary Adversarial"
    elif "/adversarial/blackbox/pixel/" in path:
        file_type = "Pixel Adversarial"
    elif "/adversarial/blackbox/spatial/" in path:
        file_type = "Spatial Transformation Adversarial"
    elif "/clean/" in path:
        file_type = "Original"
    else:
        file_type = "Unknown"
    
    if len(eval_file) - summary_cnt > 0:
        accuracy = len(ok_results) / (len(eval_file) - summary_cnt) * 100
        print(f'{file_type} Accuracy: {accuracy:.2f}%')

    if summary_cnt > 0:
        print(f'{file_type} Summary Rouge-L Score: {summary_score / summary_cnt:.2f}')

    assert len(ok_results) + len(bad_results) == len(eval_file) - summary_cnt
    return ok_results, bad_results, accuracy if len(eval_file) - summary_cnt > 0 else 0, file_type


def select_engine():
    """Interactive function to select the engine to evaluate"""
    # Import the list_available_models function from local_model_utils
    from vlm_local_client import list_available_models
    
    # Get all available local models
    local_models = list_available_models()
    
    # Create the menu options
    print("\nSelect the engine to evaluate:")
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
            #     engine = 'gpt4o'
            #     print(f"Selected: {engine}")
            #     return [engine]
            if 1 <= choice_num <= len(local_models):
                # Selected a local model (adjusted index since we removed GPT-4o)
                engine = local_models[choice_num - 1]
                print(f"Selected: {engine}")
                return [engine]
            elif choice_num == all_option:
                print("Selected: ALL LOCAL MODELS")
                engines = local_models  # Only local models, no GPT-4o
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


def aggregate_performance_metrics(file_paths):
    """
    Aggregate performance metrics from individual inference JSON files.
    
    Args:
        file_paths (list): List of file paths to process
        
    Returns:
        dict: Aggregated performance metrics by attack type
    """
    performance_data = {}
    
    for path in file_paths:
        file_name = os.path.basename(path)
        
        # Determine attack type from file path structure (NEW approach)
        if "/adversarial/whitebox/pgd/" in path:
            attack_type = "PGD"
        elif "/adversarial/whitebox/fgsm/" in path:
            attack_type = "FGSM"
        elif "/adversarial/whitebox/cw_linf/" in path:
            attack_type = "CW-L∞"
        elif "/adversarial/whitebox/deepfool/" in path:
            attack_type = "DeepFool"
        elif "/adversarial/blackbox/square/" in path:
            attack_type = "Square"
        elif "/adversarial/blackbox/simba/" in path:
            attack_type = "SimBA"
        elif "/adversarial/blackbox/boundary/" in path:
            attack_type = "Boundary"
        elif "/adversarial/blackbox/pixel/" in path:
            attack_type = "Pixel"
        elif "/adversarial/blackbox/spatial/" in path:
            attack_type = "Spatial"
        elif "/clean/" in path:
            attack_type = "Original"
        else:
            attack_type = "Unknown"
        
        # Read individual JSON lines and extract performance metrics
        try:
            with open(path, 'r') as f:
                metrics_list = []
                total_questions = 0
                cached_loads = 0
                total_loads = 0
                
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        total_questions += 1
                        
                        # Extract performance metrics if present
                        if 'performance_metrics' in entry:
                            pm = entry['performance_metrics']
                            metrics_list.append({
                                'inference_time': pm.get('inference_time_seconds', 0),
                                'gpu_allocated': pm.get('gpu_memory', {}).get('after_inference_mb', 0),
                                'gpu_peak': pm.get('gpu_memory', {}).get('peak_mb', 0),
                                'gpu_reserved': pm.get('gpu_memory', {}).get('reserved_mb', 0),
                                'gpu_total': pm.get('gpu_memory', {}).get('total_gpu_mb', 0),
                                'cpu_memory': pm.get('cpu_memory', {}).get('after_inference_mb', 0),
                                'loading_time': pm.get('model_loading', {}).get('loading_time_seconds', 0),
                                'was_cached': pm.get('model_loading', {}).get('was_cached', False)
                            })
                            
                            total_loads += 1
                            if pm.get('model_loading', {}).get('was_cached', False):
                                cached_loads += 1
                
                # Aggregate metrics
                if metrics_list:
                    avg_metrics = {
                        'avg_inference_time_seconds': sum(m['inference_time'] for m in metrics_list) / len(metrics_list),
                        'avg_gpu_memory_allocated_mb': sum(m['gpu_allocated'] for m in metrics_list) / len(metrics_list),
                        'avg_gpu_memory_peak_mb': sum(m['gpu_peak'] for m in metrics_list) / len(metrics_list),
                        'avg_gpu_memory_reserved_mb': sum(m['gpu_reserved'] for m in metrics_list) / len(metrics_list),
                        'total_gpu_memory_mb': metrics_list[0]['gpu_total'] if metrics_list else 0,  # Same for all
                        'avg_cpu_memory_mb': sum(m['cpu_memory'] for m in metrics_list) / len(metrics_list),
                        'model_loading_time_seconds': sum(m['loading_time'] for m in metrics_list),  # Total loading time
                        'cache_hit_ratio': cached_loads / total_loads if total_loads > 0 else 0,
                        'total_questions': total_questions
                    }
                    
                    performance_data[attack_type] = avg_metrics
                    
        except Exception as e:
            print(f"Warning: Could not extract performance metrics from {path}: {e}")
    
    return performance_data


def evaluate_all_files(engine, task, random_count=None):
    """Evaluate all files for a given engine and task"""
    # Get appropriate question count for this task if not provided
    if random_count is None:
        random_count = get_task_question_count(task)
    
    # NEW: Updated base directory for results (matches model_inference.py output)
    base_dir = f'results/inference/{engine}'
    
    # Find clean (original) files
    clean_pattern = f'{base_dir}/clean/{task}/eval_{engine}_{task}_{random_count}*.json'
    clean_files = glob.glob(clean_pattern)
    
    # Find adversarial files
    adversarial_files = []
    adversarial_base = f'{base_dir}/adversarial'
    
    # Check for whitebox attacks
    whitebox_dir = f'{adversarial_base}/whitebox'
    if os.path.exists(whitebox_dir):
        for attack in ['pgd', 'fgsm', 'cw_linf', 'deepfool']:
            attack_pattern = f'{whitebox_dir}/{attack}/ssim_085/{task}/eval_{engine}_{task}_{random_count}*.json'
            adversarial_files.extend(glob.glob(attack_pattern))
    
    # Check for blackbox attacks  
    blackbox_dir = f'{adversarial_base}/blackbox'
    if os.path.exists(blackbox_dir):
        for attack in ['square', 'simba', 'boundary', 'pixel', 'spatial']:
            attack_pattern = f'{blackbox_dir}/{attack}/ssim_085/{task}/eval_{engine}_{task}_{random_count}*.json'
            adversarial_files.extend(glob.glob(attack_pattern))
    
    # Combine all files
    file_paths = clean_files + adversarial_files
    
    if not file_paths:
        print(f"No evaluation files found for {engine} on task '{task}'")
        print(f"Searched in: {base_dir}")
        return
    
    print(f"Found {len(file_paths)} evaluation files for task '{task}':")
    for i, path in enumerate(file_paths):
        print(f"  [{i+1}] {os.path.relpath(path, 'results/inference')}")
    
    # Aggregate performance metrics from all files
    print(f"Aggregating performance metrics for {engine}...")
    performance_metrics = aggregate_performance_metrics(file_paths)
    
    # Evaluate each file
    results = {}
    file_types = {}
    for path in file_paths:
        file_name = os.path.basename(path)
        _, _, accuracy, file_type = evaluator(path)
        results[file_name] = accuracy
        file_types[file_name] = file_type
    
    # Generate outputs even with only clean files
    if len(results) >= 1:
        # Find the original file (clean file)
        orig_file = next((f for f in results.keys() if "/clean/" in f), None)
        
        if not orig_file:
            # Fallback: use the first file if no clean file found
            orig_file = list(results.keys())[0]
        
        orig_acc = results[orig_file]
        
        print(f"\n=== ACCURACY RESULTS FOR {engine.upper()} ON {task.upper()} ===")
        change_data = []
        
        # Add the original row as baseline reference
        change_data.append(["Original", f"{orig_acc:.2f}%", f"{orig_acc:.2f}%", "0.00%", "Baseline"])
        
        # Check for all attack types (only add if files exist)
        attack_types = {
            "PGD": next((f for f in results.keys() if "/adversarial/whitebox/pgd/" in f), None),
            "FGSM": next((f for f in results.keys() if "/adversarial/whitebox/fgsm/" in f), None),
            "CW-L∞": next((f for f in results.keys() if "/adversarial/whitebox/cw_linf/" in f), None),
            "DeepFool": next((f for f in results.keys() if "/adversarial/whitebox/deepfool/" in f), None),
            "Square": next((f for f in results.keys() if "/adversarial/blackbox/square/" in f), None),
            "SimBA": next((f for f in results.keys() if "/adversarial/blackbox/simba/" in f), None),
            "Boundary": next((f for f in results.keys() if "/adversarial/blackbox/boundary/" in f), None),
            "Pixel": next((f for f in results.keys() if "/adversarial/blackbox/pixel/" in f), None),
            "Spatial": next((f for f in results.keys() if "/adversarial/blackbox/spatial/" in f), None),
        }
        
        for attack_name, attack_file in attack_types.items():
            if attack_file:
                attack_acc = results[attack_file]
                diff = attack_acc - orig_acc
                
                if diff > 0:
                    change_type = "Improvement"
                    change_str = f"+{abs(diff):.2f}%"
                elif diff < 0:
                    change_type = "Degradation"
                    change_str = f"-{abs(diff):.2f}%"
                else:
                    change_type = "No Change"
                    change_str = "0.00%"
                    
                change_data.append([attack_name, f"{orig_acc:.2f}%", f"{attack_acc:.2f}%", change_str, change_type])
        
        print(tabulate.tabulate(change_data, 
                      headers=["Attack Type", f"{engine} Original", f"{engine} Attack", "Change", "Effect"], 
                      tablefmt="grid"))
        
        # Save results to JSON file for database storage
        save_results_to_json(engine, task, change_data, performance_metrics)


def save_results_to_json(engine, task, change_data, performance_metrics=None):
    """Save evaluation results to a JSON file for database storage"""
    # Skip if there's no data
    if not change_data:
        return
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Path to the JSON file
    json_path = f"results/robustness_{task}.json"
    
    # Backup existing JSON file if this is the first model being processed
    # (only backup once per task evaluation run)
    if os.path.exists(json_path):
        # Check if this is a fresh run by seeing if file contains current pipeline models
        try:
            with open(json_path, 'r') as f:
                existing_data = json.load(f)
            
            # Check if file contains outdated models (models not in current pipeline)
            from vlm_local_client import list_available_models
            current_models = set(list_available_models())
            existing_models = set(existing_data.get("models", {}).keys())
            
            # If there are models in the file that aren't in current pipeline, backup and start fresh
            outdated_models = existing_models - current_models
            if outdated_models:
                backup_json = json_path.replace('.json', '_backup.json')
                import shutil
                shutil.copy2(json_path, backup_json)
                print(f"📁 Backed up JSON with outdated models to {backup_json}")
                print(f"🗑️  Found outdated models: {', '.join(outdated_models)}")
                # Start fresh
                data = {
                    "models": {},
                    "metadata": {
                        "task_name": task,
                        "timestamp": "2025-07-19T07:00:00Z",
                        "version": "2.0"
                    }
                }
            else:
                # Load existing data (current models)
                data = existing_data
        except:
            # If can't read existing file, start fresh
            data = {
                "models": {},
                "metadata": {
                    "task_name": task,
                    "timestamp": "2025-07-19T07:00:00Z",
                    "version": "2.0"
                }
            }
    else:
        # Initialize the data structure for new file
        data = {
            "models": {},
            "metadata": {
                "task_name": task,
                "timestamp": "2025-07-19T07:00:00Z",
                "version": "2.0"
            }
        }
    
    # Initialize model data if not present
    if engine not in data["models"]:
        data["models"][engine] = {}
    
    # Process each row of change data
    for row in change_data:
        attack_type = row[0]
        original_accuracy = float(row[1].strip('%'))
        attack_accuracy = float(row[2].strip('%'))
        
        # Parse change value
        change_str = row[3]
        if change_str.startswith('+'):
            change = float(change_str.strip('+%'))
        elif change_str.startswith('-'):
            change = -float(change_str.strip('-%'))
        else:
            change = 0.0
        
        effect = row[4]
        
        # Create base entry
        entry = {
            "accuracy": attack_accuracy,
            "change": change,
            "effect": effect
        }
        
        # Add performance metrics if available
        if performance_metrics and attack_type in performance_metrics:
            entry["performance_metrics"] = performance_metrics[attack_type]
        
        # Store in the data structure
        data["models"][engine][attack_type] = entry
    
    # Save to file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {json_path}")
    print(f"Performance metrics integrated for {len(performance_metrics) if performance_metrics else 0} attack types")
    
    # Auto-update database after JSON creation
    update_database_from_json(json_path)


if __name__ == "__main__":
    # Ensure ground truth data exists before evaluation
    print("🔍 Checking ground truth data availability...")
    if not ensure_ground_truth_data():
        print("❌ Cannot proceed without ground truth data. Exiting.")
        sys.exit(1)

    # Select engine(s)
    engines = select_engine()
    
    # Select task(s)
    tasks = select_task()
    
    # Evaluate all files for each selected engine and task
    for engine in engines:
        print(f"\n{'='*20} Evaluating {engine} {'='*20}")
        
        for task in tasks:
            print(f"\n{'-'*20} Task: {task} {'-'*20}")
            
            # Get appropriate question count for this task
            random_count = get_task_question_count(task)
            
            # Evaluate all files for this engine and task
            evaluate_all_files(engine, task, random_count)
