import json
import re
import os
import sys
import glob
from rouge import Rouge
from tqdm import tqdm
import spacy
import nltk
import tabulate

# Download wordnet if not already downloaded
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn

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
    
    # Determine file type based on filename
    if "_BB_cw_l2" in path:
        file_type = "CW-L2 Adversarial"
    elif "_BB_cw_l0" in path:
        file_type = "CW-L0 Adversarial"
    elif "_BB_cw_linf" in path:
        file_type = "CW-L∞ Adversarial"
    elif "_BB_fgsm" in path:
        file_type = "FGSM Adversarial"
    elif "_BB_lbfgs" in path:
        file_type = "L-BFGS Adversarial"
    elif "_BB_jsma" in path:
        file_type = "JSMA Adversarial"
    elif "_BB_deepfool" in path:
        file_type = "DeepFool Adversarial"
    elif "_BB_pgd" in path:
        file_type = "PGD Adversarial"
    elif "_BB_square" in path:
        file_type = "Square Adversarial"
    elif "_BB_hop_skip_jump" in path:
        file_type = "HopSkipJump Adversarial"
    elif "_BB_pixel" in path:
        file_type = "Pixel Adversarial"
    elif "_BB_simba" in path:
        file_type = "SimBA Adversarial"
    elif "_BB_spatial" in path:
        file_type = "Spatial Transformation Adversarial"
    elif "_BB_query_efficient_bb" in path:
        file_type = "Query-Efficient Black-box Adversarial"
    elif "_BB_zoo" in path:
        file_type = "ZOO Adversarial"
    elif "_BB_boundary" in path:
        file_type = "Boundary Adversarial"
    elif "_BB_geoda" in path:
        file_type = "GeoDA Adversarial"
    else:
        file_type = "Original"
    
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
    from local_model_utils import list_available_models
    
    # Get all available local models
    local_models = list_available_models()
    
    # Create the menu options
    print("\nSelect the engine to evaluate:")
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
                engine = 'gpt4o'
                print(f"Selected: {engine}")
                return [engine]
            elif 2 <= choice_num <= len(local_models) + 1:
                # Selected a local model
                engine = local_models[choice_num - 2]
                print(f"Selected: {engine}")
                return [engine]
            elif choice_num == all_option:
                print("Selected: ALL engines")
                engines = ['gpt4o'] + local_models
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


def evaluate_all_files(engine, task, random_count=None):
    """Evaluate all files for a given engine and task"""
    # Get appropriate question count for this task if not provided
    if random_count is None:
        random_count = get_task_question_count(task)
    
    # Base directory for results
    dir_path = f'results/models/{engine}'
    
    # Pattern for finding all relevant files
    pattern = f'eval_{engine}_{task}_{random_count}*.json'
    
    # Find all matching files
    file_paths = glob.glob(os.path.join(dir_path, pattern))
    
    if not file_paths:
        print(f"No evaluation files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(file_paths)} evaluation files for task '{task}':")
    for i, path in enumerate(file_paths):
        print(f"  [{i+1}] {os.path.basename(path)}")
    
    # Evaluate each file
    results = {}
    file_types = {}
    for path in file_paths:
        file_name = os.path.basename(path)
        _, _, accuracy, file_type = evaluator(path)
        results[file_name] = accuracy
        file_types[file_name] = file_type
    
    # Print comparison if we have multiple results
    if len(results) > 1:
        # Find the original file (the one without any attack suffix)
        # This is more reliable than checking for "_adv" which might not be present
        orig_file = next((f for f in results.keys() if all(attack not in f for attack in 
                                                          ["_BB_pgd", "_BB_fgsm", "_BB_cw_l2", "_BB_cw_l0", 
                                                           "_BB_cw_linf", "_BB_lbfgs", "_BB_jsma", "_BB_deepfool", 
                                                           "_BB_square", "_BB_hop_skip_jump", "_BB_pixel", "_BB_simba",
                                                           "_BB_spatial", "_BB_query_efficient_bb", "_BB_zoo",
                                                           "_BB_boundary", "_BB_geoda"])), None)
        
        if orig_file:
            orig_acc = results[orig_file]
            
            print(f"\n=== ACCURACY CHANGES FOR {engine.upper()} ON {task.upper()} ===")
            change_data = []
            
            # Add the original row as baseline reference
            change_data.append(["Original", f"{orig_acc:.2f}%", f"{orig_acc:.2f}%", "0.00%", "Baseline"])
            
            # Check for all attack types
            attack_types = {
                "PGD": next((f for f in results.keys() if "_BB_pgd" in f), None),
                "FGSM": next((f for f in results.keys() if "_BB_fgsm" in f), None),
                "CW-L2": next((f for f in results.keys() if "_BB_cw_l2" in f), None),
                "CW-L0": next((f for f in results.keys() if "_BB_cw_l0" in f), None),
                "CW-L∞": next((f for f in results.keys() if "_BB_cw_linf" in f), None),
                "L-BFGS": next((f for f in results.keys() if "_BB_lbfgs" in f), None),
                "JSMA": next((f for f in results.keys() if "_BB_jsma" in f), None),
                "DeepFool": next((f for f in results.keys() if "_BB_deepfool" in f), None),
                "Square": next((f for f in results.keys() if "_BB_square" in f), None),
                "HopSkipJump": next((f for f in results.keys() if "_BB_hop_skip_jump" in f), None),
                "Pixel": next((f for f in results.keys() if "_BB_pixel" in f), None),
                "SimBA": next((f for f in results.keys() if "_BB_simba" in f), None),
                "Spatial": next((f for f in results.keys() if "_BB_spatial" in f), None),
                "Query-Efficient BB": next((f for f in results.keys() if "_BB_query_efficient_bb" in f), None),
                "ZOO": next((f for f in results.keys() if "_BB_zoo" in f), None),
                "Boundary": next((f for f in results.keys() if "_BB_boundary" in f), None),
                "GeoDA": next((f for f in results.keys() if "_BB_geoda" in f), None)
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
            save_results_to_json(engine, task, change_data)


def save_results_to_json(engine, task, change_data):
    """Save evaluation results to a JSON file for database storage"""
    # Skip if there's no data
    if not change_data:
        return
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Path to the JSON file
    json_path = f"results/robustness_{task}.json"
    
    # Initialize the data structure if the file doesn't exist
    if not os.path.exists(json_path):
        data = {
            "models": {},
            "metadata": {
                "task_name": task,
                "timestamp": "2025-07-19T07:00:00Z",
                "version": "1.0"
            }
        }
    else:
        # Load existing data
        with open(json_path, 'r') as f:
            data = json.load(f)
    
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
        
        # Store in the data structure
        data["models"][engine][attack_type] = {
            "accuracy": attack_accuracy,
            "change": change,
            "effect": effect
        }
    
    # Save to file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {json_path}")


if __name__ == "__main__":
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
