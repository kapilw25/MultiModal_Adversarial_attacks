import json
import re
import os
import sys
import glob
from rouge import Rouge
from tqdm import tqdm
import spacy
import nltk

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
    
    # Check if this is an adversarial evaluation
    is_adversarial = "_adv" in path
    file_type = "Adversarial" if is_adversarial else "Original"
    
    if len(eval_file) - summary_cnt > 0:
        accuracy = len(ok_results) / (len(eval_file) - summary_cnt) * 100
        print(f'{file_type} Accuracy: {accuracy:.2f}%')

    if summary_cnt > 0:
        print(f'{file_type} Summary Rouge-L Score: {summary_score / summary_cnt:.2f}')

    assert len(ok_results) + len(bad_results) == len(eval_file) - summary_cnt
    return ok_results, bad_results, accuracy if len(eval_file) - summary_cnt > 0 else 0


def extract_marker(s):
    pattern = r'(?:[A-Za-z][0-9]|[0-9][A-Za-z])'
    marker_list = []
    for match in re.finditer(pattern, s):
        marker_list.append(match[0])

    return marker_list

# For map evaluation
def evaluator_map(path):
    # Check if file exists
    if not os.path.exists(path):
        print(f"Error: File {path} not found.")
        return None, None, 0
        
    print(f"Evaluating {path}...")
    
    eval_file = []
    with open(path) as f:
        for line in f:
            eval_file.append(json.loads(line))
            
    # Check if maze dataset exists
    maze_path = './maze_dataset2/eval_3k.json'
    if not os.path.exists(maze_path):
        print(f"Error: Maze dataset file {maze_path} not found.")
        return None, None, 0
            
    data_file = []
    with open(maze_path, 'r') as f:
        for line in f:
            data_file.append(json.loads(line))

    ok_res = []
    bad_res = []

    score = 0.0
    for result in eval_file:
        index = 0
        pr = result['text']
        gt = result['truth']

        pr_list = extract_marker(pr)
        while data_file[index]['question_id'] != result['question_id']:
            index += 1
        gt_list = data_file[index]['markers']
        # gt_list = result['markers']
        # gt_list = extract_marker(gt)

        if len(gt_list) == 0:
            continue

        pr_list = list(dict.fromkeys(pr_list)) # remove duplicates

        cnt = 0
        match_index = []
        for i in range(len(pr_list)): # check if pr_list[i] in gt_list
            if pr_list[i] in gt_list:
                cnt += 1

        if cnt / len(gt_list) >= 0.9:
            ok_res.append(result)
        elif cnt / len(gt_list) <= 0.1:
            bad_res.append(result)

        score = score + cnt / len(gt_list)

    # Check if this is an adversarial evaluation
    is_adversarial = "_adv" in path
    file_type = "Adversarial" if is_adversarial else "Original"
    
    accuracy = score / len(eval_file) * 100
    print(f'{file_type} Accuracy: {accuracy:.2f}%')
    return ok_res, bad_res, accuracy


def select_engine():
    """Interactive function to select the engine to evaluate"""
    print("\nSelect the engine to evaluate:")
    print("  [1] OpenAI GPT-4o")
    print("  [2] Qwen25_VL_3B")
    
    while True:
        choice = input("\nEnter your choice (1 or 2): ")
        if choice == '1':
            engine = 'gpt4o'
            print(f"Selected: {engine}")
            return engine
        elif choice == '2':
            engine = 'Qwen25_VL_3B'
            print(f"Selected: {engine}")
            return engine
        else:
            print("Invalid choice. Please enter 1 or 2.")


def select_task():
    """Interactive function to select the task"""
    print("\nSelect the task to evaluate:")
    print("  [1] chart")
    print("  [2] table")
    print("  [3] dashboard")
    print("  [4] flowchart")
    print("  [5] relation_graph")
    print("  [6] floor_plan")
    print("  [7] visual_puzzle")
    print("  [8] map")
    
    task_mapping = {
        '1': 'chart',
        '2': 'table',
        '3': 'dashboard',
        '4': 'flowchart',
        '5': 'relation_graph',
        '6': 'floor_plan',
        '7': 'visual_puzzle',
        '8': 'map'
    }
    
    while True:
        choice = input("\nEnter your choice (1-8): ")
        if choice in task_mapping:
            selected_task = task_mapping[choice]
            print(f"Selected task: {selected_task}")
            return selected_task
        else:
            print("Invalid choice. Please enter a number between 1 and 8.")


def evaluate_all_files(engine, task, random_count=17):
    """Evaluate all files for a given engine and task"""
    # Base directory for results
    dir_path = f'results/{engine}'
    
    # Pattern for finding all relevant files
    pattern = f'eval_{engine}_{task}_{random_count}*.json'
    
    # Find all matching files
    file_paths = glob.glob(os.path.join(dir_path, pattern))
    
    if not file_paths:
        print(f"No evaluation files found matching pattern: {pattern}")
        return
    
    print(f"Found {len(file_paths)} evaluation files:")
    for i, path in enumerate(file_paths):
        print(f"  [{i+1}] {os.path.basename(path)}")
    
    # Evaluate each file
    results = {}
    for path in file_paths:
        file_name = os.path.basename(path)
        is_map = task == 'map'
        
        if is_map:
            _, _, accuracy = evaluator_map(path)
        else:
            _, _, accuracy = evaluator(path)
        
        results[file_name] = accuracy
    
    # Print comparison if we have both original and adversarial results
    if len(results) > 1:
        print("\n=== ACCURACY COMPARISON ===")
        for file_name, accuracy in results.items():
            file_type = "Adversarial" if "_adv" in file_name else "Original"
            print(f"{file_type} ({file_name}): {accuracy:.2f}%")
        
        # If we have both original and adversarial, calculate the difference
        orig_file = next((f for f in results.keys() if "_adv" not in f), None)
        adv_file = next((f for f in results.keys() if "_adv" in f), None)
        
        if orig_file and adv_file:
            orig_acc = results[orig_file]
            adv_acc = results[adv_file]
            diff = orig_acc - adv_acc
            print(f"\nAccuracy drop due to adversarial attack: {diff:.2f}%")


if __name__ == "__main__":
    # Select engine
    engine = select_engine()
    
    # Select task
    task = select_task()
    
    # Fixed sample count
    random_count = 17
    
    # Evaluate all files for this engine and task
    evaluate_all_files(engine, task, random_count)
