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
    if "_adv_cw_l2" in path:
        file_type = "CW-L2 Adversarial"
    elif "_adv_cw_l0" in path:
        file_type = "CW-L0 Adversarial"
    elif "_adv_cw_linf" in path:
        file_type = "CW-L∞ Adversarial"
    elif "_adv_fgsm" in path:
        file_type = "FGSM Adversarial"
    elif "_adv_lbfgs" in path:
        file_type = "L-BFGS Adversarial"
    elif "_adv_jsma" in path:
        file_type = "JSMA Adversarial"
    elif "_adv_deepfool" in path:
        file_type = "DeepFool Adversarial"
    elif "_adv" in path:
        file_type = "PGD Adversarial"
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
    print("\nSelect the engine to evaluate:")
    print("  [1] OpenAI GPT-4o")
    print("  [2] Qwen25_VL_3B")
    print("  [3] ALL")
    
    while True:
        choice = input("\nEnter your choice (1, 2, or 3): ")
        if choice == '1':
            engine = 'gpt4o'
            print(f"Selected: {engine}")
            return [engine]
        elif choice == '2':
            engine = 'Qwen25_VL_3B'
            print(f"Selected: {engine}")
            return [engine]
        elif choice == '3':
            print("Selected: ALL engines")
            return ['gpt4o', 'Qwen25_VL_3B']
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def evaluate_all_files(engine, task="chart", random_count=17):
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
    file_types = {}
    for path in file_paths:
        file_name = os.path.basename(path)
        _, _, accuracy, file_type = evaluator(path)
        results[file_name] = accuracy
        file_types[file_name] = file_type
    
    # Print comparison if we have multiple results
    if len(results) > 1:
        # If we have original and adversarial results, calculate the differences
        orig_file = next((f for f in results.keys() if "_adv" not in f), None)
        
        if orig_file:
            orig_acc = results[orig_file]
            
            print(f"\n=== ACCURACY CHANGES FOR {engine.upper()} ===")
            change_data = []
            
            # Add the original row as baseline reference
            change_data.append(["Original", f"{orig_acc:.2f}%", f"{orig_acc:.2f}%", "0.00%", "Baseline"])
            
            # Check for all attack types
            attack_types = {
                "PGD": next((f for f in results.keys() if "_adv" in f and "_fgsm" not in f and "_cw" not in f and "_lbfgs" not in f and "_jsma" not in f and "_deepfool" not in f), None),
                "FGSM": next((f for f in results.keys() if "_adv_fgsm" in f), None),
                "CW-L2": next((f for f in results.keys() if "_adv_cw_l2" in f), None),
                "CW-L0": next((f for f in results.keys() if "_adv_cw_l0" in f), None),
                "CW-L∞": next((f for f in results.keys() if "_adv_cw_linf" in f), None),
                "L-BFGS": next((f for f in results.keys() if "_adv_lbfgs" in f), None),
                "JSMA": next((f for f in results.keys() if "_adv_jsma" in f), None),
                "DeepFool": next((f for f in results.keys() if "_adv_deepfool" in f), None)
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


if __name__ == "__main__":
    # Select engine(s)
    engines = select_engine()
    
    # Fixed task
    task = "chart"
    
    # Fixed sample count
    random_count = 17
    
    # Evaluate all files for each selected engine
    for engine in engines:
        print(f"\n{'='*20} Evaluating {engine} {'='*20}")
        evaluate_all_files(engine, task, random_count)
