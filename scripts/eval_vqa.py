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
    for path in file_paths:
        file_name = os.path.basename(path)
        _, _, accuracy = evaluator(path)
        results[file_name] = accuracy
    
    # Print comparison if we have multiple results
    if len(results) > 1:
        print("\n=== ACCURACY COMPARISON ===")
        for file_name, accuracy in results.items():
            if "_adv_cw_l2" in file_name:
                file_type = "Adversarial (CW-L2)"
            elif "_adv_cw_l0" in file_name:
                file_type = "Adversarial (CW-L0)"
            elif "_adv_cw_linf" in file_name:
                file_type = "Adversarial (CW-L∞)"
            elif "_adv_fgsm" in file_name:
                file_type = "Adversarial (FGSM)"
            elif "_adv" in file_name:
                file_type = "Adversarial (PGD)"
            else:
                file_type = "Original"
            print(f"{file_type} ({file_name}): {accuracy:.2f}%")
        
        # If we have original and adversarial results, calculate the differences
        orig_file = next((f for f in results.keys() if "_adv" not in f), None)
        
        if orig_file:
            orig_acc = results[orig_file]
            
            # Check for PGD adversarial file
            pgd_file = next((f for f in results.keys() if "_adv" in f and "_fgsm" not in f and "_cw" not in f), None)
            if pgd_file:
                pgd_acc = results[pgd_file]
                pgd_diff = pgd_acc - orig_acc
                if pgd_diff > 0:
                    print(f"PGD: +{abs(pgd_diff):.2f}% (improvement)")
                elif pgd_diff < 0:
                    print(f"PGD: -{abs(pgd_diff):.2f}% (degradation)")
                else:
                    print(f"PGD: 0.00% (no change)")
            
            # Check for FGSM adversarial file
            fgsm_file = next((f for f in results.keys() if "_adv_fgsm" in f), None)
            if fgsm_file:
                fgsm_acc = results[fgsm_file]
                fgsm_diff = fgsm_acc - orig_acc
                if fgsm_diff > 0:
                    print(f"FGSM: +{abs(fgsm_diff):.2f}% (improvement)")
                elif fgsm_diff < 0:
                    print(f"FGSM: -{abs(fgsm_diff):.2f}% (degradation)")
                else:
                    print(f"FGSM: 0.00% (no change)")
                
            # Check for CW-L2 adversarial file
            cw_l2_file = next((f for f in results.keys() if "_adv_cw_l2" in f), None)
            if cw_l2_file:
                cw_l2_acc = results[cw_l2_file]
                cw_l2_diff = cw_l2_acc - orig_acc
                if cw_l2_diff > 0:
                    print(f"CW-L2: +{abs(cw_l2_diff):.2f}% (improvement)")
                elif cw_l2_diff < 0:
                    print(f"CW-L2: -{abs(cw_l2_diff):.2f}% (degradation)")
                else:
                    print(f"CW-L2: 0.00% (no change)")
                
            # Check for CW-L0 adversarial file
            cw_l0_file = next((f for f in results.keys() if "_adv_cw_l0" in f), None)
            if cw_l0_file:
                cw_l0_acc = results[cw_l0_file]
                cw_l0_diff = cw_l0_acc - orig_acc
                if cw_l0_diff > 0:
                    print(f"CW-L0: +{abs(cw_l0_diff):.2f}% (improvement)")
                elif cw_l0_diff < 0:
                    print(f"CW-L0: -{abs(cw_l0_diff):.2f}% (degradation)")
                else:
                    print(f"CW-L0: 0.00% (no change)")
                
            # Check for CW-L∞ adversarial file
            cw_linf_file = next((f for f in results.keys() if "_adv_cw_linf" in f), None)
            if cw_linf_file:
                cw_linf_acc = results[cw_linf_file]
                cw_linf_diff = cw_linf_acc - orig_acc
                if cw_linf_diff > 0:
                    print(f"CW-L∞: +{abs(cw_linf_diff):.2f}% (improvement)")
                elif cw_linf_diff < 0:
                    print(f"CW-L∞: -{abs(cw_linf_diff):.2f}% (degradation)")
                else:
                    print(f"CW-L∞: 0.00% (no change)")


if __name__ == "__main__":
    # Select engine
    engine = select_engine()
    
    # Fixed task
    task = "chart"
    
    # Fixed sample count
    random_count = 17
    
    # Evaluate all files for this engine and task
    evaluate_all_files(engine, task, random_count)
