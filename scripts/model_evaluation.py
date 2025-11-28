#!/usr/bin/env python3
"""
Simplified Model Evaluation Script (Database-First Approach)

Goal: Read from inference_results table, evaluate answers, and create extended table
with evaluation results for each question.

This creates a new table with all inference_results columns plus evaluation metrics.
"""

import os
import sys
import re
import sqlite3
from tqdm import tqdm
import nltk
from rouge import Rouge

# Download required NLTK data
nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
DB_PATH = "results/centralized_pipeline.db"

def are_synonyms(word1, word2):
    """Check if two words are synonyms using WordNet"""
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    for s1 in synsets1:
        for s2 in synsets2:
            if s1 == s2:
                return True
    return False

def is_number(s):
    """Check if string represents a number"""
    try:
        s = s.replace(',', '')
        float(s)
        return True
    except ValueError:
        return False

def str_to_num(s):
    """Convert string to number"""
    s = s.replace(',', '')
    if is_number(s):
        return float(s)

def extract_number(s):
    """Extract numbers from string"""
    pattern = r'[\d]+[\,\d]*[\.]{0,1}[\d]+'
    if re.search(pattern, s) is not None:
        result = []
        for catch in re.finditer(pattern, s):
            result.append(catch[0])
        return result
    else:
        return []

def relaxed_accuracy(pr, gt):
    """Check if prediction is within 5% of ground truth"""
    return abs(float(pr) - float(gt)) <= 0.05 * abs(float(gt))

def clean_answer_text(text):
    """Clean and normalize answer text"""
    if not text:
        return ""

    text = text.strip().lower()

    # Extract answer from "the answer is ..." pattern
    pattern = r'the answer is (.*?)(?:\.\s|$)'
    match = re.search(pattern, text)
    if match:
        text = match.group(1)

    # Remove trailing punctuation and units
    if len(text) > 0:
        if text[-1] == '.':
            text = text[:-1]
            if len(text) >= 1 and text[-1] == '.':
                text = text[:-1]
        if len(text) >= 1 and text[-1] == '%':
            text = text[:-1]
        if text.endswith("\u00b0c"):
            text = text[:-2]

    return text

def evaluate_answer(model_response, ground_truth, question_type):
    """
    Evaluate model response against ground truth
    Returns: (is_correct: bool, confidence_score: float, evaluation_method: str)

    Based on model_evaluation.py logic with question type awareness
    """
    if not model_response or not ground_truth:
        return False, 0.0, "empty_response"

    # Clean both responses
    pr = clean_answer_text(model_response)
    gt = clean_answer_text(ground_truth)

    if not pr:
        return False, 0.0, "empty_cleaned_response"

    # Special handling for SUMMARY type - use ROUGE-L score
    if question_type == 'SUMMARY':
        try:
            rouge = Rouge()
            scores = rouge.get_scores(gt, pr, avg=True)
            rouge_l_score = scores['rouge-l']['f']
            # Consider correct if ROUGE-L > 0.5
            is_correct = rouge_l_score > 0.5
            return is_correct, rouge_l_score, "rouge_l"
        except:
            return False, 0.0, "rouge_error"

    # Numerical answers with relaxed accuracy (±5%)
    if is_number(pr) and is_number(gt):
        if relaxed_accuracy(str_to_num(pr), str_to_num(gt)):
            return True, 1.0, "numerical_exact"
        else:
            return False, 0.0, "numerical_mismatch"

    # Ground truth is numerical, but response might contain numbers
    elif is_number(gt):
        numeric_values = extract_number(pr)
        for v in numeric_values:
            if relaxed_accuracy(str_to_num(v), str_to_num(gt)):
                return True, 1.0, "numerical_extracted"
        return False, 0.0, "numerical_not_found"

    # Multiple choice answers (a, b, c, d)
    elif pr in ['a', 'b', 'c', 'd'] or gt in ['a', 'b', 'c', 'd']:
        if pr == gt:
            return True, 1.0, "multiple_choice_exact"
        else:
            return False, 0.0, "multiple_choice_mismatch"

    # List/array format answers [...]
    elif len(gt) >= 2 and gt[0] == '[' and gt[-1] == ']':
        if pr == gt:
            return True, 1.0, "list_exact"
        else:
            return False, 0.0, "list_mismatch"

    # Coordinate format answers (x, y)
    elif len(gt) >= 2 and gt[0] == '(' and gt[-1] == ')':
        first = gt[1]
        second = gt[-2]
        pr_values = extract_number(pr)
        if len(pr_values) == 2 and pr_values[0] == first and pr_values[1] == second:
            return True, 1.0, "coordinate_exact"
        else:
            return False, 0.0, "coordinate_mismatch"

    # String containment
    elif pr in gt or gt in pr:
        return True, 0.8, "string_containment"

    # Synonym matching
    elif are_synonyms(pr, gt):
        return True, 0.7, "synonym_match"

    # No match found
    else:
        return False, 0.0, "no_match"

def create_results_evaluation_table():
    """Copy inference_results table and rename as results_evaluation with added evaluation columns"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if results_evaluation table already exists
    cursor.execute('''
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='results_evaluation'
    ''')

    if cursor.fetchone():
        print("✅ results_evaluation table already exists")
        conn.close()
        return

    print("🔄 Creating results_evaluation table from inference_results...")

    # Create new table by copying inference_results structure and data
    # Insert evaluation columns after [prompt, model_response, ground_truth] triplet
    cursor.execute('''
    CREATE TABLE results_evaluation AS
    SELECT
        result_id,
        question_id,
        prompt,
        model_response,
        ground_truth,
        -- NEW EVALUATION COLUMNS (added after the triplet)
        CAST(NULL AS BOOLEAN) as is_correct,
        CAST(NULL AS REAL) as confidence_score,
        CAST(NULL AS TEXT) as evaluation_method,
        -- CONTINUE WITH ORIGINAL COLUMNS
        question_type,
        answer_id,
        markers,
        model_id,
        adversarial,
        task,
        attack_type,
        epsilon_target,
        epsilon_l_inf,
        inference_image_path,
        clean_image_path,
        timestamp,
        inference_time_seconds,
        gpu_before_mb,
        gpu_after_mb,
        gpu_peak_mb,
        gpu_reserved_mb,
        total_gpu_mb,
        cpu_before_mb,
        cpu_after_mb,
        was_cached,
        loading_time_seconds,
        batch_size,
        position_in_batch,
        total_batch_questions
    FROM inference_results
    ''')

    # Create indexes for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_eval_model_id ON results_evaluation(model_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_eval_attack_type ON results_evaluation(attack_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_eval_is_correct ON results_evaluation(is_correct)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_eval_question_type ON results_evaluation(question_type)')

    # Get count for confirmation
    cursor.execute('SELECT COUNT(*) FROM results_evaluation')
    count = cursor.fetchone()[0]

    conn.commit()
    conn.close()
    print(f"✅ Created results_evaluation table with {count} records copied from inference_results")

def populate_evaluation_results():
    """Update results_evaluation table with evaluation scores"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if evaluation columns already have data
    cursor.execute('SELECT COUNT(*) FROM results_evaluation WHERE is_correct IS NOT NULL')
    already_evaluated = cursor.fetchone()[0]

    # Get total count
    cursor.execute('SELECT COUNT(*) FROM results_evaluation')
    total_count = cursor.fetchone()[0]

    if total_count == 0:
        print("❌ No data found in results_evaluation table")
        conn.close()
        return

    if already_evaluated > 0:
        print(f"⚠️  Found {already_evaluated} already evaluated records. Updating all records...")

    print(f"📊 Evaluating {total_count} records in results_evaluation table...")

    # Process in batches for memory efficiency
    batch_size = 1000

    for offset in range(0, total_count, batch_size):
        cursor.execute('''
            SELECT result_id, prompt, model_response, ground_truth, question_type
            FROM results_evaluation
            LIMIT ? OFFSET ?
        ''', (batch_size, offset))

        batch_results = cursor.fetchall()

        # Process each result in the batch
        updates = []
        for row in tqdm(batch_results, desc=f"Evaluating batch {offset//batch_size + 1}"):
            result_id, prompt, model_response, ground_truth, question_type = row

            # Evaluate the answer
            is_correct, confidence_score, evaluation_method = evaluate_answer(
                model_response, ground_truth, question_type
            )

            updates.append((is_correct, confidence_score, evaluation_method, result_id))

        # Update batch in results_evaluation
        cursor.executemany('''
            UPDATE results_evaluation
            SET is_correct = ?, confidence_score = ?, evaluation_method = ?
            WHERE result_id = ?
        ''', updates)

        conn.commit()
        print(f"   ✅ Evaluated batch {offset//batch_size + 1}: {len(updates)} records")

    conn.close()
    print(f"🎉 Evaluation complete! All {total_count} results evaluated")

def generate_evaluation_summary():
    """Generate summary statistics from evaluation results"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("\n" + "="*60)
    print("📊 EVALUATION SUMMARY")
    print("="*60)

    # Overall accuracy
    cursor.execute('''
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
            AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) * 100 as accuracy_pct
        FROM results_evaluation
    ''')
    total, correct, accuracy = cursor.fetchone()
    print(f"Overall: {correct}/{total} correct ({accuracy:.2f}%)")

    # Accuracy by model
    print(f"\n📱 Accuracy by Model:")
    cursor.execute('''
        SELECT
            model_id,
            COUNT(*) as total,
            SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
            AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) * 100 as accuracy_pct
        FROM results_evaluation
        GROUP BY model_id
        ORDER BY accuracy_pct DESC
    ''')
    for model_id, total, correct, accuracy in cursor.fetchall():
        print(f"   {model_id}: {correct}/{total} ({accuracy:.2f}%)")

    # Accuracy by attack type
    print(f"\n⚔️  Accuracy by Attack Type:")
    cursor.execute('''
        SELECT
            attack_type,
            COUNT(*) as total,
            SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
            AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) * 100 as accuracy_pct
        FROM results_evaluation
        GROUP BY attack_type
        ORDER BY accuracy_pct DESC
    ''')
    for attack_type, total, correct, accuracy in cursor.fetchall():
        print(f"   {attack_type}: {correct}/{total} ({accuracy:.2f}%)")

    # Accuracy by question type
    print(f"\n❓ Accuracy by Question Type:")
    cursor.execute('''
        SELECT
            question_type,
            COUNT(*) as total,
            SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
            AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) * 100 as accuracy_pct
        FROM results_evaluation
        GROUP BY question_type
        ORDER BY accuracy_pct DESC
    ''')
    for question_type, total, correct, accuracy in cursor.fetchall():
        print(f"   {question_type}: {correct}/{total} ({accuracy:.2f}%)")

    # Evaluation method distribution
    print(f"\n🔍 Evaluation Method Distribution:")
    cursor.execute('''
        SELECT
            evaluation_method,
            COUNT(*) as count,
            AVG(CASE WHEN is_correct = 1 THEN 1.0 ELSE 0.0 END) * 100 as success_rate
        FROM results_evaluation
        GROUP BY evaluation_method
        ORDER BY count DESC
    ''')
    for method, count, success_rate in cursor.fetchall():
        print(f"   {method}: {count} cases ({success_rate:.1f}% success rate)")

    conn.close()

def main():
    """Main execution function"""
    print("="*80)
    print("🧮 SIMPLIFIED MODEL EVALUATION ENGINE (Database-First)")
    print("="*80)
    print("Goal: Copy inference_results → results_evaluation table with evaluation columns")
    print("Columns added: is_correct, confidence_score, evaluation_method")

    # Check if inference_results table has data
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM inference_results')
    count = cursor.fetchone()[0]
    conn.close()

    if count == 0:
        print("❌ No data found in inference_results table. Run model_inference_2.py first.")
        return

    print(f"✅ Found {count} records in inference_results table")

    # Create results_evaluation table (copy of inference_results with added columns)
    create_results_evaluation_table()

    # Populate with evaluations
    populate_evaluation_results()

    # Generate summary
    generate_evaluation_summary()

    print(f"\n💾 Database: {DB_PATH} → results_evaluation table")
    print("🎯 Use this table for robustness analysis and detailed accuracy metrics")

if __name__ == "__main__":
    main()