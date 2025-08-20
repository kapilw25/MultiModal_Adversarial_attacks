#!/usr/bin/env python3
"""
Comprehensive analysis of model inference completion status
Checks all models, tasks, and attacks for file validity and completion
"""

import os
import json
import glob
from collections import defaultdict

# Expected question counts per task (from ground truth files)
EXPECTED_COUNTS = {
    'chart': 81,
    'table': 65, 
    'road_map': 3,
    'dashboard': 60,
    'flowchart': 60,
    'relation_graph': 56,
    'planar_layout': 51,
    'visual_puzzle': 18
}

# All possible attacks (including original)
ALL_ATTACKS = [
    '',  # Original (no attack suffix)
    '_BB_boundary',
    '_BB_cw_l0', 
    '_BB_cw_l2',
    '_BB_cw_linf',
    '_BB_deepfool',
    '_BB_fgsm',
    '_BB_geoda',
    '_BB_hop_skip_jump',
    '_BB_jsma',
    '_BB_lbfgs',
    '_BB_pgd',
    '_BB_pixel',
    '_BB_query_efficient_bb',
    '_BB_simba',
    '_BB_spatial',
    '_BB_square',
    '_BB_zoo'
]

def count_lines_in_file(filepath):
    """Count non-empty lines in a JSON file"""
    try:
        with open(filepath, 'r') as f:
            return sum(1 for line in f if line.strip())
    except Exception as e:
        return -1  # Error reading file

def analyze_model_directory(model_dir):
    """Analyze a single model directory for completion status"""
    model_name = os.path.basename(model_dir)
    results = {
        'model': model_name,
        'tasks': {},
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0,
        'missing_files': 0
    }
    
    for task in EXPECTED_COUNTS:
        expected_count = EXPECTED_COUNTS[task]
        task_results = {
            'expected_count': expected_count,
            'attacks': {},
            'files_found': 0,
            'files_valid': 0,
            'files_invalid': 0,
            'files_missing': 0
        }
        
        for attack in ALL_ATTACKS:
            # Build expected filename
            if attack == '':
                filename = f"eval_{model_name}_{task}_{expected_count}.json"
            else:
                filename = f"eval_{model_name}_{task}_{expected_count}{attack}.json"
            
            filepath = os.path.join(model_dir, filename)
            
            if os.path.exists(filepath):
                line_count = count_lines_in_file(filepath)
                task_results['files_found'] += 1
                results['total_files'] += 1
                
                if line_count == expected_count:
                    status = 'VALID'
                    task_results['files_valid'] += 1
                    results['valid_files'] += 1
                elif line_count == -1:
                    status = 'ERROR'
                    task_results['files_invalid'] += 1
                    results['invalid_files'] += 1
                else:
                    status = f'INVALID({line_count}!={expected_count})'
                    task_results['files_invalid'] += 1
                    results['invalid_files'] += 1
                
                task_results['attacks'][attack if attack else 'original'] = {
                    'filename': filename,
                    'status': status,
                    'line_count': line_count
                }
            else:
                task_results['files_missing'] += 1
                results['missing_files'] += 1
                task_results['attacks'][attack if attack else 'original'] = {
                    'filename': filename,
                    'status': 'MISSING',
                    'line_count': 0
                }
        
        results['tasks'][task] = task_results
    
    return results

def print_summary_report():
    """Print comprehensive summary report"""
    models_dir = 'results/models'
    if not os.path.exists(models_dir):
        print("❌ results/models directory not found!")
        return
    
    model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    model_dirs.sort()
    
    all_results = []
    total_expected = len(model_dirs) * len(EXPECTED_COUNTS) * len(ALL_ATTACKS)
    total_found = 0
    total_valid = 0
    total_invalid = 0
    total_missing = 0
    
    print("🔍 COMPREHENSIVE MODEL INFERENCE ANALYSIS")
    print("=" * 80)
    print(f"📊 Expected total files: {total_expected}")
    print(f"📁 Models to analyze: {len(model_dirs)}")
    print(f"📋 Tasks per model: {len(EXPECTED_COUNTS)}")  
    print(f"⚔️  Attacks per task: {len(ALL_ATTACKS)}")
    print()
    
    for model_dir_name in model_dirs:
        model_path = os.path.join(models_dir, model_dir_name)
        result = analyze_model_directory(model_path)
        all_results.append(result)
        
        total_found += result['total_files']
        total_valid += result['valid_files']
        total_invalid += result['invalid_files']
        total_missing += result['missing_files']
        
        # Model summary
        completion_pct = (result['valid_files'] / (len(EXPECTED_COUNTS) * len(ALL_ATTACKS))) * 100
        print(f"🤖 {result['model']}:")
        print(f"   ✅ Valid: {result['valid_files']}")
        print(f"   ❌ Invalid: {result['invalid_files']}")
        print(f"   ⏳ Missing: {result['missing_files']}")
        print(f"   📈 Completion: {completion_pct:.1f}%")
        
        # Show problematic tasks
        for task, task_data in result['tasks'].items():
            if task_data['files_invalid'] > 0 or task_data['files_missing'] > 0:
                print(f"      📋 {task}: {task_data['files_valid']}/{len(ALL_ATTACKS)} valid")
        print()
    
    print("=" * 80)
    print("📊 OVERALL SUMMARY:")
    print(f"   📁 Total files found: {total_found}")
    print(f"   ✅ Valid files: {total_valid}")
    print(f"   ❌ Invalid files: {total_invalid}")
    print(f"   ⏳ Missing files: {total_missing}")
    print(f"   📈 Overall completion: {(total_valid/total_expected)*100:.1f}%")
    print()
    
    # Detailed breakdown by task
    print("📋 TASK BREAKDOWN:")
    for task in EXPECTED_COUNTS:
        task_valid = sum(r['tasks'][task]['files_valid'] for r in all_results)
        task_total = len(model_dirs) * len(ALL_ATTACKS)
        task_pct = (task_valid / task_total) * 100
        print(f"   {task}: {task_valid}/{task_total} ({task_pct:.1f}%)")
    
    # Show incomplete models
    print("\n⏳ INCOMPLETE MODELS:")
    for result in all_results:
        total_possible = len(EXPECTED_COUNTS) * len(ALL_ATTACKS)
        if result['valid_files'] < total_possible:
            missing = total_possible - result['valid_files']
            print(f"   {result['model']}: {missing} files missing/invalid")

if __name__ == "__main__":
    print_summary_report()