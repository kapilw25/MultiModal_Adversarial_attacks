#!/usr/bin/env python3
"""
Generate detailed completion table for all VLM models across all tasks
"""

import os
import json
from collections import defaultdict

# Expected question counts per task
EXPECTED_COUNTS = {
    'chart': 81, # 3 images × 27 questions each
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
        return -1

def analyze_model_completion():
    """Analyze completion status for all models and create detailed table"""
    models_dir = 'results/models'
    if not os.path.exists(models_dir):
        print("❌ results/models directory not found!")
        return
    
    # All 18 models from the inference script in the order they appear
    all_models = [
        'Qwen25_VL_3B',
        'Qwen25_VL_7B', 
        'Qwen2_VL_2B',
        'Gemma3_VL_4B',
        'PaliGemma_VL_3B',
        'DeepSeek1_VL_1pt3B',
        'DeepSeek1_VL_7B',
        'SmolVLM2_pt25B',
        'SmolVLM2_pt5B',
        'SmolVLM2_2pt2B',
        'Phi3pt5_vision_4B',
        'Florence2_pt23B',
        'Florence2_pt77B',
        'Moondream2_2B',
        'GLMEdge_2B',
        'InternVL3_1B',
        'InternVL3_2B',
        'InternVL25_4B'
    ]
    
    # Collect data for table
    table_data = {}
    
    for model_name in all_models:
        model_path = os.path.join(models_dir, model_name)
        table_data[model_name] = {}
        
        # Check if model directory exists
        model_exists = os.path.exists(model_path)
        
        for task in EXPECTED_COUNTS:
            expected_count = EXPECTED_COUNTS[task]
            files_found = 0
            files_valid = 0
            
            if model_exists:
                for attack in ALL_ATTACKS:
                    # Build expected filename
                    if attack == '':
                        filename = f"eval_{model_name}_{task}_{expected_count}.json"
                    else:
                        filename = f"eval_{model_name}_{task}_{expected_count}{attack}.json"
                    
                    filepath = os.path.join(model_path, filename)
                    
                    if os.path.exists(filepath):
                        line_count = count_lines_in_file(filepath)
                        files_found += 1
                        if line_count == expected_count:
                            files_valid += 1
            
            # Store completion stats for this task
            completion_pct = (files_valid / len(ALL_ATTACKS)) * 100
            table_data[model_name][task] = {
                'valid': files_valid,
                'total': len(ALL_ATTACKS),
                'percentage': completion_pct,
                'exists': model_exists
            }
    
    return table_data, all_models

def print_detailed_table():
    """Print detailed completion table"""
    table_data, models = analyze_model_completion()
    
    print("🔍 VLM MODEL COMPLETION STATISTICS TABLE")
    print("=" * 120)
    print()
    
    # Header
    header = f"{'Model Engine':<20}"
    for task in EXPECTED_COUNTS:
        header += f"{task:<15}"
    header += f"{'Overall':<15}"
    print(header)
    print("-" * 135)
    
    # Data rows
    overall_stats = {}
    for model in models:
        # Check if model has any results
        model_exists = any(table_data[model][task]['exists'] for task in EXPECTED_COUNTS)
        
        if model_exists:
            row = f"{model:<20}"
        else:
            row = f"{model:<20}"
        
        model_total_valid = 0
        model_total_possible = 0
        
        for task in EXPECTED_COUNTS:
            stats = table_data[model][task]
            valid = stats['valid']
            total = stats['total']
            pct = stats['percentage']
            exists = stats['exists']
            
            if not exists:
                # Model directory doesn't exist yet
                cell = "⏳ Pending"
            elif valid == 0 and total > 0:
                # Model directory exists but no files yet
                cell = f"0/{total} (0%)"
            else:
                # Normal case
                cell = f"{valid}/{total} ({pct:.0f}%)"
            
            row += f"{cell:<15}"
            
            model_total_valid += valid
            model_total_possible += total
        
        # Overall column
        if not model_exists:
            overall_cell = "⏳ Not Started"
        else:
            overall_pct = (model_total_valid / model_total_possible) * 100 if model_total_possible > 0 else 0
            overall_cell = f"{model_total_valid}/{model_total_possible} ({overall_pct:.1f}%)"
        
        row += f"{overall_cell:<15}"
        
        print(row)
        
        # Store for summary
        overall_stats[model] = {
            'valid': model_total_valid,
            'total': model_total_possible,
            'percentage': overall_pct if model_exists and model_total_possible > 0 else 0,
            'exists': model_exists
        }
    
    print("-" * 135)
    
    # Summary statistics
    print()
    print("📊 SUMMARY STATISTICS:")
    print()
    
    # Task-wise completion
    print("📋 Task Completion Rates:")
    for task in EXPECTED_COUNTS:
        task_valid = sum(table_data[model][task]['valid'] for model in models)
        task_total = len(models) * len(ALL_ATTACKS)
        task_pct = (task_valid / task_total) * 100
        print(f"   {task:<15}: {task_valid:>3}/{task_total:<3} ({task_pct:>5.1f}%)")
    
    print()
    
    # Model ranking
    print("🏆 Model Completion Ranking:")
    sorted_models = sorted(models, key=lambda m: overall_stats[m]['percentage'], reverse=True)
    for i, model in enumerate(sorted_models, 1):
        stats = overall_stats[model]
        if not stats['exists']:
            status = "⏳ Not Started"
            display_stats = "  -/144 (  0.0%)"
        elif stats['percentage'] < 1:
            status = "🔄 Processing"
            display_stats = f"{stats['valid']:>3}/{stats['total']:<3} ({stats['percentage']:>5.1f}%)"
        elif stats['percentage'] < 95:
            status = "🔄 Processing"
            display_stats = f"{stats['valid']:>3}/{stats['total']:<3} ({stats['percentage']:>5.1f}%)"
        else:
            status = "✅ Complete"
            display_stats = f"{stats['valid']:>3}/{stats['total']:<3} ({stats['percentage']:>5.1f}%)"
        print(f"   {i:>2}. {model:<20}: {display_stats} {status}")
    
    print()
    
    # Overall totals
    total_valid = sum(overall_stats[model]['valid'] for model in models)
    total_possible = len(models) * len(EXPECTED_COUNTS) * len(ALL_ATTACKS)  # 18 models × 8 tasks × 18 attacks = 2592
    total_pct = (total_valid / total_possible) * 100
    
    models_started = sum(1 for model in models if overall_stats[model]['exists'])
    models_pending = len(models) - models_started
    
    print(f"🎯 GRAND TOTAL: {total_valid}/{total_possible} ({total_pct:.1f}%) files completed")
    print(f"📊 PROGRESS: {models_started}/18 models started, {models_pending} models pending")
    
    print()
    print("=" * 135)

if __name__ == "__main__":
    print_detailed_table()