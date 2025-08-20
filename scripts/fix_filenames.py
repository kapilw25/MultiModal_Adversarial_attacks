#!/usr/bin/env python3
"""
Script to rename result files to match actual question counts instead of legacy hardcoded counts.
"""

import os
import glob

def get_correct_mapping():
    """Return mapping of incorrect counts to correct counts"""
    return {
        'planar_layout_72': 'planar_layout_51',
        'relation_graph_57': 'relation_graph_56', 
        'table_66': 'table_65'
        # Note: visual_puzzle_18 is actually correct (18 processed after skipping corrupted images)
        # The ground truth has 33, but due to image corruption, only 18 are processed
    }

def rename_files_in_directory(directory):
    """Rename all files in the given directory"""
    mapping = get_correct_mapping()
    renamed_count = 0
    
    print(f"Processing directory: {directory}")
    
    for old_pattern, new_pattern in mapping.items():
        # Find all files matching the old pattern
        pattern = os.path.join(directory, f"*{old_pattern}*")
        files = glob.glob(pattern)
        
        for old_file in files:
            new_file = old_file.replace(old_pattern, new_pattern)
            
            if os.path.exists(old_file):
                print(f"  Renaming: {os.path.basename(old_file)} -> {os.path.basename(new_file)}")
                os.rename(old_file, new_file)
                renamed_count += 1
    
    return renamed_count

def main():
    """Main function to rename files in all model directories"""
    results_dir = "results/models"
    total_renamed = 0
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print("=== Fixing Legacy Filename Conflicts ===")
    print("Renaming files to match actual question counts...\n")
    
    # Process each model directory
    for model_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, model_dir)
        if os.path.isdir(model_path):
            renamed = rename_files_in_directory(model_path)
            total_renamed += renamed
            if renamed > 0:
                print(f"  ✅ Renamed {renamed} files in {model_dir}")
            else:
                print(f"  ✅ No files to rename in {model_dir}")
    
    print(f"\n🎉 Total files renamed: {total_renamed}")
    print("✅ All filename conflicts resolved!")

if __name__ == "__main__":
    main()