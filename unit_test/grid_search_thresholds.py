#!/usr/bin/env python
"""
Grid search script to find optimal perceptual constraint thresholds for Square Attack
that achieve a target performance degradation.
"""

import os
import sys
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd
from datetime import datetime
import tempfile
import re

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
IMAGE_PATH = "data/test_extracted/chart/20231114102825506748.png"
RESULTS_DIR = "results/threshold_search"
PLOTS_DIR = "results/threshold_search/plots"
TARGET_DEGRADATION = 20.0  # Target performance degradation percentage
ENGINE = "Qwen25_VL_3B"
TASK = "chart"

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Grid search parameters - using a smaller set for testing
ssim_thresholds = [0.85, 0.75]  # More relaxed SSIM thresholds
lpips_thresholds = [0.10, 0.20]  # Higher LPIPS thresholds (more perceptual difference allowed)
clip_thresholds = [0.80, 0.70]   # Lower CLIP thresholds (less semantic similarity required)

# Fixed parameters
eps = 0.15
norm = "inf"
max_iter = 200
p_init = 0.3

def run_square_attack(ssim_threshold, lpips_threshold, clip_threshold):
    """Run the Square Attack with the given thresholds."""
    # Create a timestamp for unique identification
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Run the Square Attack
    cmd = [
        "python", "attack_models/true_black_box_attacks/v10_square_attack.py",
        "--image_path", IMAGE_PATH,
        "--eps", str(eps),
        "--norm", norm,
        "--max_iter", str(max_iter),
        "--p_init", str(p_init),
        "--ssim_threshold", str(ssim_threshold),
        "--lpips_threshold", str(lpips_threshold),
        "--clip_threshold", str(clip_threshold),
        "--threshold_optimized"
    ]
    
    print(f"Running Square Attack with SSIM={ssim_threshold}, LPIPS={lpips_threshold}, CLIP={clip_threshold}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running Square Attack: {result.stderr}")
        return None
    
    # Extract the path where the adversarial image was saved
    output_path = None
    for line in result.stdout.split('\n'):
        if "Saved adversarial image to" in line:
            output_path = line.split("Saved adversarial image to")[1].strip()
            break
    
    if not output_path:
        print("Could not find output path in Square Attack output")
        return None
    
    return output_path

def run_eval_model(adv_image_path):
    """
    Run eval_model.py with automated inputs to evaluate the adversarial image.
    
    Args:
        adv_image_path: Path to the adversarial image
        
    Returns:
        Path to the evaluation results JSON file
    """
    # Create a temporary script to run the evaluation
    temp_script = f"""
import os
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from the original eval_model.py
from scripts.local_llm_tools import create_model, generate_response

def load_data(task, count=None):
    \"\"\"Load data from the specified task.\"\"\"
    data_file = f"data/test_extracted/{{task}}/data.json"
    with open(data_file, "r") as f:
        data_list = json.load(f)
    
    if count is not None:
        data_list = data_list[:count]
    
    return data_list

def process_response(response):
    \"\"\"Process the response to extract the answer.\"\"\"
    return response.strip()

# Constants
ENGINE = "{ENGINE}"
TASK = "{TASK}"
SAMPLE_COUNT = 1  # Only process one image for efficiency
ADV_IMAGE_PATH = "{adv_image_path}"

# Load data
data_list = load_data(TASK, SAMPLE_COUNT)

# Create model
model, processor = create_model(ENGINE)

# Generate responses
results = []
for data in data_list:
    # Use the adversarial image
    img_path = ADV_IMAGE_PATH
    
    # Generate response
    response = generate_response(model, processor, ENGINE, img_path, data["question"])
    
    # Process response
    processed_response = process_response(response)
    
    # Add to results
    result = {{
        "id": data["id"],
        "image": data["image"],
        "question": data["question"],
        "answer": data["answer"],
        "response": processed_response
    }}
    results.append(result)

# Save results
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
output_file = f"results/{{ENGINE}}/eval_{{ENGINE}}_{{TASK}}_search_{{timestamp}}.json"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Results saved to {{output_file}}")
print(output_file)  # Print the output file path for the parent script to capture
"""
    
    # Write the temporary evaluation script
    temp_script_path = os.path.join(RESULTS_DIR, "temp_eval_model.py")
    with open(temp_script_path, "w") as f:
        f.write(temp_script)
    
    # Run the evaluation script
    print("Running model evaluation...")
    result = subprocess.run(
        ["python", temp_script_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error running evaluation: {result.stderr}")
        return None
    
    # Extract the output file path from the script's output
    output_lines = result.stdout.strip().split('\n')
    eval_output_file = output_lines[-1]
    
    return eval_output_file

def calculate_accuracy(eval_file):
    """
    Calculate accuracy using a simplified version of eval_vqa.py logic.
    
    Args:
        eval_file: Path to the evaluation results JSON file
        
    Returns:
        Tuple of (adversarial_accuracy, original_accuracy, degradation)
    """
    # Get the original evaluation file
    original_eval_file = f"results/{ENGINE}/eval_{ENGINE}_{TASK}_17.json"
    
    if not os.path.exists(original_eval_file):
        print(f"Error: Original evaluation file {original_eval_file} not found.")
        print("Please run 'python scripts/eval_model.py' with the Original (No Attack) option first.")
        sys.exit(1)
    
    # Load the evaluation results
    with open(eval_file, "r") as f:
        adv_results = json.load(f)
    
    with open(original_eval_file, "r") as f:
        original_results = json.load(f)
    
    # Find the corresponding original result
    adv_id = adv_results[0]["id"]
    original_result = next((r for r in original_results if r["id"] == adv_id), None)
    
    if not original_result:
        print(f"Error: Could not find original result for ID {adv_id}")
        return 0, 0, 0
    
    # Calculate accuracy for adversarial image
    adv_correct = 0
    for result in adv_results:
        answer = result["answer"].lower()
        response = result["response"].lower()
        if answer in response:
            adv_correct += 1
    
    adv_accuracy = (adv_correct / len(adv_results)) * 100
    
    # Calculate accuracy for original image
    orig_correct = 0
    answer = original_result["answer"].lower()
    response = original_result["response"].lower()
    if answer in response:
        orig_correct += 1
    
    orig_accuracy = (orig_correct / 1) * 100
    
    # Calculate degradation
    degradation = orig_accuracy - adv_accuracy
    
    return adv_accuracy, orig_accuracy, degradation

def main():
    """Main function to run the grid search."""
    results = []
    
    # Check if we have results from previous runs
    results_file = os.path.join(RESULTS_DIR, "grid_search_results.json")
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} previous results from {results_file}")
    
    # Get combinations that have already been tested
    tested_combinations = {(r["ssim_threshold"], r["lpips_threshold"], r["clip_threshold"]) for r in results}
    
    # Run grid search
    for ssim, lpips, clip in product(ssim_thresholds, lpips_thresholds, clip_thresholds):
        # Skip if this combination has already been tested
        if (ssim, lpips, clip) in tested_combinations:
            print(f"Skipping already tested combination: SSIM={ssim}, LPIPS={lpips}, CLIP={clip}")
            continue
        
        try:
            # Run Square Attack
            adv_image_path = run_square_attack(ssim, lpips, clip)
            if not adv_image_path:
                print(f"Skipping evaluation for SSIM={ssim}, LPIPS={lpips}, CLIP={clip} due to attack failure")
                continue
            
            # Run evaluation
            eval_file = run_eval_model(adv_image_path)
            if not eval_file:
                print(f"Skipping accuracy calculation for SSIM={ssim}, LPIPS={lpips}, CLIP={clip} due to evaluation failure")
                continue
            
            # Calculate accuracy and degradation
            accuracy, original_accuracy, degradation = calculate_accuracy(eval_file)
            
            # Store results
            result = {
                "ssim_threshold": ssim,
                "lpips_threshold": lpips,
                "clip_threshold": clip,
                "accuracy": accuracy,
                "original_accuracy": original_accuracy,
                "degradation": degradation,
                "eval_file": eval_file,
                "adv_image_path": adv_image_path
            }
            results.append(result)
            
            # Save results after each iteration
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"Results: Accuracy={accuracy:.2f}%, Original={original_accuracy:.2f}%, Degradation={degradation:.2f}%")
            
            # Check if we've reached the target degradation
            if abs(degradation - TARGET_DEGRADATION) < 5.0:  # Within 5% of target
                print(f"Found combination close to target degradation of {TARGET_DEGRADATION}%!")
                print(f"SSIM={ssim}, LPIPS={lpips}, CLIP={clip}, Degradation={degradation:.2f}%")
                break
                
        except Exception as e:
            print(f"Error with combination SSIM={ssim}, LPIPS={lpips}, CLIP={clip}: {e}")
    
    # Create a DataFrame for easier analysis
    if results:
        df = pd.DataFrame(results)
        
        # Plot results
        plot_results(df)
        
        # Find the combination closest to the target degradation
        df['distance_to_target'] = abs(df['degradation'] - TARGET_DEGRADATION)
        best_row = df.loc[df['distance_to_target'].idxmin()]
        
        print("\nBest combination found:")
        print(f"SSIM threshold: {best_row['ssim_threshold']}")
        print(f"LPIPS threshold: {best_row['lpips_threshold']}")
        print(f"CLIP threshold: {best_row['clip_threshold']}")
        print(f"Resulting degradation: {best_row['degradation']:.2f}%")
        print(f"(Target was {TARGET_DEGRADATION}%)")
        
        # Run with the best parameters on all images
        print("\nTo run the attack with these parameters on all images, use:")
        print(f"python attack_models/true_black_box_attacks/v10_square_attack.py --image_path data/test_extracted/chart/ --eps {eps} --norm {norm} --max_iter {max_iter} --p_init {p_init} --ssim_threshold {best_row['ssim_threshold']} --lpips_threshold {best_row['lpips_threshold']} --clip_threshold {best_row['clip_threshold']} --threshold_optimized")
    else:
        print("No results were collected. Please check the error messages above.")

def plot_results(df):
    """Plot the results of the grid search."""
    if df.empty:
        print("No results to plot")
        return
    
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        df['ssim_threshold'],
        df['lpips_threshold'],
        df['clip_threshold'],
        c=df['degradation'],
        cmap='viridis',
        s=100,
        alpha=0.7
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Performance Degradation (%)')
    
    # Set labels
    ax.set_xlabel('SSIM Threshold')
    ax.set_ylabel('LPIPS Threshold')
    ax.set_zlabel('CLIP Threshold')
    
    # Set title
    plt.title('Effect of Perceptual Constraints on Model Performance Degradation')
    
    # Save the plot
    plt.savefig(os.path.join(PLOTS_DIR, '3d_scatter_degradation.png'))
    
    # Create 2D plots for each pair of parameters
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # SSIM vs LPIPS
    for clip in df['clip_threshold'].unique():
        subset = df[df['clip_threshold'] == clip]
        axes[0].scatter(
            subset['ssim_threshold'],
            subset['lpips_threshold'],
            c=subset['degradation'],
            cmap='viridis',
            s=100,
            alpha=0.7,
            label=f'CLIP={clip}'
        )
    
    axes[0].set_xlabel('SSIM Threshold')
    axes[0].set_ylabel('LPIPS Threshold')
    axes[0].set_title('SSIM vs LPIPS (color = degradation)')
    axes[0].legend()
    
    # SSIM vs CLIP
    for lpips in df['lpips_threshold'].unique():
        subset = df[df['lpips_threshold'] == lpips]
        axes[1].scatter(
            subset['ssim_threshold'],
            subset['clip_threshold'],
            c=subset['degradation'],
            cmap='viridis',
            s=100,
            alpha=0.7,
            label=f'LPIPS={lpips}'
        )
    
    axes[1].set_xlabel('SSIM Threshold')
    axes[1].set_ylabel('CLIP Threshold')
    axes[1].set_title('SSIM vs CLIP (color = degradation)')
    axes[1].legend()
    
    # LPIPS vs CLIP
    for ssim in df['ssim_threshold'].unique():
        subset = df[df['ssim_threshold'] == ssim]
        axes[2].scatter(
            subset['lpips_threshold'],
            subset['clip_threshold'],
            c=subset['degradation'],
            cmap='viridis',
            s=100,
            alpha=0.7,
            label=f'SSIM={ssim}'
        )
    
    axes[2].set_xlabel('LPIPS Threshold')
    axes[2].set_ylabel('CLIP Threshold')
    axes[2].set_title('LPIPS vs CLIP (color = degradation)')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '2d_scatter_degradation.png'))
    
    # Create elbow chart
    plt.figure(figsize=(10, 6))
    
    # Sort by degradation
    df_sorted = df.sort_values('degradation')
    
    # Create a combined parameter string for x-axis
    df_sorted['param_combo'] = df_sorted.apply(
        lambda row: f"S:{row['ssim_threshold']:.2f}\nL:{row['lpips_threshold']:.2f}\nC:{row['clip_threshold']:.2f}",
        axis=1
    )
    
    plt.plot(df_sorted['param_combo'], df_sorted['degradation'], 'o-', linewidth=2, markersize=8)
    plt.axhline(y=TARGET_DEGRADATION, color='r', linestyle='--', label=f'Target ({TARGET_DEGRADATION}%)')
    
    plt.xlabel('Parameter Combinations (SSIM/LPIPS/CLIP)')
    plt.ylabel('Performance Degradation (%)')
    plt.title('Elbow Chart: Parameter Combinations vs Performance Degradation')
    plt.xticks(rotation=90)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(PLOTS_DIR, 'elbow_chart_degradation.png'))
    
    print(f"Plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    main()
