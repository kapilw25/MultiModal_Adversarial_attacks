#!/usr/bin/env python3
"""
Data analysis script for generating line plots from the robustness database.
These graphs show degradation percentage for each VLM model across all attacks.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sqlite3
import os

# Database path
DB_PATH = "/home/016649880@SJSUAD/Multi-modal-Self-instruct/results/robustness.db"

# Create output directory if it doesn't exist
output_dir = '/home/016649880@SJSUAD/Multi-modal-Self-instruct/results/data_analysis/plots'
os.makedirs(output_dir, exist_ok=True)

def load_data_from_db():
    """Load data from the SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    
    # Query the attack_comparison table
    query = """
    SELECT 
        attack_type, 
        gpt4o_accuracy, gpt4o_change,
        qwen_accuracy, qwen_change,
        gemma_accuracy, gemma_change
    FROM attack_comparison
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Add attack type classification (Transfer vs. Black-Box)
    transfer_attacks = ['PGD', 'FGSM', 'CW-L2', 'CW-L0', 'CW-L∞', 'L-BFGS', 'JSMA', 'DeepFool']
    df['attack_category'] = df['attack_type'].apply(
        lambda x: 'Transfer' if x in transfer_attacks else 
                  'Baseline' if x == 'Original' else 'Black-Box'
    )
    
    return df

def plot_model_degradation_line(df, output_path):
    """
    Create a line plot showing degradation percentage for each VLM model across all attacks.
    """
    # Filter out the baseline for sorting
    df_no_baseline = df[df['attack_type'] != 'Original'].copy()
    
    # Calculate average change across models for sorting
    df_no_baseline['avg_change'] = (df_no_baseline['gpt4o_change'] + 
                                   df_no_baseline['qwen_change'] + 
                                   df_no_baseline['gemma_change']) / 3
    
    # Sort attacks by average effectiveness (most negative change first)
    attack_order = df_no_baseline.sort_values('avg_change')['attack_type'].tolist()
    
    # Add Original back at the beginning
    attack_order = ['Original'] + attack_order
    
    # Filter the dataframe to include Original and sort by the determined order
    df_plot = df.copy()
    df_plot['attack_order'] = df_plot['attack_type'].apply(lambda x: attack_order.index(x) if x in attack_order else 999)
    df_plot = df_plot.sort_values('attack_order')
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Plot lines for each model
    plt.plot(df_plot['attack_type'], df_plot['gpt4o_change'], 'o-', linewidth=2, markersize=8, label='GPT-4o')
    plt.plot(df_plot['attack_type'], df_plot['qwen_change'], 's-', linewidth=2, markersize=8, label='Qwen-VL-3B')
    plt.plot(df_plot['attack_type'], df_plot['gemma_change'], '^-', linewidth=2, markersize=8, label='Gemma-VL-4B')
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.title('VLM Robustness Against Different Attacks', fontsize=16)
    plt.xlabel('Attack Type', fontsize=14)
    plt.ylabel('Accuracy Change (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add annotations for degradation and improvement regions
    plt.fill_between([-0.5, len(attack_order) - 0.5], -60, 0, color='red', alpha=0.1, label='Degradation')
    plt.fill_between([-0.5, len(attack_order) - 0.5], 0, 20, color='green', alpha=0.1, label='Improvement')
    
    # Adjust y-axis limits to show all data with some padding
    plt.ylim(min(df_plot['qwen_change'].min(), df_plot['gpt4o_change'].min(), df_plot['gemma_change'].min()) - 5,
             max(df_plot['qwen_change'].max(), df_plot['gpt4o_change'].max(), df_plot['gemma_change'].max()) + 5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved model degradation line plot to {output_path}")

def plot_attack_category_line(df, output_path):
    """
    Create a line plot showing degradation by attack category (Transfer vs. Black-Box).
    """
    # Filter out the baseline
    df_attacks = df[df['attack_type'] != 'Original'].copy()
    
    # Group by attack category and calculate average change for each model
    category_data = []
    
    for category in ['Transfer', 'Black-Box']:
        category_df = df_attacks[df_attacks['attack_category'] == category]
        
        category_data.append({
            'Attack Category': category,
            'GPT-4o': category_df['gpt4o_change'].mean(),
            'Qwen-VL-3B': category_df['qwen_change'].mean(),
            'Gemma-VL-4B': category_df['gemma_change'].mean()
        })
    
    category_df = pd.DataFrame(category_data)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot lines for each model
    plt.plot(category_df['Attack Category'], category_df['GPT-4o'], 'o-', linewidth=2, markersize=10, label='GPT-4o')
    plt.plot(category_df['Attack Category'], category_df['Qwen-VL-3B'], 's-', linewidth=2, markersize=10, label='Qwen-VL-3B')
    plt.plot(category_df['Attack Category'], category_df['Gemma-VL-4B'], '^-', linewidth=2, markersize=10, label='Gemma-VL-4B')
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add labels and title
    plt.title('VLM Robustness by Attack Category', fontsize=16)
    plt.xlabel('Attack Category', fontsize=14)
    plt.ylabel('Average Accuracy Change (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Add value labels for each point
    for i, category in enumerate(category_df['Attack Category']):
        for model, marker in [('GPT-4o', 'o'), ('Qwen-VL-3B', 's'), ('Gemma-VL-4B', '^')]:
            value = category_df.iloc[i][model]
            plt.annotate(f'{value:.1f}%', 
                        (category, value),
                        textcoords="offset points",
                        xytext=(0, 10 if value > 0 else -15),
                        ha='center')
    
    # Add annotations for degradation and improvement regions
    plt.fill_between([-0.5, 1.5], -40, 0, color='red', alpha=0.1, label='Degradation')
    plt.fill_between([-0.5, 1.5], 0, 10, color='green', alpha=0.1, label='Improvement')
    
    # Adjust y-axis limits to show all data with some padding
    plt.ylim(min(category_df['GPT-4o'].min(), category_df['Qwen-VL-3B'].min(), category_df['Gemma-VL-4B'].min()) - 5,
             max(category_df['GPT-4o'].max(), category_df['Qwen-VL-3B'].max(), category_df['Gemma-VL-4B'].max()) + 5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved attack category line plot to {output_path}")

def plot_model_baseline_comparison(df, output_path):
    """
    Create a line plot showing baseline accuracy and average accuracy under attack for each model.
    """
    # Get baseline accuracy for each model
    baseline = df[df['attack_type'] == 'Original'].iloc[0]
    
    # Calculate average accuracy under attack for each model
    df_attacks = df[df['attack_type'] != 'Original'].copy()
    
    avg_under_attack = {
        'GPT-4o': df_attacks['gpt4o_accuracy'].mean(),
        'Qwen-VL-3B': df_attacks['qwen_accuracy'].mean(),
        'Gemma-VL-4B': df_attacks['gemma_accuracy'].mean()
    }
    
    # Prepare data for plotting
    models = ['GPT-4o', 'Qwen-VL-3B', 'Gemma-VL-4B']
    baseline_values = [baseline['gpt4o_accuracy'], baseline['qwen_accuracy'], baseline['gemma_accuracy']]
    attack_values = [avg_under_attack['GPT-4o'], avg_under_attack['Qwen-VL-3B'], avg_under_attack['Gemma-VL-4B']]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot lines
    plt.plot(models, baseline_values, 'o-', linewidth=2, markersize=10, label='Baseline Accuracy', color='blue')
    plt.plot(models, attack_values, 's-', linewidth=2, markersize=10, label='Avg. Under Attack', color='red')
    
    # Add labels and title
    plt.title('Baseline vs. Under Attack Accuracy Comparison', fontsize=16)
    plt.xlabel('Vision-Language Model', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Add value labels for each point
    for i, model in enumerate(models):
        # Baseline value
        plt.annotate(f'{baseline_values[i]:.1f}%', 
                    (model, baseline_values[i]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center')
        
        # Attack value
        plt.annotate(f'{attack_values[i]:.1f}%', 
                    (model, attack_values[i]),
                    textcoords="offset points",
                    xytext=(0, -15),
                    ha='center')
    
    # Calculate and display robustness score (percentage of accuracy retained under attack)
    for i, model in enumerate(models):
        robustness = (attack_values[i] / baseline_values[i]) * 100
        plt.annotate(f'Retains {robustness:.1f}% of accuracy', 
                    (model, (baseline_values[i] + attack_values[i]) / 2),
                    textcoords="offset points",
                    xytext=(0, -30),
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved model baseline comparison to {output_path}")

def main():
    """Main function to generate all plots"""
    print("Loading data from database...")
    df = load_data_from_db()
    
    print("Generating line plots...")
    
    # Plot 1: Model degradation line plot
    plot_model_degradation_line(df, os.path.join(output_dir, 'model_degradation_line.png'))
    
    # Plot 2: Attack category line plot
    plot_attack_category_line(df, os.path.join(output_dir, 'attack_category_line.png'))
    
    # Plot 3: Model baseline comparison
    plot_model_baseline_comparison(df, os.path.join(output_dir, 'model_baseline_comparison.png'))
    
    print("All line plots have been generated successfully!")

if __name__ == "__main__":
    main()
