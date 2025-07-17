#!/usr/bin/env python3
"""
Data analysis script for generating plots from the robustness database.
This script reads data from the SQLite database and generates various plots
to visualize model performance under different attack types.
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator
import sys

# Add the parent directory to sys.path to import local_llm_tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.local_llm_tools import MODEL_MAPPING

# Define the database path
DB_PATH = "results/robustness.db"
# Define the output directory for plots
PLOT_DIR = "results/data_analysis/plots1"

# Ensure the plot directory exists
os.makedirs(PLOT_DIR, exist_ok=True)

# Define a mapping from normalized model names to display names
# This will be used for plot labels
DISPLAY_NAMES = {
    "gpt4o": "GPT-4o",
    "qwen25_vl_3b": "Qwen2.5-VL-3B",
    "qwen25_vl_7b": "Qwen2.5-VL-7B",
    "qwen2_vl_2b": "Qwen2-VL-2B",
    "gemma3_vl_4b": "Gemma-3-4B",
    "paligemma_vl_3b": "PaliGemma-3B",
    "deepseek1_vl_1pt3b": "DeepSeek-VL-1.3B",
    "deepseek1_vl_7b": "DeepSeek-VL-7B",
    "smolvlm2_pt25b": "SmolVLM2-256M",
    "smolvlm2_pt5b": "SmolVLM2-500M",
    "smolvlm2_2pt2b": "SmolVLM2-2.2B",
    "phi3pt5_vision_4b": "Phi-3.5-Vision-4B",
    "florence2_pt23b": "Florence-2-Base",
    "florence2_pt77b": "Florence-2-Large",
    "moondream2_2b": "Moondream2-2B",
    "glmedge_2b": "GLM-Edge-V-2B",
    "internvl3_1b": "InternVL3-1B",
    "internvl3_2b": "InternVL3-2B",
    "internvl25_4b": "InternVL2.5-4B"
}

# Define a color palette for the models
MODEL_COLORS = {
    "gpt4o": "#0077B6",               # Blue
    "qwen25_vl_3b": "#FF7F0E",        # Orange
    "qwen25_vl_7b": "#FF9E4A",        # Light Orange
    "qwen2_vl_2b": "#FFB347",         # Pale Orange
    "gemma3_vl_4b": "#2CA02C",        # Green
    "paligemma_vl_3b": "#98DF8A",     # Light Green
    "deepseek1_vl_1pt3b": "#D62728",  # Red
    "deepseek1_vl_7b": "#FF9999",     # Light Red
    "smolvlm2_pt25b": "#9467BD",      # Purple
    "smolvlm2_pt5b": "#C5B0D5",       # Light Purple
    "smolvlm2_2pt2b": "#8C564B",      # Brown
    "phi3pt5_vision_4b": "#E377C2",   # Pink
    "florence2_pt23b": "#7F7F7F",     # Gray
    "florence2_pt77b": "#BCBD22",     # Olive
    "moondream2_2b": "#17BECF",       # Cyan
    "glmedge_2b": "#FFBB78",          # Light Orange
    "internvl3_1b": "#AEC7E8",        # Light Blue
    "internvl3_2b": "#FFBB78",        # Light Orange
    "internvl25_4b": "#98DF8A"        # Light Green
}

def normalize_model_name(model_name):
    """
    Normalize model name to match database column names.
    
    Args:
        model_name (str): Original model name
        
    Returns:
        str: Normalized model name suitable for SQL column
    """
    import re
    
    # Convert to lowercase
    name = model_name.lower()
    
    # Replace special characters with underscores
    name = re.sub(r'[^a-z0-9]', '_', name)
    
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    return name

def load_data_from_db():
    """
    Load data from the SQLite database.
    
    Returns:
        pandas.DataFrame: The loaded data
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM attack_comparison", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error loading data from database: {e}")
        return None

def plot_model_degradation_line(df):
    """
    Create a line plot showing model degradation across different attack types.
    
    Args:
        df (pandas.DataFrame): The loaded data
    """
    # Get all change columns
    change_cols = [col for col in df.columns if col.endswith('_change')]
    
    # Select a subset of models to avoid overcrowding the plot
    # Prioritize models mentioned in the README
    selected_models = ['gpt4o_change', 'qwen25_vl_3b_change', 'gemma3_vl_4b_change']
    
    # Filter to only include selected models that exist in the data
    selected_models = [col for col in selected_models if col in change_cols]
    
    # If we don't have enough models, add more from what's available
    if len(selected_models) < 3:
        additional_models = [col for col in change_cols if col not in selected_models]
        selected_models.extend(additional_models[:3 - len(selected_models)])
    
    # Create a new dataframe with just the attack type and selected model changes
    plot_df = df[['attack_type'] + selected_models].copy()
    
    # Sort by attack type to ensure consistent ordering
    plot_df = plot_df.sort_values('attack_type')
    
    # Set up the plot
    plt.figure(figsize=(14, 8))
    
    # Plot each model
    for col in selected_models:
        model_name = col.replace('_change', '')
        display_name = DISPLAY_NAMES.get(model_name, model_name)
        color = MODEL_COLORS.get(model_name, 'gray')
        
        plt.plot(plot_df['attack_type'], plot_df[col], 
                 marker='o', linewidth=2, markersize=8, 
                 label=display_name, color=color)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Customize the plot
    plt.title('Model Accuracy Change Under Different Attacks', fontsize=16)
    plt.xlabel('Attack Type', fontsize=14)
    plt.ylabel('Accuracy Change (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(PLOT_DIR, 'model_degradation_line.png'), dpi=300)
    plt.close()
    
    print(f"Created model degradation line plot: {os.path.join(PLOT_DIR, 'model_degradation_line.png')}")

def plot_attack_effectiveness_heatmap(df):
    """
    Create a heatmap showing the effectiveness of different attacks across models.
    
    Args:
        df (pandas.DataFrame): The loaded data
    """
    # Get all change columns
    change_cols = [col for col in df.columns if col.endswith('_change')]
    
    # Create a pivot table for the heatmap
    # Rows are attack types, columns are models
    heatmap_data = df.pivot_table(
        index='attack_type',
        values=[col for col in change_cols],
        aggfunc='mean'
    )
    
    # Rename columns to display names
    heatmap_data.columns = [DISPLAY_NAMES.get(col.replace('_change', ''), col.replace('_change', '')) 
                           for col in heatmap_data.columns]
    
    # Set up the plot
    plt.figure(figsize=(14, 10))
    
    # Create the heatmap
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt=".1f", 
                cmap="RdYlGn_r",  # Red for negative (bad), green for positive (good)
                center=0,
                linewidths=.5,
                cbar_kws={'label': 'Accuracy Change (%)'}
               )
    
    # Customize the plot
    plt.title('Attack Effectiveness Across Models', fontsize=16)
    plt.ylabel('Attack Type', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(PLOT_DIR, 'attack_effectiveness_heatmap.png'), dpi=300)
    plt.close()
    
    print(f"Created attack effectiveness heatmap: {os.path.join(PLOT_DIR, 'attack_effectiveness_heatmap.png')}")

def plot_model_robustness_radar(df):
    """
    Create a radar chart showing model robustness across different attack types.
    
    Args:
        df (pandas.DataFrame): The loaded data
    """
    # Get all change columns
    change_cols = [col for col in df.columns if col.endswith('_change')]
    
    # Select a subset of models to avoid overcrowding the plot
    selected_models = ['gpt4o_change', 'qwen25_vl_3b_change', 'gemma3_vl_4b_change']
    
    # Filter to only include selected models that exist in the data
    selected_models = [col for col in selected_models if col in change_cols]
    
    # If we don't have enough models, add more from what's available
    if len(selected_models) < 3:
        additional_models = [col for col in change_cols if col not in selected_models]
        selected_models.extend(additional_models[:3 - len(selected_models)])
    
    # Select a subset of attack types for readability
    # Prioritize the most effective attacks
    attack_effectiveness = df.groupby('attack_type')[selected_models].mean().min(axis=1).sort_values()
    selected_attacks = attack_effectiveness.index[:8].tolist()
    
    # Filter data to include only selected attacks
    radar_df = df[df['attack_type'].isin(selected_attacks)].copy()
    
    # Set up the plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Number of attack types
    N = len(selected_attacks)
    
    # Angles for each attack type (in radians)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # Close the loop
    angles += angles[:1]
    
    # Plot each model
    for col in selected_models:
        model_name = col.replace('_change', '')
        display_name = DISPLAY_NAMES.get(model_name, model_name)
        color = MODEL_COLORS.get(model_name, 'gray')
        
        # Get values for this model
        values = radar_df.set_index('attack_type')[col].reindex(selected_attacks).tolist()
        
        # Close the loop
        values += values[:1]
        
        # Plot the model
        ax.plot(angles, values, linewidth=2, label=display_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(selected_attacks, fontsize=10)
    
    # Add a title
    plt.title('Model Robustness Across Attack Types', fontsize=16, y=1.1)
    
    # Add a legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'model_robustness_radar.png'), dpi=300)
    plt.close()
    
    print(f"Created model robustness radar chart: {os.path.join(PLOT_DIR, 'model_robustness_radar.png')}")

def plot_model_size_vs_robustness(df):
    """
    Create a scatter plot showing the relationship between model size and robustness.
    
    Args:
        df (pandas.DataFrame): The loaded data
    """
    # Define model sizes in billions of parameters
    model_sizes = {
        "gpt4o": 25.0,  # Estimated
        "qwen25_vl_3b": 3.0,
        "qwen25_vl_7b": 7.0,
        "qwen2_vl_2b": 2.0,
        "gemma3_vl_4b": 4.0,
        "paligemma_vl_3b": 3.0,
        "deepseek1_vl_1pt3b": 1.3,
        "deepseek1_vl_7b": 7.0,
        "smolvlm2_pt25b": 0.256,
        "smolvlm2_pt5b": 0.5,
        "smolvlm2_2pt2b": 2.2,
        "phi3pt5_vision_4b": 4.15,
        "florence2_pt23b": 0.23,
        "florence2_pt77b": 0.77,
        "moondream2_2b": 1.93,
        "glmedge_2b": 2.0,
        "internvl3_1b": 1.0,
        "internvl3_2b": 2.0,
        "internvl25_4b": 4.0
    }
    
    # Get all change columns
    change_cols = [col for col in df.columns if col.endswith('_change')]
    
    # Calculate average change for each model
    model_avg_changes = {}
    for col in change_cols:
        model_name = col.replace('_change', '')
        model_avg_changes[model_name] = df[col].mean()
    
    # Create lists for plotting
    models = []
    sizes = []
    changes = []
    colors = []
    
    for model, avg_change in model_avg_changes.items():
        if model in model_sizes:
            models.append(DISPLAY_NAMES.get(model, model))
            sizes.append(model_sizes[model])
            changes.append(avg_change)
            colors.append(MODEL_COLORS.get(model, 'gray'))
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Create the scatter plot
    scatter = plt.scatter(sizes, changes, c=colors, s=100, alpha=0.7)
    
    # Add labels for each point
    for i, model in enumerate(models):
        plt.annotate(model, (sizes[i], changes[i]), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center')
    
    # Add a trend line
    z = np.polyfit(sizes, changes, 1)
    p = np.poly1d(z)
    plt.plot(sorted(sizes), p(sorted(sizes)), "r--", alpha=0.7)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Customize the plot
    plt.title('Model Size vs. Robustness Against Attacks', fontsize=16)
    plt.xlabel('Model Size (Billions of Parameters)', fontsize=14)
    plt.ylabel('Average Accuracy Change (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Use log scale for x-axis to better visualize the range of model sizes
    plt.xscale('log')
    plt.xlim(0.2, 30)
    
    # Add text explaining the interpretation
    plt.figtext(0.5, 0.01, 
                "Higher values indicate better robustness (less accuracy degradation)",
                ha="center", fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(PLOT_DIR, 'model_size_vs_robustness.png'), dpi=300)
    plt.close()
    
    print(f"Created model size vs. robustness plot: {os.path.join(PLOT_DIR, 'model_size_vs_robustness.png')}")

def plot_attack_comparison_bar(df):
    """
    Create a bar chart comparing the effectiveness of different attacks.
    
    Args:
        df (pandas.DataFrame): The loaded data
    """
    # Get all change columns
    change_cols = [col for col in df.columns if col.endswith('_change')]
    
    # Calculate average change for each attack type
    attack_avg_changes = df.groupby('attack_type')[change_cols].mean().mean(axis=1).sort_values()
    
    # Set up the plot
    plt.figure(figsize=(14, 8))
    
    # Create the bar chart
    bars = plt.bar(attack_avg_changes.index, attack_avg_changes.values, color='skyblue')
    
    # Color bars based on value (red for negative, green for positive)
    for i, bar in enumerate(bars):
        if attack_avg_changes.values[i] < 0:
            bar.set_color('#FF7F7F')  # Light red
        else:
            bar.set_color('#7FBF7F')  # Light green
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Customize the plot
    plt.title('Average Effectiveness of Different Attack Types', fontsize=16)
    plt.xlabel('Attack Type', fontsize=14)
    plt.ylabel('Average Accuracy Change (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(attack_avg_changes.values):
        plt.text(i, v + (0.5 if v >= 0 else -1.5), 
                f"{v:.1f}%", 
                ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(PLOT_DIR, 'attack_comparison_bar.png'), dpi=300)
    plt.close()
    
    print(f"Created attack comparison bar chart: {os.path.join(PLOT_DIR, 'attack_comparison_bar.png')}")

def main():
    """Main function to run the script."""
    print("Starting data analysis...")
    
    # Load data from the database
    df = load_data_from_db()
    
    if df is None or df.empty:
        print("No data found in the database. Please run store_results_db.py first.")
        return
    
    print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns.")
    
    # Generate plots
    plot_model_degradation_line(df)
    plot_attack_effectiveness_heatmap(df)
    plot_model_robustness_radar(df)
    plot_model_size_vs_robustness(df)
    plot_attack_comparison_bar(df)
    
    print(f"\nAll plots saved to {PLOT_DIR}")

if __name__ == "__main__":
    main()
