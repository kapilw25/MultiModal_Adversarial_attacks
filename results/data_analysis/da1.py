#!/usr/bin/env python3
"""
Data analysis script for generating plots from the normalized robustness database.
This script reads data from the SQLite database with the new normalized structure
and generates various plots to visualize model performance across different dimensions:
- Attack types
- Model families
- Size categories
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

# Define colors for model families
FAMILY_COLORS = {
    "OpenAI": "#0077B6",          # Blue
    "Qwen VL": "#FF7F0E",         # Orange
    "Google": "#2CA02C",          # Green
    "DeepSeek VL": "#D62728",     # Red
    "SmolVLM": "#9467BD",         # Purple
    "Microsoft": "#E377C2",       # Pink
    "Moondream": "#17BECF",       # Cyan
    "GLM Edge": "#FFBB78",        # Light Orange
    "InternVL": "#AEC7E8",        # Light Blue
    "Salesforce": "#7F7F7F",      # Gray
    "LLaVA Hybrid": "#BCBD22",    # Olive
    "DeepSeek VL2": "#8C564B",    # Brown
    "Other": "#C7C7C7"            # Light Gray
}

# Define colors for size categories
SIZE_COLORS = {
    "(0-1]B": "#FDE725",          # Yellow
    "(1-2]B": "#5DC863",          # Green
    "(2-3]B": "#21908C",          # Teal
    "(3-4]B": "#3B528B",          # Blue
    "(4-5]B": "#440154",          # Purple
    "(5-6]B": "#FDE725",          # Yellow
    "(6-7]B": "#5DC863",          # Green
    "Cloud API": "#000000"         # Black
}

# Define colors for attack categories
ATTACK_COLORS = {
    "Transfer": "#FF7F0E",        # Orange
    "Black-Box": "#1F77B4",       # Blue
    "Original": "#2CA02C"         # Green
}

def load_data_from_db():
    """
    Load data from the normalized SQLite database.
    
    Returns:
        pandas.DataFrame: The loaded data with all dimension information
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Query that joins all dimension tables with the fact table
        query = """
        SELECT 
            r.result_id,
            t.task_name,
            a.attack_name,
            a.attack_category,
            m.model_name,
            f.family_name AS model_family,
            s.size_range AS size_category,
            r.accuracy,
            r.accuracy_change,
            r.timestamp
        FROM results r
        JOIN attack_types a ON r.attack_id = a.attack_id
        JOIN models m ON r.model_id = m.model_id
        JOIN tasks t ON r.task_id = t.task_id
        JOIN model_families f ON m.family_id = f.family_id
        JOIN size_categories s ON m.size_id = s.size_id
        """
        
        df = pd.read_sql_query(query, conn)
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
    # Select a subset of models to avoid overcrowding the plot
    # Prioritize models mentioned in the README
    selected_models = ['gpt4o', 'qwen25_vl_3b', 'gemma3_vl_4b']
    
    # Filter to only include selected models that exist in the data
    selected_models = [model for model in selected_models if model in df['model_name'].unique()]
    
    # If we don't have enough models, add more from what's available
    if len(selected_models) < 3:
        additional_models = [model for model in df['model_name'].unique() if model not in selected_models]
        selected_models.extend(additional_models[:3 - len(selected_models)])
    
    # Create a pivot table for plotting
    plot_df = df[df['model_name'].isin(selected_models)].pivot_table(
        index='attack_name',
        columns='model_name',
        values='accuracy_change',
        aggfunc='mean'
    ).reset_index()
    
    # Sort by attack name to ensure consistent ordering
    plot_df = plot_df.sort_values('attack_name')
    
    # Set up the plot
    plt.figure(figsize=(14, 8))
    
    # Plot each model
    for model in selected_models:
        if model in plot_df.columns:
            display_name = DISPLAY_NAMES.get(model, model)
            color = MODEL_COLORS.get(model, 'gray')
            
            plt.plot(plot_df['attack_name'], plot_df[model], 
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
    # Create a pivot table for the heatmap
    # Rows are attack types, columns are models
    heatmap_data = df.pivot_table(
        index='attack_name',
        columns='model_name',
        values='accuracy_change',
        aggfunc='mean'
    )
    
    # Rename columns to display names
    heatmap_data.columns = [DISPLAY_NAMES.get(col, col) for col in heatmap_data.columns]
    
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

def plot_model_family_robustness(df):
    """
    Create a bar chart showing average robustness by model family.
    
    Args:
        df (pandas.DataFrame): The loaded data
    """
    # Calculate average accuracy change by model family
    family_avg = df.groupby('model_family')['accuracy_change'].mean().sort_values()
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Create the bar chart
    bars = plt.bar(family_avg.index, family_avg.values)
    
    # Color bars based on model family
    for i, bar in enumerate(bars):
        family = family_avg.index[i]
        bar.set_color(FAMILY_COLORS.get(family, 'gray'))
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Customize the plot
    plt.title('Average Robustness by Model Family', fontsize=16)
    plt.xlabel('Model Family', fontsize=14)
    plt.ylabel('Average Accuracy Change (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(family_avg.values):
        plt.text(i, v + (0.5 if v >= 0 else -1.5), 
                f"{v:.1f}%", 
                ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(PLOT_DIR, 'model_family_robustness.png'), dpi=300)
    plt.close()
    
    print(f"Created model family robustness bar chart: {os.path.join(PLOT_DIR, 'model_family_robustness.png')}")

def plot_size_category_robustness(df):
    """
    Create a bar chart showing average robustness by size category.
    
    Args:
        df (pandas.DataFrame): The loaded data
    """
    # Calculate average accuracy change by size category
    # Define a custom sort order for size categories
    size_order = ['(0-1]B', '(1-2]B', '(2-3]B', '(3-4]B', '(4-5]B', '(5-6]B', '(6-7]B', 'Cloud API']
    
    # Filter to only include size categories that exist in the data
    size_order = [size for size in size_order if size in df['size_category'].unique()]
    
    # Calculate average accuracy change by size category
    size_avg = df.groupby('size_category')['accuracy_change'].mean()
    
    # Reindex based on the custom sort order
    size_avg = size_avg.reindex(size_order)
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Create the bar chart
    bars = plt.bar(size_avg.index, size_avg.values)
    
    # Color bars based on size category
    for i, bar in enumerate(bars):
        size = size_avg.index[i]
        bar.set_color(SIZE_COLORS.get(size, 'gray'))
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Customize the plot
    plt.title('Average Robustness by Model Size Category', fontsize=16)
    plt.xlabel('Model Size Category', fontsize=14)
    plt.ylabel('Average Accuracy Change (%)', fontsize=14)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(size_avg.values):
        plt.text(i, v + (0.5 if v >= 0 else -1.5), 
                f"{v:.1f}%", 
                ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(PLOT_DIR, 'size_category_robustness.png'), dpi=300)
    plt.close()
    
    print(f"Created size category robustness bar chart: {os.path.join(PLOT_DIR, 'size_category_robustness.png')}")

def plot_attack_category_effectiveness(df):
    """
    Create a bar chart showing the effectiveness of different attack categories.
    
    Args:
        df (pandas.DataFrame): The loaded data
    """
    # Calculate average change for each attack category
    attack_cat_avg = df.groupby('attack_category')['accuracy_change'].mean().sort_values()
    
    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Create the bar chart
    bars = plt.bar(attack_cat_avg.index, attack_cat_avg.values)
    
    # Color bars based on attack category
    for i, bar in enumerate(bars):
        category = attack_cat_avg.index[i]
        bar.set_color(ATTACK_COLORS.get(category, 'gray'))
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Customize the plot
    plt.title('Average Effectiveness by Attack Category', fontsize=16)
    plt.xlabel('Attack Category', fontsize=14)
    plt.ylabel('Average Accuracy Change (%)', fontsize=14)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for i, v in enumerate(attack_cat_avg.values):
        plt.text(i, v + (0.5 if v >= 0 else -1.5), 
                f"{v:.1f}%", 
                ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(PLOT_DIR, 'attack_category_effectiveness.png'), dpi=300)
    plt.close()
    
    print(f"Created attack category effectiveness bar chart: {os.path.join(PLOT_DIR, 'attack_category_effectiveness.png')}")

def plot_family_vs_attack_category_heatmap(df):
    """
    Create a heatmap showing the effectiveness of attack categories across model families.
    
    Args:
        df (pandas.DataFrame): The loaded data
    """
    # Create a pivot table for the heatmap
    heatmap_data = df.pivot_table(
        index='model_family',
        columns='attack_category',
        values='accuracy_change',
        aggfunc='mean'
    )
    
    # Set up the plot
    plt.figure(figsize=(10, 8))
    
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
    plt.title('Attack Category Effectiveness Across Model Families', fontsize=16)
    plt.ylabel('Model Family', fontsize=14)
    plt.xlabel('Attack Category', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(PLOT_DIR, 'family_vs_attack_category_heatmap.png'), dpi=300)
    plt.close()
    
    print(f"Created family vs attack category heatmap: {os.path.join(PLOT_DIR, 'family_vs_attack_category_heatmap.png')}")

def plot_size_vs_attack_category_heatmap(df):
    """
    Create a heatmap showing the effectiveness of attack categories across model size categories.
    
    Args:
        df (pandas.DataFrame): The loaded data
    """
    # Define a custom sort order for size categories
    size_order = ['(0-1]B', '(1-2]B', '(2-3]B', '(3-4]B', '(4-5]B', '(5-6]B', '(6-7]B', 'Cloud API']
    
    # Filter to only include size categories that exist in the data
    size_order = [size for size in size_order if size in df['size_category'].unique()]
    
    # Create a pivot table for the heatmap
    heatmap_data = df.pivot_table(
        index='size_category',
        columns='attack_category',
        values='accuracy_change',
        aggfunc='mean'
    )
    
    # Reindex based on the custom sort order
    heatmap_data = heatmap_data.reindex(size_order)
    
    # Set up the plot
    plt.figure(figsize=(10, 8))
    
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
    plt.title('Attack Category Effectiveness Across Model Size Categories', fontsize=16)
    plt.ylabel('Model Size Category', fontsize=14)
    plt.xlabel('Attack Category', fontsize=14)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(PLOT_DIR, 'size_vs_attack_category_heatmap.png'), dpi=300)
    plt.close()
    
    print(f"Created size vs attack category heatmap: {os.path.join(PLOT_DIR, 'size_vs_attack_category_heatmap.png')}")

def plot_3d_dimension_analysis(df):
    """
    Create a 3D scatter plot showing the relationship between model family, size category, and robustness.
    
    Args:
        df (pandas.DataFrame): The loaded data
    """
    # Calculate average accuracy change by model family and size category
    grouped_data = df.groupby(['model_family', 'size_category'])['accuracy_change'].mean().reset_index()
    
    # Set up the plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a mapping of model families to numeric values for the x-axis
    families = grouped_data['model_family'].unique()
    family_to_num = {family: i for i, family in enumerate(families)}
    
    # Create a mapping of size categories to numeric values for the y-axis
    size_order = ['(0-1]B', '(1-2]B', '(2-3]B', '(3-4]B', '(4-5]B', '(5-6]B', '(6-7]B', 'Cloud API']
    size_to_num = {size: i for i, size in enumerate(size_order) if size in grouped_data['size_category'].unique()}
    
    # Create lists for plotting
    x = [family_to_num[family] for family in grouped_data['model_family']]
    y = [size_to_num.get(size, -1) for size in grouped_data['size_category']]
    z = grouped_data['accuracy_change'].values
    colors = [FAMILY_COLORS.get(family, 'gray') for family in grouped_data['model_family']]
    
    # Create the scatter plot
    scatter = ax.scatter(x, y, z, c=colors, s=100, alpha=0.7)
    
    # Add labels for each point
    for i in range(len(x)):
        ax.text(x[i], y[i], z[i], 
                f"{grouped_data['model_family'].iloc[i]}, {grouped_data['size_category'].iloc[i]}", 
                fontsize=8)
    
    # Customize the plot
    ax.set_title('Model Family, Size Category, and Robustness', fontsize=16)
    ax.set_xlabel('Model Family', fontsize=14)
    ax.set_ylabel('Size Category', fontsize=14)
    ax.set_zlabel('Average Accuracy Change (%)', fontsize=14)
    
    # Set custom tick labels
    ax.set_xticks(list(family_to_num.values()))
    ax.set_xticklabels(list(family_to_num.keys()), rotation=45, ha='right')
    
    ax.set_yticks(list(size_to_num.values()))
    ax.set_yticklabels(list(size_to_num.keys()))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(PLOT_DIR, '3d_dimension_analysis.png'), dpi=300)
    plt.close()
    
    print(f"Created 3D dimension analysis plot: {os.path.join(PLOT_DIR, '3d_dimension_analysis.png')}")

def main():
    """Main function to run the script."""
    print("Starting data analysis with normalized database structure...")
    
    # Load data from the database
    df = load_data_from_db()
    
    if df is None or df.empty:
        print("No data found in the database. Please run store_results_db.py first.")
        return
    
    print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns.")
    
    # Generate plots
    plot_model_degradation_line(df)
    plot_attack_effectiveness_heatmap(df)
    plot_model_family_robustness(df)
    plot_size_category_robustness(df)
    plot_attack_category_effectiveness(df)
    plot_family_vs_attack_category_heatmap(df)
    plot_size_vs_attack_category_heatmap(df)
    plot_3d_dimension_analysis(df)
    
    print(f"\nAll plots saved to {PLOT_DIR}")

if __name__ == "__main__":
    main()
