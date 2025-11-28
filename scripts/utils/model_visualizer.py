#!/usr/bin/env python3
"""
Research Paper Data Analysis Script for VLM Adversarial Robustness
==================================================================

Updated to use centralized_pipeline.db schema (2025)

This script generates publication-ready visualization plots for the research paper:
"Benchmarking Vision-Language Models' Robustness Against Multi-Modal Adversarial Threats"

Data Sources:
- inference_results: Raw VLM responses with performance metrics
- robustness_evaluation: Aggregated accuracy and robustness metrics
- attack_executions: Attack generation metadata

Output: results/research_plots/*.png
"""

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator
import sys
from datetime import datetime
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')

# Configure plotting style
plt.style.use('default')
sns.set_palette("husl")

# Database path - UPDATED for centralized_pipeline.db
DB_PATH = "results/centralized_pipeline.db"
# Output directory for research paper plots
PLOT_DIR = "results/research_plots"

# Ensure plot directory exists
os.makedirs(PLOT_DIR, exist_ok=True)

class DatabaseNotFoundError(Exception):
    """Custom exception for missing database."""
    pass


# =================== MODEL METADATA ===================
# Since the new schema is denormalized, we define model metadata here

MODEL_METADATA = {
    # Qwen models
    'Qwen25_VL_3B': {'display_name': 'Qwen2.5-VL-3B', 'family': 'Qwen', 'size_b': 3.75, 'size_category': '(3-4]B'},
    'Qwen25_VL_7B': {'display_name': 'Qwen2.5-VL-7B', 'family': 'Qwen', 'size_b': 8.29, 'size_category': '(7-8]B'},
    'Qwen2_VL_2B': {'display_name': 'Qwen2-VL-2B', 'family': 'Qwen', 'size_b': 2.21, 'size_category': '(2-3]B'},

    # Google models
    'Gemma3_VL_4B': {'display_name': 'Gemma-3-4B', 'family': 'Google', 'size_b': 4.3, 'size_category': '(4-5]B'},
    'PaliGemma_VL_3B': {'display_name': 'PaliGemma-3B', 'family': 'Google', 'size_b': 2.92, 'size_category': '(2-3]B'},

    # DeepSeek models
    'DeepSeek1_VL_1pt3B': {'display_name': 'DeepSeek-VL-1.3B', 'family': 'DeepSeek', 'size_b': 1.98, 'size_category': '(1-2]B'},
    'DeepSeek1_VL_7B': {'display_name': 'DeepSeek-VL-7B', 'family': 'DeepSeek', 'size_b': 7.34, 'size_category': '(7-8]B'},

    # SmolVLM2 models
    'SmolVLM2_pt25B': {'display_name': 'SmolVLM2-256M', 'family': 'SmolVLM', 'size_b': 0.256, 'size_category': '(0-1]B'},
    'SmolVLM2_pt5B': {'display_name': 'SmolVLM2-500M', 'family': 'SmolVLM', 'size_b': 0.507, 'size_category': '(0-1]B'},
    'SmolVLM2_2pt2B': {'display_name': 'SmolVLM2-2.2B', 'family': 'SmolVLM', 'size_b': 2.25, 'size_category': '(2-3]B'},

    # InternVL models
    'InternVL3_1B': {'display_name': 'InternVL3-1B', 'family': 'InternVL', 'size_b': 0.938, 'size_category': '(0-1]B'},
    'InternVL3_2B': {'display_name': 'InternVL3-2B', 'family': 'InternVL', 'size_b': 2.09, 'size_category': '(2-3]B'},
    'InternVL25_4B': {'display_name': 'InternVL2.5-4B', 'family': 'InternVL', 'size_b': 3.71, 'size_category': '(3-4]B'},

    # Other models
    'Moondream2_2B': {'display_name': 'Moondream2', 'family': 'Moondream', 'size_b': 1.93, 'size_category': '(1-2]B'},
    'LLAVA_1pt5_7B': {'display_name': 'LLaVA-1.5-7B', 'family': 'LLaVA', 'size_b': 7.06, 'size_category': '(7-8]B'},
    'LLAVA_v1pt6_Mistral_7B': {'display_name': 'LLaVA-v1.6-Mistral-7B', 'family': 'LLaVA', 'size_b': 7.57, 'size_category': '(7-8]B'},
}

ATTACK_CATEGORY_MAP = {
    'original': 'Original',
    'whitebox': 'White-Box',
    'blackbox': 'Black-Box',
}


class VLMDataAnalyzer:
    """VLM Data Analyzer for centralized_pipeline.db schema."""

    def __init__(self):
        self.db_path = DB_PATH
        self.conn = None
        self.model_info = MODEL_METADATA.copy()
        self.family_colors = {}
        self.size_colors = {}
        self.attack_colors = {}
        self.model_colors = {}

    def connect_db(self):
        """Establish database connection."""
        if not os.path.exists(self.db_path):
            raise DatabaseNotFoundError(f"Database not found: {self.db_path}")

        self.conn = sqlite3.connect(self.db_path)
        print(f"Connected to database: {self.db_path}")

    def close_db(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def load_dynamic_configurations(self):
        """Load model configurations and generate color palettes."""
        try:
            # Discover models from inference_results
            cursor = self.conn.cursor()
            cursor.execute('SELECT DISTINCT model_id FROM inference_results ORDER BY model_id')
            db_models = [row[0] for row in cursor.fetchall()]

            # Add any models from DB not in metadata
            for model in db_models:
                if model not in self.model_info:
                    self.model_info[model] = {
                        'display_name': model.replace('_', '-'),
                        'family': 'Unknown',
                        'size_b': 1.0,
                        'size_category': 'Unknown'
                    }

            # Generate color palettes
            self._generate_color_palettes()

            print(f"Loaded {len(self.model_info)} models, {len(db_models)} found in database")
            return True

        except Exception as e:
            print(f"Error loading configurations: {e}")
            return False

    def _generate_color_palettes(self):
        """Generate color palettes dynamically."""
        # Model families
        families = list(set([info['family'] for info in self.model_info.values()]))
        family_colors = sns.color_palette("husl", len(families))
        self.family_colors = {family: matplotlib.colors.rgb2hex(color)
                             for family, color in zip(sorted(families), family_colors)}

        # Size categories
        sizes = list(set([info['size_category'] for info in self.model_info.values()]))
        size_colors = sns.color_palette("viridis", len(sizes))
        self.size_colors = {size: matplotlib.colors.rgb2hex(color)
                           for size, color in zip(sorted(sizes), size_colors)}

        # Model-specific colors
        model_colors = sns.color_palette("tab20", len(self.model_info))
        self.model_colors = {}
        for i, model_name in enumerate(sorted(self.model_info.keys())):
            if i < len(model_colors):
                self.model_colors[model_name] = matplotlib.colors.rgb2hex(model_colors[i])
            else:
                hash_color = hashlib.md5(model_name.encode()).hexdigest()[:6]
                self.model_colors[model_name] = f"#{hash_color}"

        # Attack categories
        self.attack_colors = {
            "White-Box": "#FF7F0E",
            "Black-Box": "#1F77B4",
            "Original": "#2CA02C"
        }

        print(f"Generated color palettes: {len(families)} families, {len(sizes)} sizes")

    def load_robustness_data(self):
        """Load robustness data from robustness_evaluation table."""
        try:
            query = """
            SELECT
                eval_id,
                model_name,
                attack_type,
                task_name,
                accuracy,
                accuracy_change,
                effect,
                avg_inference_time_seconds,
                avg_gpu_memory_peak_mb,
                total_questions
            FROM robustness_evaluation
            """

            df = pd.read_sql_query(query, self.conn)

            if df.empty:
                print("Warning: robustness_evaluation table is empty")
                return None

            # Add model metadata
            df['model_family'] = df['model_name'].apply(
                lambda x: self.model_info.get(x, {}).get('family', 'Unknown')
            )
            df['size_category'] = df['model_name'].apply(
                lambda x: self.model_info.get(x, {}).get('size_category', 'Unknown')
            )
            df['display_name'] = df['model_name'].apply(
                lambda x: self.model_info.get(x, {}).get('display_name', x)
            )

            # Map attack categories
            df['attack_category'] = df['attack_type'].apply(
                lambda x: 'Original' if x == 'original' else ('White-Box' if x == 'whitebox' else 'Black-Box')
            )

            print(f"Loaded robustness data: {len(df)} records")
            return df

        except Exception as e:
            print(f"Error loading robustness data: {e}")
            return None

    def load_robustness_from_inference(self):
        """Calculate robustness data directly from inference_results if robustness_evaluation is empty."""
        try:
            # Get accuracy per model/attack/task combination
            query = """
            SELECT
                model_id as model_name,
                attack_type,
                task as task_name,
                epsilon_target,
                COUNT(*) as total_questions,
                AVG(inference_time_seconds) as avg_inference_time_seconds,
                AVG(gpu_peak_mb) as avg_gpu_memory_peak_mb
            FROM inference_results
            GROUP BY model_id, attack_type, task, epsilon_target
            """

            df = pd.read_sql_query(query, self.conn)

            if df.empty:
                print("Warning: No data in inference_results")
                return None

            # Add model metadata
            df['model_family'] = df['model_name'].apply(
                lambda x: self.model_info.get(x, {}).get('family', 'Unknown')
            )
            df['size_category'] = df['model_name'].apply(
                lambda x: self.model_info.get(x, {}).get('size_category', 'Unknown')
            )
            df['display_name'] = df['model_name'].apply(
                lambda x: self.model_info.get(x, {}).get('display_name', x)
            )

            # Map attack categories
            df['attack_category'] = df['attack_type'].apply(
                lambda x: 'Original' if x == 'original' else ('White-Box' if x == 'whitebox' else 'Black-Box')
            )

            # Placeholder accuracy (will need evaluation script to calculate actual values)
            df['accuracy'] = 50.0  # Placeholder
            df['accuracy_change'] = 0.0  # Placeholder

            print(f"Loaded inference data: {len(df)} records (accuracy is placeholder - run model_evaluation.py first)")
            return df

        except Exception as e:
            print(f"Error loading inference data: {e}")
            return None

    def load_performance_data(self):
        """Load performance metrics from inference_results table."""
        try:
            query = """
            SELECT
                model_id as model_name,
                attack_type,
                AVG(inference_time_seconds) as avg_inference_time_seconds,
                AVG(gpu_peak_mb) as avg_gpu_memory_peak_mb,
                AVG(gpu_reserved_mb) as avg_gpu_memory_reserved_mb,
                AVG(total_gpu_mb) as total_gpu_memory_mb,
                AVG(cpu_before_mb) as avg_cpu_memory_mb,
                AVG(loading_time_seconds) as model_loading_time_seconds,
                AVG(CASE WHEN was_cached = 1 THEN 1.0 ELSE 0.0 END) as cache_hit_ratio,
                COUNT(*) as total_questions
            FROM inference_results
            WHERE inference_time_seconds IS NOT NULL
            GROUP BY model_id, attack_type
            """

            df = pd.read_sql_query(query, self.conn)

            if df.empty:
                print("Warning: No performance data in inference_results")
                return None

            # Add model metadata
            df['family_name'] = df['model_name'].apply(
                lambda x: self.model_info.get(x, {}).get('family', 'Unknown')
            )
            df['size_category'] = df['model_name'].apply(
                lambda x: self.model_info.get(x, {}).get('size_category', 'Unknown')
            )

            print(f"Loaded performance data: {len(df)} records")
            return df

        except Exception as e:
            print(f"Error loading performance data: {e}")
            return None

    def load_attack_data(self):
        """Load attack generation data from attack_executions table."""
        try:
            query = """
            SELECT
                attack_name,
                attack_category,
                epsilon_target,
                AVG(execution_time_seconds) as avg_attack_time,
                COUNT(*) as total_executions,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_executions
            FROM attack_executions
            GROUP BY attack_name, attack_category, epsilon_target
            """

            df = pd.read_sql_query(query, self.conn)

            if df.empty:
                print("Warning: No attack data in attack_executions")
                return None

            df['success_rate'] = df['successful_executions'] / df['total_executions']

            print(f"Loaded attack data: {len(df)} records")
            return df

        except Exception as e:
            print(f"Error loading attack data: {e}")
            return None

    # =================== PLOT FUNCTIONS ===================

    def plot_baseline_performance(self, df):
        """Create baseline performance comparison (Original attack only)."""
        baseline_df = df[df['attack_type'] == 'original'].copy()
        if baseline_df.empty:
            print("Warning: No baseline data found")
            return

        # Aggregate by model
        baseline_agg = baseline_df.groupby('model_name').agg({
            'accuracy': 'mean',
            'display_name': 'first',
            'model_family': 'first'
        }).reset_index()

        plt.figure(figsize=(14, 8))
        baseline_sorted = baseline_agg.sort_values('accuracy', ascending=False)

        bars = plt.bar(range(len(baseline_sorted)), baseline_sorted['accuracy'])

        # Color by family
        for i, (_, row) in enumerate(baseline_sorted.iterrows()):
            family = row['model_family']
            bars[i].set_color(self.family_colors.get(family, 'gray'))

        plt.title('Baseline Performance (No Attack)', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.xticks(range(len(baseline_sorted)), baseline_sorted['display_name'], rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)

        # Add value labels
        for i, v in enumerate(baseline_sorted['accuracy']):
            plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '01_baseline_performance.png'), dpi=300)
        plt.close()
        print("Created: 01_baseline_performance.png")

    def plot_attack_effectiveness_heatmap(self, df):
        """Create heatmap of attack effectiveness across models."""
        # Filter out original (no attack)
        attack_df = df[df['attack_type'] != 'original'].copy()
        if attack_df.empty:
            print("Warning: No attack data for heatmap")
            return

        heatmap_data = attack_df.pivot_table(
            index='attack_type',
            columns='display_name',
            values='accuracy_change',
            aggfunc='mean'
        )

        if heatmap_data.empty:
            print("Warning: Cannot create heatmap - no data")
            return

        plt.figure(figsize=(14, 10))
        sns.heatmap(heatmap_data,
                    annot=True, fmt=".1f",
                    cmap="RdYlGn_r", center=0,
                    linewidths=.5,
                    cbar_kws={'label': 'Accuracy Change (%)'})

        plt.title('Attack Effectiveness Across Models', fontsize=16)
        plt.ylabel('Attack Type', fontsize=14)
        plt.xlabel('Model', fontsize=14)
        plt.tight_layout()

        plt.savefig(os.path.join(PLOT_DIR, '03_attack_effectiveness_heatmap.png'), dpi=300)
        plt.close()
        print("Created: 03_attack_effectiveness_heatmap.png")

    def plot_raw_accuracy_heatmap(self, df):
        """Create heatmap of raw accuracy values."""
        heatmap_data = df.pivot_table(
            index='attack_type',
            columns='display_name',
            values='accuracy',
            aggfunc='mean'
        )

        if heatmap_data.empty:
            print("Warning: Cannot create raw accuracy heatmap - no data")
            return

        plt.figure(figsize=(14, 10))
        sns.heatmap(heatmap_data,
                    annot=True, fmt=".1f",
                    cmap="YlOrRd", vmin=0, vmax=100,
                    linewidths=.5,
                    cbar_kws={'label': 'Accuracy (%)'})

        plt.title('Raw Accuracy Values Across Models and Attacks', fontsize=16)
        plt.ylabel('Attack Type', fontsize=14)
        plt.xlabel('Model', fontsize=14)
        plt.tight_layout()

        plt.savefig(os.path.join(PLOT_DIR, '04_raw_accuracy_heatmap.png'), dpi=300)
        plt.close()
        print("Created: 04_raw_accuracy_heatmap.png")

    def plot_model_family_robustness(self, df):
        """Create bar chart of model family robustness."""
        attack_df = df[df['attack_type'] != 'original'].copy()
        if attack_df.empty:
            print("Warning: No attack data for family robustness")
            return

        family_avg = attack_df.groupby('model_family')['accuracy_change'].mean().sort_values()

        plt.figure(figsize=(12, 8))
        bars = plt.bar(family_avg.index, family_avg.values)

        for i, bar in enumerate(bars):
            family = family_avg.index[i]
            bar.set_color(self.family_colors.get(family, 'gray'))

        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.title('Average Robustness by Model Family', fontsize=16)
        plt.xlabel('Model Family', fontsize=14)
        plt.ylabel('Average Accuracy Change (%)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)

        for i, v in enumerate(family_avg.values):
            plt.text(i, v + (0.5 if v >= 0 else -1.5),
                    f"{v:.1f}%", ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '05_model_family_robustness.png'), dpi=300)
        plt.close()
        print("Created: 05_model_family_robustness.png")

    def plot_size_category_robustness(self, df):
        """Create bar chart of size category robustness."""
        attack_df = df[df['attack_type'] != 'original'].copy()
        if attack_df.empty:
            print("Warning: No attack data for size robustness")
            return

        size_order = ['(0-1]B', '(1-2]B', '(2-3]B', '(3-4]B', '(4-5]B', '(5-6]B', '(6-7]B', '(7-8]B']
        size_order = [size for size in size_order if size in attack_df['size_category'].unique()]

        size_avg = attack_df.groupby('size_category')['accuracy_change'].mean()
        size_avg = size_avg.reindex([s for s in size_order if s in size_avg.index])

        plt.figure(figsize=(12, 8))
        bars = plt.bar(size_avg.index, size_avg.values)

        for i, bar in enumerate(bars):
            size = size_avg.index[i]
            bar.set_color(self.size_colors.get(size, 'gray'))

        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.title('Average Robustness by Model Size Category', fontsize=16)
        plt.xlabel('Model Size Category', fontsize=14)
        plt.ylabel('Average Accuracy Change (%)', fontsize=14)
        plt.grid(True, axis='y', alpha=0.3)

        for i, v in enumerate(size_avg.values):
            plt.text(i, v + (0.5 if v >= 0 else -1.5),
                    f"{v:.1f}%", ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '06_size_category_robustness.png'), dpi=300)
        plt.close()
        print("Created: 06_size_category_robustness.png")

    def plot_architecture_vulnerability_analysis(self, df):
        """Create architecture vulnerability visualization."""
        attacks_only_df = df[df['attack_type'] != 'original'].copy()
        if attacks_only_df.empty:
            print("Warning: No attack data for vulnerability analysis")
            return

        # Aggregate by model
        vulnerability_data = attacks_only_df.groupby(['model_name', 'model_family', 'size_category']).agg({
            'accuracy_change': 'mean',
            'display_name': 'first'
        }).reset_index()

        vulnerability_data = vulnerability_data.sort_values('accuracy_change')

        plt.figure(figsize=(16, 10))

        bars = plt.bar(range(len(vulnerability_data)), vulnerability_data['accuracy_change'],
                      alpha=0.8, width=0.6)

        # Color bars by family
        for i, (_, row) in enumerate(vulnerability_data.iterrows()):
            family_color = self.family_colors.get(row['model_family'], 'gray')
            bars[i].set_color(family_color)
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(1)

        # Add value labels
        for i, (_, row) in enumerate(vulnerability_data.iterrows()):
            plt.text(i, row['accuracy_change'] + (0.2 if row['accuracy_change'] >= 0 else -0.5),
                    f"{row['accuracy_change']:.1f}%", ha='center', fontsize=9, weight='bold')

        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.7, linewidth=1)
        plt.title('Architecture Vulnerability Analysis: Individual Model Performance', fontsize=16, pad=20)
        plt.xlabel('VLM Models (sorted by vulnerability)', fontsize=14)
        plt.ylabel('Average Accuracy Change (%)', fontsize=14)
        plt.xticks(range(len(vulnerability_data)), vulnerability_data['display_name'],
                  rotation=45, ha='right', fontsize=9)
        plt.grid(True, axis='y', alpha=0.3)

        # Create legend for families
        family_handles = [plt.Rectangle((0,0),1,1, color=color, alpha=0.8, label=family)
                         for family, color in self.family_colors.items()
                         if family in vulnerability_data['model_family'].values]
        if family_handles:
            plt.legend(handles=family_handles, title="Model Family",
                      loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '08_architecture_vulnerability_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 08_architecture_vulnerability_analysis.png")

    def plot_attack_transferability_analysis(self, df):
        """Create attack transferability analysis."""
        attacks_only_df = df[df['attack_type'] != 'original'].copy()
        if attacks_only_df.empty:
            print("Warning: No attack data for transferability analysis")
            return

        attack_success = attacks_only_df.groupby(['attack_type', 'attack_category'])['accuracy_change'].agg(['mean', 'std']).reset_index()
        attack_success['degradation'] = -attack_success['mean']
        attack_success = attack_success.sort_values('degradation', ascending=False)

        plt.figure(figsize=(14, 8))

        x_pos = range(len(attack_success))
        bars = plt.bar(x_pos, attack_success['degradation'],
                      yerr=attack_success['std'].fillna(0), capsize=4, alpha=0.8, width=0.7)

        category_colors = {'White-Box': '#FF6B35', 'Black-Box': '#004E89', 'Original': '#2CA02C'}

        for i, (_, row) in enumerate(attack_success.iterrows()):
            color = category_colors.get(row['attack_category'], 'gray')
            bars[i].set_color(color)
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(1)

        # Create legend
        category_handles = [plt.Rectangle((0,0),1,1, color=color, label=f'{cat} Attacks')
                          for cat, color in category_colors.items()
                          if cat in attack_success['attack_category'].values]
        plt.legend(handles=category_handles, title="Attack Type", loc='upper right', fontsize=12)

        plt.title('Attack Effectiveness: Performance Degradation Analysis\n(Sorted by Effectiveness)', fontsize=16)
        plt.xlabel('Attack Methods', fontsize=14)
        plt.ylabel('Average Performance Degradation (%)', fontsize=14)
        plt.xticks(x_pos, attack_success['attack_type'], rotation=45, ha='right', fontsize=10)
        plt.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '10_attack_transferability_analysis.png'), dpi=300)
        plt.close()
        print("Created: 10_attack_transferability_analysis.png")

    def plot_size_vs_memory(self, df):
        """Create scatter plot of model size vs GPU memory."""
        if df is None or df.empty:
            print("Warning: No performance data for size vs memory plot")
            return

        # Aggregate by model
        model_data = df.groupby(['model_name', 'family_name']).agg({
            'avg_gpu_memory_peak_mb': 'mean'
        }).reset_index()

        # Add model metadata
        model_data['display_name'] = model_data['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('display_name', x)
        )
        model_data['model_size_b'] = model_data['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('size_b', 1.0)
        )

        # Filter out rows with missing data
        model_data = model_data.dropna(subset=['avg_gpu_memory_peak_mb'])

        if model_data.empty:
            print("Warning: No valid data for size vs memory plot")
            return

        plt.figure(figsize=(14, 10))

        for _, row in model_data.iterrows():
            family = row['family_name']
            plt.scatter(row['model_size_b'], row['avg_gpu_memory_peak_mb'] / 1000,
                       color=self.family_colors.get(family, 'gray'),
                       s=120, alpha=0.8,
                       edgecolors='black', linewidth=0.5)

            plt.annotate(row['display_name'],
                        (row['model_size_b'], row['avg_gpu_memory_peak_mb'] / 1000),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, weight='bold', rotation=45,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9))

        # Add trend line
        x = model_data['model_size_b']
        y = model_data['avg_gpu_memory_peak_mb'] / 1000
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(x.min(), x.max(), 100)
            plt.plot(x_trend, p(x_trend), "k--", alpha=0.8, linewidth=2.5, label="Memory Trend")

        plt.xlabel('Model Size (Billion Parameters)', fontsize=14)
        plt.ylabel('Average GPU Memory Peak (GB)', fontsize=14)
        plt.title('Model Size vs Average GPU Memory Peak', fontsize=16)
        plt.grid(True, alpha=0.3)

        # Legend for families
        family_handles = [plt.scatter([], [], color=color, s=100, label=family)
                        for family, color in self.family_colors.items()
                        if family in model_data['family_name'].values]
        if family_handles:
            plt.legend(handles=family_handles, title="Model Family",
                      bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '12_size_vs_memory.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 12_size_vs_memory.png")

    def plot_top_models_radar(self, perf_df, robust_df):
        """Create radar chart comparing top models."""
        if perf_df is None or perf_df.empty:
            print("Warning: No performance data for radar chart")
            return

        # Aggregate performance metrics by model
        model_scores = perf_df.groupby('model_name').agg({
            'avg_inference_time_seconds': 'mean',
            'avg_gpu_memory_peak_mb': 'mean',
            'model_loading_time_seconds': 'mean',
            'cache_hit_ratio': 'mean',
            'family_name': 'first'
        }).reset_index()

        # Add model metadata
        model_scores['display_name'] = model_scores['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('display_name', x)
        )
        model_scores['model_size_b'] = model_scores['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('size_b', 1.0)
        )

        # Calculate normalized scores (0-1, higher = better)
        model_scores['size_efficiency'] = 1 / (model_scores['model_size_b'] + 1)
        model_scores['memory_efficiency'] = 1 / (model_scores['avg_gpu_memory_peak_mb'].fillna(1000) / 1000 + 1)
        model_scores['loading_speed'] = 1 / (model_scores['model_loading_time_seconds'].fillna(10) + 1)
        model_scores['inference_speed'] = 1 / (model_scores['avg_inference_time_seconds'].fillna(1) + 1)
        model_scores['cache_efficiency'] = model_scores['cache_hit_ratio'].fillna(0.5)

        # Add robustness score if available
        if robust_df is not None and not robust_df.empty:
            attack_df = robust_df[robust_df['attack_type'] != 'original']
            if not attack_df.empty:
                robustness = attack_df.groupby('model_name')['accuracy_change'].mean()
                robustness_norm = (robustness - robustness.min()) / (robustness.max() - robustness.min() + 0.01)
                model_scores['robustness_score'] = model_scores['model_name'].map(robustness_norm).fillna(0.5)
            else:
                model_scores['robustness_score'] = 0.5
        else:
            model_scores['robustness_score'] = 0.5

        # Calculate overall score and select top 6
        model_scores['overall_score'] = (
            model_scores['size_efficiency'] + model_scores['memory_efficiency'] +
            model_scores['loading_speed'] + model_scores['inference_speed'] +
            model_scores['cache_efficiency'] + model_scores['robustness_score']
        ) / 6

        top_models = model_scores.nlargest(min(6, len(model_scores)), 'overall_score')

        if len(top_models) < 3:
            print("Warning: Not enough models for radar chart (need at least 3)")
            return

        # Create radar chart
        categories = ['Size Efficiency', 'Memory Efficiency', 'Loading Speed',
                     'Inference Speed', 'Cache Efficiency', 'Robustness']

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

        for i, (_, model) in enumerate(top_models.iterrows()):
            values = [model['size_efficiency'], model['memory_efficiency'], model['loading_speed'],
                     model['inference_speed'], model['cache_efficiency'], model['robustness_score']]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2.5, markersize=6,
                   label=f"{model['display_name']}", color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title('VLM Performance Analysis\nMulti-Dimensional Comparison', size=16, pad=30, weight='bold')
        ax.legend(loc='center', bbox_to_anchor=(1.35, 0.5), fontsize=10)
        ax.grid(True, alpha=0.4)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '13_top_models_radar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("Created: 13_top_models_radar.png")

    def plot_attack_metrics_comparison(self, attack_df):
        """Create attack metrics comparison plot."""
        if attack_df is None or attack_df.empty:
            print("Warning: No attack data for metrics comparison")
            return

        plt.figure(figsize=(14, 8))

        colors = {'whitebox': '#FF6B35', 'blackbox': '#004E89'}

        for category in attack_df['attack_category'].unique():
            cat_data = attack_df[attack_df['attack_category'] == category]
            plt.scatter(cat_data['epsilon_target'], cat_data['avg_attack_time'],
                       s=cat_data['success_rate'] * 200 + 50,
                       alpha=0.7, color=colors.get(category, 'gray'),
                       label=f'{category.title()} Attacks', edgecolors='black', linewidth=1.5)

            # Add labels
            for _, row in cat_data.iterrows():
                plt.annotate(row['attack_name'],
                           (row['epsilon_target'], row['avg_attack_time']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.xlabel('Epsilon Target', fontsize=14)
        plt.ylabel('Average Attack Time (seconds)', fontsize=14)
        plt.title('Attack Generation: Epsilon vs Time\n(Bubble size = Success Rate)', fontsize=16)
        plt.legend(title="Attack Category", fontsize=11)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '15_attack_raw_metrics_comparison.png'), dpi=300)
        plt.close()
        print("Created: 15_attack_raw_metrics_comparison.png")

    def generate_all_plots(self):
        """Generate all research paper visualization plots."""
        print(f"\nStarting research paper plot generation...")
        print(f"Target directory: {PLOT_DIR}")

        # Load robustness data
        print(f"\nLoading robustness data...")
        robustness_df = self.load_robustness_data()

        # If robustness_evaluation is empty, try inference_results
        if robustness_df is None or robustness_df.empty:
            print("robustness_evaluation empty, trying inference_results...")
            robustness_df = self.load_robustness_from_inference()

        if robustness_df is not None and not robustness_df.empty:
            print("\nGenerating robustness plots...")
            self.plot_baseline_performance(robustness_df)
            self.plot_attack_effectiveness_heatmap(robustness_df)
            self.plot_raw_accuracy_heatmap(robustness_df)
            self.plot_model_family_robustness(robustness_df)
            self.plot_size_category_robustness(robustness_df)
            self.plot_architecture_vulnerability_analysis(robustness_df)
            self.plot_attack_transferability_analysis(robustness_df)
        else:
            print("Warning: No robustness data found - skipping robustness plots")

        # Load performance data
        print(f"\nLoading performance data...")
        performance_df = self.load_performance_data()

        if performance_df is not None and not performance_df.empty:
            print("\nGenerating performance plots...")
            self.plot_size_vs_memory(performance_df)
            self.plot_top_models_radar(performance_df, robustness_df)
        else:
            print("Warning: No performance data found - skipping performance plots")

        # Load attack data
        print(f"\nLoading attack data...")
        attack_df = self.load_attack_data()

        if attack_df is not None and not attack_df.empty:
            print("\nGenerating attack plots...")
            self.plot_attack_metrics_comparison(attack_df)
        else:
            print("Warning: No attack data found - skipping attack plots")

        # Count generated plots
        plot_count = len([f for f in os.listdir(PLOT_DIR) if f.endswith('.png')])
        print(f"\nPlot generation complete!")
        print(f"Generated {plot_count} plots in: {PLOT_DIR}")


def main():
    """Main execution function."""
    print("=" * 60)
    print("VLM Data Analysis & Visualization")
    print("Database: centralized_pipeline.db")
    print("=" * 60)

    analyzer = VLMDataAnalyzer()

    try:
        # Connect to database
        analyzer.connect_db()

        # Load configurations
        if not analyzer.load_dynamic_configurations():
            print("Failed to load configurations from database")
            return

        # Generate all plots
        analyzer.generate_all_plots()

    except DatabaseNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the evaluation pipeline first to generate the database.")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

    finally:
        analyzer.close_db()
        print(f"\nAnalysis complete!")


if __name__ == "__main__":
    main()
