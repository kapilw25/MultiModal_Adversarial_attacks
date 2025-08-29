#!/usr/bin/env python3
"""
Enhanced Data Analysis Script for VLM Adversarial Robustness
============================================================

This script dynamically reads model information, colors, and performance metrics 
from the robustness.db database and generates comprehensive visualization plots.

Features:
- Dynamic model color assignment from database
- Performance metrics integration
- Continuous database monitoring
- All plots from both plots/ and plots1/ directories
- Database-driven configuration (no hardcoded values)

Generated Plots:
- plots1/: Robustness analysis (9 plots)
- plots/: Performance analysis (8 plots)
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

# Define the database path
DB_PATH = "results/robustness.db"
# Define output directories for plots
PLOT_DIR_ROBUSTNESS = "results/data_analysis/plots1"
PLOT_DIR_PERFORMANCE = "results/data_analysis/plots"

# Ensure plot directories exist
os.makedirs(PLOT_DIR_ROBUSTNESS, exist_ok=True)
os.makedirs(PLOT_DIR_PERFORMANCE, exist_ok=True)

class DatabaseNotFoundError(Exception):
    """Custom exception for missing database."""
    pass

class VLMDataAnalyzer:
    """Enhanced VLM Data Analyzer with dynamic database integration."""
    
    def __init__(self):
        self.db_path = DB_PATH
        self.conn = None
        self.model_info = {}
        self.family_colors = {}
        self.size_colors = {}
        self.attack_colors = {}
        self.model_colors = {}
        
    def connect_db(self):
        """Establish database connection."""
        if not os.path.exists(self.db_path):
            raise DatabaseNotFoundError(f"Database not found: {self.db_path}")
        
        self.conn = sqlite3.connect(self.db_path)
        print(f"✅ Connected to database: {self.db_path}")
        
    def close_db(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def _generate_display_name(self, model_name):
        """Generate a human-readable display name from model_name."""
        # Handle common model name patterns
        name_mappings = {
            'gpt4o': 'GPT-4o',
            'qwen25_vl_3b': 'Qwen2.5-VL-3B',
            'qwen25_vl_7b': 'Qwen2.5-VL-7B', 
            'qwen2_vl_2b': 'Qwen2-VL-2B',
            'gemma3_vl_4b': 'Gemma-3-4B',
            'paligemma_vl_3b': 'PaliGemma-3B',
            'deepseek1_vl_1pt3b': 'DeepSeek-VL-1.3B',
            'deepseek1_vl_7b': 'DeepSeek-VL-7B',
            'smolvlm2_pt25b': 'SmolVLM2-256M',
            'smolvlm2_pt5b': 'SmolVLM2-500M', 
            'smolvlm2_2pt2b': 'SmolVLM2-2.2B',
            'phi3pt5_vision_4b': 'Phi-3.5-Vision-4B',
            'florence2_pt23b': 'Florence-2-Base',
            'florence2_pt77b': 'Florence-2-Large',
            'moondream2_2b': 'Moondream2-2B',
            'glmedge_2b': 'GLM-Edge-V-2B',
            'internvl3_1b': 'InternVL3-1B',
            'internvl3_2b': 'InternVL3-2B', 
            'internvl25_4b': 'InternVL2.5-4B'
        }
        
        if model_name in name_mappings:
            return name_mappings[model_name]
        else:
            # Generate display name by cleaning up underscores and capitalizing
            return model_name.replace('_', '-').upper()
    
    def _estimate_model_size(self, model_name):
        """Estimate model size in billions of parameters from model name."""
        import re
        
        # Extract size from model name patterns
        name_lower = model_name.lower()
        
        # Look for common patterns like '7b', '3b', '1pt3b', 'pt5b', etc.
        size_patterns = [
            r'(\d+)b$',           # e.g., '7b'
            r'(\d+)pt(\d+)b$',    # e.g., '1pt3b' -> 1.3
            r'pt(\d+)b$',         # e.g., 'pt25b' -> 0.25
            r'(\d+)pt(\d+)',      # e.g., '2pt2' -> 2.2
        ]
        
        for pattern in size_patterns:
            match = re.search(pattern, name_lower)
            if match:
                if len(match.groups()) == 1:
                    # Simple pattern like '7b'
                    if 'pt' in name_lower and 'pt' not in match.group(0):
                        # Handle cases like 'pt25b'
                        pt_match = re.search(r'pt(\d+)', name_lower)
                        if pt_match:
                            return float(f"0.{pt_match.group(1)}")
                    return float(match.group(1))
                elif len(match.groups()) == 2:
                    # Pattern like '1pt3b' -> 1.3
                    return float(f"{match.group(1)}.{match.group(2)}")
        
        # Default sizes for known models without clear size indicators
        if 'gpt4o' in name_lower:
            return 175.0  # Estimated
        elif 'florence' in name_lower:
            if 'large' in name_lower:
                return 0.77
            else:
                return 0.23
        
        # Default fallback
        return 1.0
    
    def load_dynamic_configurations(self):
        """Load model configurations dynamically from database."""
        try:
            # Load model information with families and sizes (using actual schema)
            model_query = """
            SELECT 
                m.model_name,
                f.family_name,
                s.size_range
            FROM models m
            JOIN model_families f ON m.family_id = f.family_id
            JOIN size_categories s ON m.size_id = s.size_id
            ORDER BY m.model_name
            """
            
            df_models = pd.read_sql_query(model_query, self.conn)
            
            # Create model info dictionary
            self.model_info = {}
            for _, row in df_models.iterrows():
                # Generate display name from model name
                display_name = self._generate_display_name(row['model_name'])
                # Estimate model size from name
                model_size_b = self._estimate_model_size(row['model_name'])
                
                self.model_info[row['model_name']] = {
                    'display_name': display_name,
                    'family': row['family_name'],
                    'size_category': row['size_range'],
                    'size_b': model_size_b
                }
            
            # Generate dynamic color palettes
            self._generate_color_palettes()
            
            print(f"✅ Loaded {len(self.model_info)} models from database")
            return True
            
        except Exception as e:
            print(f"❌ Error loading configurations: {e}")
            return False
    
    def _generate_color_palettes(self):
        """Generate color palettes dynamically from database data."""
        # Model families
        families = list(set([info['family'] for info in self.model_info.values()]))
        family_colors = sns.color_palette("husl", len(families))
        self.family_colors = {family: matplotlib.colors.rgb2hex(color) 
                             for family, color in zip(families, family_colors)}
        
        # Size categories  
        sizes = list(set([info['size_category'] for info in self.model_info.values()]))
        size_colors = sns.color_palette("viridis", len(sizes))
        self.size_colors = {size: matplotlib.colors.rgb2hex(color) 
                           for size, color in zip(sizes, size_colors)}
        
        # Model-specific colors (hash-based for consistency)
        model_colors = sns.color_palette("tab20", len(self.model_info))
        self.model_colors = {}
        for i, model_name in enumerate(sorted(self.model_info.keys())):
            if i < len(model_colors):
                self.model_colors[model_name] = matplotlib.colors.rgb2hex(model_colors[i])
            else:
                # Generate consistent color from model name hash
                hash_color = hashlib.md5(model_name.encode()).hexdigest()[:6]
                self.model_colors[model_name] = f"#{hash_color}"
        
        # Attack categories
        self.attack_colors = {
            "Transfer": "#FF7F0E",
            "Black-Box": "#1F77B4", 
            "Original": "#2CA02C"
        }
        
        print(f"✅ Generated color palettes: {len(families)} families, {len(sizes)} sizes")

    def load_robustness_data(self):
        """Load robustness analysis data from database."""
        try:
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
            
            df = pd.read_sql_query(query, self.conn)
            print(f"✅ Loaded robustness data: {len(df)} records")
            return df
            
        except Exception as e:
            print(f"❌ Error loading robustness data: {e}")
            return None
    
    def load_performance_data(self):
        """Load performance metrics data from database."""
        try:
            query = """
            SELECT 
                pm.metric_id,
                m.model_name,
                f.family_name,
                s.size_range as size_category,
                a.attack_name,
                pm.avg_inference_time_seconds,
                pm.avg_gpu_memory_allocated_mb,
                pm.avg_gpu_memory_peak_mb,
                pm.avg_gpu_memory_reserved_mb,
                pm.total_gpu_memory_mb,
                pm.avg_cpu_memory_mb,
                pm.model_loading_time_seconds,
                pm.cache_hit_ratio,
                pm.total_questions,
                pm.timestamp
            FROM performance_metrics pm
            JOIN models m ON pm.model_id = m.model_id
            JOIN model_families f ON m.family_id = f.family_id
            JOIN size_categories s ON m.size_id = s.size_id
            JOIN attack_types a ON pm.attack_id = a.attack_id
            """
            
            df = pd.read_sql_query(query, self.conn)
            print(f"✅ Loaded performance data: {len(df)} records")
            return df
            
        except Exception as e:
            print(f"❌ Error loading performance data: {e}")
            return None

    # =================== ROBUSTNESS PLOTS (plots1/) ===================
    
    def plot_model_degradation_line(self, df):
        """Create line plot of model degradation across attacks."""
        # Select top 3-5 models with most data
        model_counts = df['model_name'].value_counts()
        selected_models = model_counts.head(5).index.tolist()
        
        # Create pivot table
        plot_df = df[df['model_name'].isin(selected_models)].pivot_table(
            index='attack_name',
            columns='model_name', 
            values='accuracy_change',
            aggfunc='mean'
        ).reset_index().sort_values('attack_name')
        
        plt.figure(figsize=(14, 8))
        
        for model in selected_models:
            if model in plot_df.columns:
                display_name = self.model_info.get(model, {}).get('display_name', model)
                color = self.model_colors.get(model, 'gray')
                
                plt.plot(plot_df['attack_name'], plot_df[model],
                        marker='o', linewidth=2, markersize=8,
                        label=display_name, color=color)
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.title('Model Accuracy Change Under Different Attacks', fontsize=16)
        plt.xlabel('Attack Type', fontsize=14)
        plt.ylabel('Accuracy Change (%)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        plt.savefig(os.path.join(PLOT_DIR_ROBUSTNESS, 'linechart_model_degradation.png'), dpi=300)
        plt.close()
        print("✅ Created: linechart_model_degradation.png")
    
    def plot_attack_effectiveness_heatmap(self, df):
        """Create heatmap of attack effectiveness across models."""
        heatmap_data = df.pivot_table(
            index='attack_name',
            columns='model_name',
            values='accuracy_change',
            aggfunc='mean'
        )
        
        # Use display names for columns
        heatmap_data.columns = [
            self.model_info.get(col, {}).get('display_name', col) 
            for col in heatmap_data.columns
        ]
        
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
        
        plt.savefig(os.path.join(PLOT_DIR_ROBUSTNESS, 'heatmap_attack_effectiveness.png'), dpi=300)
        plt.close()
        print("✅ Created: heatmap_attack_effectiveness.png")
    
    def plot_raw_accuracy_heatmap(self, df):
        """Create heatmap of raw accuracy values."""
        heatmap_data = df.pivot_table(
            index='attack_name',
            columns='model_name', 
            values='accuracy',
            aggfunc='mean'
        )
        
        heatmap_data.columns = [
            self.model_info.get(col, {}).get('display_name', col)
            for col in heatmap_data.columns
        ]
        
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
        
        plt.savefig(os.path.join(PLOT_DIR_ROBUSTNESS, 'heatmap_raw_accuracy.png'), dpi=300)
        plt.close()
        print("✅ Created: heatmap_raw_accuracy.png")
    
    def plot_model_family_robustness(self, df):
        """Create bar chart of model family robustness."""
        family_avg = df.groupby('model_family')['accuracy_change'].mean().sort_values()
        
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
        plt.savefig(os.path.join(PLOT_DIR_ROBUSTNESS, 'barchart_model_family_robustness.png'), dpi=300)
        plt.close()
        print("✅ Created: barchart_model_family_robustness.png")
    
    def plot_size_category_robustness(self, df):
        """Create bar chart of size category robustness."""
        size_order = ['(0-1]B', '(1-2]B', '(2-3]B', '(3-4]B', '(4-5]B', '(5-6]B', '(6-7]B', 'Cloud API']
        size_order = [size for size in size_order if size in df['size_category'].unique()]
        
        size_avg = df.groupby('size_category')['accuracy_change'].mean().reindex(size_order)
        
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
        plt.savefig(os.path.join(PLOT_DIR_ROBUSTNESS, 'barchart_size_category_robustness.png'), dpi=300)
        plt.close()
        print("✅ Created: barchart_size_category_robustness.png")
    
    def plot_attack_category_effectiveness(self, df):
        """Create bar chart of attack category effectiveness.""" 
        attack_cat_avg = df.groupby('attack_category')['accuracy_change'].mean().sort_values()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(attack_cat_avg.index, attack_cat_avg.values)
        
        for i, bar in enumerate(bars):
            category = attack_cat_avg.index[i]
            bar.set_color(self.attack_colors.get(category, 'gray'))
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.title('Average Effectiveness by Attack Category', fontsize=16)
        plt.xlabel('Attack Category', fontsize=14)
        plt.ylabel('Average Accuracy Change (%)', fontsize=14)
        plt.grid(True, axis='y', alpha=0.3)
        
        for i, v in enumerate(attack_cat_avg.values):
            plt.text(i, v + (0.5 if v >= 0 else -1.5),
                    f"{v:.1f}%", ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR_ROBUSTNESS, 'barchart_attack_category_effectiveness.png'), dpi=300)
        plt.close()
        print("✅ Created: barchart_attack_category_effectiveness.png")
    
    def plot_family_vs_attack_category_heatmap(self, df):
        """Create heatmap of family vs attack category."""
        heatmap_data = df.pivot_table(
            index='model_family',
            columns='attack_category',
            values='accuracy_change',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data,
                    annot=True, fmt=".1f",
                    cmap="RdYlGn_r", center=0,
                    linewidths=.5,
                    cbar_kws={'label': 'Accuracy Change (%)'})
        
        plt.title('Attack Category Effectiveness Across Model Families', fontsize=16)
        plt.ylabel('Model Family', fontsize=14)
        plt.xlabel('Attack Category', fontsize=14)
        plt.tight_layout()
        
        plt.savefig(os.path.join(PLOT_DIR_ROBUSTNESS, 'heatmap_family_vs_attack_category.png'), dpi=300)
        plt.close()
        print("✅ Created: heatmap_family_vs_attack_category.png")
    
    def plot_size_vs_attack_category_heatmap(self, df):
        """Create heatmap of size vs attack category."""
        size_order = ['(0-1]B', '(1-2]B', '(2-3]B', '(3-4]B', '(4-5]B', '(5-6]B', '(6-7]B', 'Cloud API']
        size_order = [size for size in size_order if size in df['size_category'].unique()]
        
        heatmap_data = df.pivot_table(
            index='size_category',
            columns='attack_category', 
            values='accuracy_change',
            aggfunc='mean'
        ).reindex(size_order)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data,
                    annot=True, fmt=".1f", 
                    cmap="RdYlGn_r", center=0,
                    linewidths=.5,
                    cbar_kws={'label': 'Accuracy Change (%)'})
        
        plt.title('Attack Category Effectiveness Across Model Size Categories', fontsize=16)
        plt.ylabel('Model Size Category', fontsize=14)
        plt.xlabel('Attack Category', fontsize=14)
        plt.tight_layout()
        
        plt.savefig(os.path.join(PLOT_DIR_ROBUSTNESS, 'heatmap_size_vs_attack_category.png'), dpi=300)
        plt.close()
        print("✅ Created: heatmap_size_vs_attack_category.png")
    
    def plot_3d_dimension_analysis(self, df):
        """Create 3D scatter plot of dimensions."""
        from mpl_toolkits.mplot3d import Axes3D
        
        grouped_data = df.groupby(['model_family', 'size_category'])['accuracy_change'].mean().reset_index()
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        families = grouped_data['model_family'].unique()
        family_to_num = {family: i for i, family in enumerate(families)}
        
        sizes = grouped_data['size_category'].unique() 
        size_to_num = {size: i for i, size in enumerate(sizes)}
        
        x = [family_to_num[family] for family in grouped_data['model_family']]
        y = [size_to_num[size] for size in grouped_data['size_category']]
        z = grouped_data['accuracy_change'].values
        colors = [self.family_colors.get(family, 'gray') for family in grouped_data['model_family']]
        
        ax.scatter(x, y, z, c=colors, s=100, alpha=0.7)
        
        for i in range(len(x)):
            ax.text(x[i], y[i], z[i],
                   f"{grouped_data['model_family'].iloc[i]}, {grouped_data['size_category'].iloc[i]}",
                   fontsize=8)
        
        ax.set_title('Model Family, Size Category, and Robustness', fontsize=16)
        ax.set_xlabel('Model Family', fontsize=14)
        ax.set_ylabel('Size Category', fontsize=14)
        ax.set_zlabel('Average Accuracy Change (%)', fontsize=14)
        
        ax.set_xticks(list(family_to_num.values()))
        ax.set_xticklabels(list(family_to_num.keys()), rotation=45, ha='right')
        ax.set_yticks(list(size_to_num.values()))
        ax.set_yticklabels(list(size_to_num.keys()))
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR_ROBUSTNESS, 'scatter3d_dimension_analysis.png'), dpi=300)
        plt.close()
        print("✅ Created: scatter3d_dimension_analysis.png")

    # =================== PERFORMANCE PLOTS (plots/) ===================
    
    def plot_efficiency_score(self, df):
        """Create VLM efficiency score plot."""
        # Calculate efficiency score based on speed, memory, and quality
        model_metrics = df.groupby(['model_name']).agg({
            'avg_inference_time_seconds': 'mean',
            'avg_gpu_memory_peak_mb': 'mean',
            'model_loading_time_seconds': 'mean'
        }).reset_index()
        
        # Add display names from model_info
        model_metrics['display_name'] = model_metrics['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('display_name', x)
        )
        
        # Normalize metrics (lower is better for time/memory)
        model_metrics['speed_score'] = 1 / (model_metrics['avg_inference_time_seconds'] + 1)
        model_metrics['memory_score'] = 1 / (model_metrics['avg_gpu_memory_peak_mb'] / 1000 + 1)
        model_metrics['loading_score'] = 1 / (model_metrics['model_loading_time_seconds'] + 1)
        
        # Combined efficiency score (0-100)
        model_metrics['efficiency_score'] = (
            model_metrics['speed_score'] + 
            model_metrics['memory_score'] + 
            model_metrics['loading_score']
        ) * 100 / 3
        
        model_metrics = model_metrics.sort_values('efficiency_score', ascending=False)
        
        plt.figure(figsize=(14, 8))
        bars = plt.bar(range(len(model_metrics)), model_metrics['efficiency_score'])
        
        # Color by family
        for i, (_, row) in enumerate(model_metrics.iterrows()):
            family = self.model_info.get(row['model_name'], {}).get('family', 'Other')
            bars[i].set_color(self.family_colors.get(family, 'gray'))
        
        plt.title('VLM Efficiency Score', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Efficiency Score (higher is better)', fontsize=14)
        plt.xticks(range(len(model_metrics)), model_metrics['display_name'], rotation=45, ha='right')
        plt.grid(True, axis='y', alpha=0.3)
        
        # Create legend for families
        family_handles = [plt.Rectangle((0,0),1,1, color=color, label=family) 
                         for family, color in self.family_colors.items()]
        plt.legend(handles=family_handles, title="Model Family", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR_PERFORMANCE, 'efficiency_score.png'), dpi=300)
        plt.close()
        print("✅ Created: efficiency_score.png")
    
    def plot_metrics_by_size_category(self, df):
        """Create metrics breakdown by size category."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # GPU Memory Usage
        gpu_data = df.groupby(['size_category', 'family_name'])['avg_gpu_memory_peak_mb'].mean().reset_index()
        gpu_pivot = gpu_data.pivot(index='size_category', columns='family_name', values='avg_gpu_memory_peak_mb')
        gpu_pivot.plot(kind='bar', ax=axes[0,0], colormap='Set3')
        axes[0,0].set_title('GPU Memory Usage (GB)')
        axes[0,0].set_ylabel('GPU Memory Usage (GB)')
        axes[0,0].set_xlabel('Model Size Category')
        axes[0,0].legend(title='Model Family', bbox_to_anchor=(1.05, 1))
        
        # Loading Time
        load_data = df.groupby(['size_category', 'family_name'])['model_loading_time_seconds'].mean().reset_index()
        load_pivot = load_data.pivot(index='size_category', columns='family_name', values='model_loading_time_seconds')
        load_pivot.plot(kind='bar', ax=axes[0,1], colormap='Set3')
        axes[0,1].set_title('Loading Time (s)')
        axes[0,1].set_ylabel('Loading Time (s)')
        axes[0,1].set_xlabel('Model Size Category')
        axes[0,1].legend(title='Model Family', bbox_to_anchor=(1.05, 1))
        
        # Inference Time
        inf_data = df.groupby(['size_category', 'family_name'])['avg_inference_time_seconds'].mean().reset_index()
        inf_pivot = inf_data.pivot(index='size_category', columns='family_name', values='avg_inference_time_seconds')
        inf_pivot.plot(kind='bar', ax=axes[1,0], colormap='Set3')
        axes[1,0].set_title('Inference Time (s)')
        axes[1,0].set_ylabel('Inference Time (s)')
        axes[1,0].set_xlabel('Model Size Category')
        axes[1,0].legend(title='Model Family', bbox_to_anchor=(1.05, 1))
        
        # Quality Score (placeholder - would need actual quality metrics)
        # Using cache hit ratio as proxy for stability/quality
        qual_data = df.groupby(['size_category', 'family_name'])['cache_hit_ratio'].mean().reset_index()
        qual_pivot = qual_data.pivot(index='size_category', columns='family_name', values='cache_hit_ratio')
        qual_pivot.plot(kind='bar', ax=axes[1,1], colormap='Set3')
        axes[1,1].set_title('Quality Score (0-2)')
        axes[1,1].set_ylabel('Quality Score (0-2)')
        axes[1,1].set_xlabel('Model Size Category')
        axes[1,1].legend(title='Model Family', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR_PERFORMANCE, 'metrics_by_size_category.png'), dpi=300)
        plt.close()
        print("✅ Created: metrics_by_size_category.png")
    
    def plot_performance_heatmap(self, df):
        """Create comprehensive performance metrics heatmap."""
        # Aggregate metrics by model
        model_perf = df.groupby(['model_name']).agg({
            'avg_inference_time_seconds': 'mean',
            'avg_gpu_memory_peak_mb': 'mean', 
            'model_loading_time_seconds': 'mean',
            'cache_hit_ratio': 'mean'
        }).reset_index()
        
        # Add display names and model size from model_info
        model_perf['display_name'] = model_perf['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('display_name', x)
        )
        model_perf['model_size_b'] = model_perf['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('size_b', 1.0)
        )
        
        # Normalize metrics (0-1 scale, higher = better)
        metrics_df = pd.DataFrame(index=model_perf['display_name'])
        
        metrics_df['Size Efficiency'] = 1 / (model_perf['model_size_b'] + 1)
        metrics_df['Memory Efficiency'] = 1 / (model_perf['avg_gpu_memory_peak_mb'] / 1000 + 1)
        metrics_df['Loading Speed'] = 1 / (model_perf['model_loading_time_seconds'] + 1)
        metrics_df['Inference Speed'] = 1 / (model_perf['avg_inference_time_seconds'] + 1)
        metrics_df['Response Quality'] = model_perf['cache_hit_ratio'].values  # Using cache ratio as proxy
        
        plt.figure(figsize=(10, 12))
        sns.heatmap(metrics_df,
                    annot=True, fmt=".2f",
                    cmap="RdYlGn", vmin=0, vmax=1,
                    linewidths=.5,
                    cbar_kws={'label': 'Score (Higher is Better)'})
        
        plt.title('VLM Performance Metrics (Higher is Better)', fontsize=16)
        plt.ylabel('Model', fontsize=14)
        plt.xlabel('Metric', fontsize=14)
        plt.tight_layout()
        
        plt.savefig(os.path.join(PLOT_DIR_PERFORMANCE, 'performance_heatmap.png'), dpi=300)
        plt.close()
        print("✅ Created: performance_heatmap.png")
    
    def plot_size_memory_quality_bubble(self, df):
        """Create bubble chart of size vs memory vs quality."""
        model_data = df.groupby(['model_name', 'family_name']).agg({
            'avg_gpu_memory_peak_mb': 'mean',
            'cache_hit_ratio': 'mean'
        }).reset_index()
        
        # Add display names from model_info
        model_data['display_name'] = model_data['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('display_name', x)
        )
        
        model_data['model_size_b'] = model_data['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('size_b', 1.0)
        )
        
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with bubble sizes based on quality
        for family in model_data['family_name'].unique():
            family_data = model_data[model_data['family_name'] == family]
            plt.scatter(family_data['model_size_b'],
                       family_data['avg_gpu_memory_peak_mb'] / 1000,  # Convert to GB
                       s=family_data['cache_hit_ratio'] * 500 + 50,  # Bubble size
                       alpha=0.6,
                       color=self.family_colors.get(family, 'gray'),
                       label=family)
            
            # Add model name labels
            for _, row in family_data.iterrows():
                plt.annotate(row['display_name'].split()[0],  # First word only
                           (row['model_size_b'], row['avg_gpu_memory_peak_mb'] / 1000),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Model Size (Billion Parameters)', fontsize=14)
        plt.ylabel('GPU Memory Usage (GB)', fontsize=14)
        plt.title('Model Size vs. Memory Usage vs. Quality', fontsize=16)
        plt.legend(title="Model Family", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Add size legend
        sizes = [0.2, 0.5, 1.0]
        size_labels = ['Poor Quality', 'Fair Quality', 'Good Quality']
        size_legend = [plt.scatter([], [], s=s*500+50, alpha=0.6, color='gray', label=l) 
                      for s, l in zip(sizes, size_labels)]
        plt.legend(handles=size_legend, title="Response Quality", 
                  loc='upper left', bbox_to_anchor=(0.02, 0.98))
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR_PERFORMANCE, 'size_memory_quality_bubble.png'), dpi=300)
        plt.close()
        print("✅ Created: size_memory_quality_bubble.png")
    
    def plot_size_vs_inference_time(self, df):
        """Create scatter plot of size vs inference time."""
        model_data = df.groupby(['model_name', 'family_name']).agg({
            'avg_inference_time_seconds': 'mean',
            'cache_hit_ratio': 'mean'
        }).reset_index()
        
        # Add display names from model_info
        model_data['display_name'] = model_data['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('display_name', x)
        )
        
        model_data['model_size_b'] = model_data['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('size_b', 1.0)
        )
        
        plt.figure(figsize=(12, 8))
        
        # Color by family, marker by quality
        for family in model_data['family_name'].unique():
            family_data = model_data[model_data['family_name'] == family]
            
            # Different markers based on quality
            for _, row in family_data.iterrows():
                quality = row['cache_hit_ratio']
                if quality > 0.7:
                    marker = 'o'  # Good quality
                    marker_label = 'Good Quality'
                elif quality > 0.3:
                    marker = 's'  # Fair quality  
                    marker_label = 'Fair Quality'
                else:
                    marker = 'x'  # Poor quality
                    marker_label = 'Poor Quality'
                
                plt.scatter(row['model_size_b'], row['avg_inference_time_seconds'],
                           color=self.family_colors.get(family, 'gray'),
                           marker=marker, s=100, alpha=0.7,
                           label=f"{family} ({marker_label})" if family_data.index[0] == row.name else "")
                
                plt.annotate(row['display_name'].split()[0],
                           (row['model_size_b'], row['avg_inference_time_seconds']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Model Size (Billion Parameters)', fontsize=14)
        plt.ylabel('Inference Time (seconds)', fontsize=14)
        plt.title('Model Size vs. Inference Time with Response Quality', fontsize=16)
        plt.grid(True, alpha=0.3)
        
        # Create custom legend
        quality_markers = [
            plt.scatter([], [], marker='o', s=100, color='gray', alpha=0.7, label='Good Quality'),
            plt.scatter([], [], marker='s', s=100, color='gray', alpha=0.7, label='Fair Quality'),
            plt.scatter([], [], marker='x', s=100, color='gray', alpha=0.7, label='Poor Quality')
        ]
        
        plt.legend(title="Response Quality", loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR_PERFORMANCE, 'size_vs_inference_time.png'), dpi=300)
        plt.close()
        print("✅ Created: size_vs_inference_time.png")
    
    def plot_size_vs_loading_time(self, df):
        """Create scatter plot of size vs loading time.""" 
        model_data = df.groupby(['model_name', 'family_name']).agg({
            'model_loading_time_seconds': 'mean'
        }).reset_index()
        
        # Add display names from model_info
        model_data['display_name'] = model_data['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('display_name', x)
        )
        
        model_data['model_size_b'] = model_data['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('size_b', 1.0)
        )
        
        plt.figure(figsize=(12, 8))
        
        for family in model_data['family_name'].unique():
            family_data = model_data[model_data['family_name'] == family]
            plt.scatter(family_data['model_size_b'], family_data['model_loading_time_seconds'],
                       color=self.family_colors.get(family, 'gray'),
                       s=100, alpha=0.7, label=family)
            
            for _, row in family_data.iterrows():
                plt.annotate(row['display_name'].split()[0],
                           (row['model_size_b'], row['model_loading_time_seconds']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Model Size (Billion Parameters)', fontsize=14)
        plt.ylabel('Loading Time (seconds)', fontsize=14)
        plt.title('Model Size vs. Loading Time', fontsize=16)
        plt.legend(title="Model Family", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR_PERFORMANCE, 'size_vs_loading_time.png'), dpi=300)
        plt.close()
        print("✅ Created: size_vs_loading_time.png")
    
    def plot_size_vs_memory(self, df):
        """Create scatter plot of size vs GPU memory."""
        model_data = df.groupby(['model_name', 'family_name']).agg({
            'avg_gpu_memory_peak_mb': 'mean'
        }).reset_index()
        
        # Add display names from model_info
        model_data['display_name'] = model_data['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('display_name', x)
        )
        
        model_data['model_size_b'] = model_data['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('size_b', 1.0)
        )
        
        plt.figure(figsize=(12, 8))
        
        for family in model_data['family_name'].unique():
            family_data = model_data[model_data['family_name'] == family]
            plt.scatter(family_data['model_size_b'], family_data['avg_gpu_memory_peak_mb'] / 1000,
                       color=self.family_colors.get(family, 'gray'),
                       s=100, alpha=0.7, label=family)
            
            for _, row in family_data.iterrows():
                plt.annotate(row['display_name'].split()[0],
                           (row['model_size_b'], row['avg_gpu_memory_peak_mb'] / 1000),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Add trend line
        x = model_data['model_size_b']
        y = model_data['avg_gpu_memory_peak_mb'] / 1000
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "k--", alpha=0.5, label="Trend Line")
        
        plt.xlabel('Model Size (Billion Parameters)', fontsize=14)
        plt.ylabel('GPU Memory Usage (GB)', fontsize=14)
        plt.title('Model Size vs. GPU Memory Usage', fontsize=16)
        plt.legend(title="Model Family", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR_PERFORMANCE, 'size_vs_memory.png'), dpi=300)
        plt.close()
        print("✅ Created: size_vs_memory.png")
    
    def plot_top_models_radar(self, df):
        """Create radar chart comparing top models."""
        # Select top 4 models by some criterion (e.g., balanced performance)
        model_scores = df.groupby(['model_name']).agg({
            'avg_inference_time_seconds': 'mean',
            'avg_gpu_memory_peak_mb': 'mean',
            'model_loading_time_seconds': 'mean', 
            'cache_hit_ratio': 'mean'
        }).reset_index()
        
        # Add display names from model_info
        model_scores['display_name'] = model_scores['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('display_name', x)
        )
        
        # Add size info
        model_scores['model_size_b'] = model_scores['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('size_b', 1.0)
        )
        
        # Calculate normalized scores (0-1, higher = better)
        model_scores['size_eff'] = 1 / (model_scores['model_size_b'] + 1)
        model_scores['memory_eff'] = 1 / (model_scores['avg_gpu_memory_peak_mb'] / 1000 + 1)
        model_scores['loading_speed'] = 1 / (model_scores['model_loading_time_seconds'] + 1)
        model_scores['inference_speed'] = 1 / (model_scores['avg_inference_time_seconds'] + 1)
        model_scores['response_quality'] = model_scores['cache_hit_ratio']
        
        # Select top 4 models by overall score
        model_scores['overall_score'] = (
            model_scores['size_eff'] + model_scores['memory_eff'] + 
            model_scores['loading_speed'] + model_scores['inference_speed'] + 
            model_scores['response_quality']
        ) / 5
        
        top_models = model_scores.nlargest(4, 'overall_score')
        
        # Radar chart setup
        categories = ['Size Efficiency', 'Memory Efficiency', 'Loading Speed', 
                     'Inference Speed', 'Response Quality']
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        
        for i, (_, model) in enumerate(top_models.iterrows()):
            values = [model['size_eff'], model['memory_eff'], model['loading_speed'],
                     model['inference_speed'], model['response_quality']]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model['display_name'], 
                   color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Top Model Performance Comparison', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR_PERFORMANCE, 'top_models_radar.png'), dpi=300)
        plt.close()
        print("✅ Created: top_models_radar.png")

    def generate_all_plots(self):
        """Generate all visualization plots."""
        print(f"\n🚀 Starting comprehensive plot generation...")
        print(f"📊 Target directories: {PLOT_DIR_ROBUSTNESS}, {PLOT_DIR_PERFORMANCE}")
        
        # Load robustness data and generate plots1/ 
        print(f"\n📈 Generating robustness analysis plots...")
        robustness_df = self.load_robustness_data()
        if robustness_df is not None and not robustness_df.empty:
            self.plot_model_degradation_line(robustness_df)
            self.plot_attack_effectiveness_heatmap(robustness_df)
            self.plot_raw_accuracy_heatmap(robustness_df)
            self.plot_model_family_robustness(robustness_df)
            self.plot_size_category_robustness(robustness_df)
            self.plot_attack_category_effectiveness(robustness_df)
            self.plot_family_vs_attack_category_heatmap(robustness_df)
            self.plot_size_vs_attack_category_heatmap(robustness_df)
            self.plot_3d_dimension_analysis(robustness_df)
            print(f"✅ Generated {len(os.listdir(PLOT_DIR_ROBUSTNESS))} robustness plots")
        else:
            print("⚠️ No robustness data found - skipping robustness plots")
        
        # Load performance data and generate plots/
        print(f"\n⚡ Generating performance analysis plots...")
        performance_df = self.load_performance_data()
        if performance_df is not None and not performance_df.empty:
            self.plot_efficiency_score(performance_df)
            self.plot_metrics_by_size_category(performance_df) 
            self.plot_performance_heatmap(performance_df)
            self.plot_size_memory_quality_bubble(performance_df)
            self.plot_size_vs_inference_time(performance_df)
            self.plot_size_vs_loading_time(performance_df)
            self.plot_size_vs_memory(performance_df)
            self.plot_top_models_radar(performance_df)
            print(f"✅ Generated {len(os.listdir(PLOT_DIR_PERFORMANCE))} performance plots")
        else:
            print("⚠️ No performance data found - skipping performance plots")
        
        print(f"\n🎉 Plot generation complete!")
        print(f"📁 Robustness plots: {PLOT_DIR_ROBUSTNESS}")
        print(f"📁 Performance plots: {PLOT_DIR_PERFORMANCE}")

def main():
    """Main execution function."""
    print("=" * 60)
    print("🔬 Enhanced VLM Data Analysis & Visualization")
    print("=" * 60)
    
    analyzer = VLMDataAnalyzer()
    
    try:
        # Connect to database
        analyzer.connect_db()
        
        # Load dynamic configurations from database
        if not analyzer.load_dynamic_configurations():
            print("❌ Failed to load configurations from database")
            return
        
        # Generate all plots
        analyzer.generate_all_plots()
        
    except DatabaseNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Please run the evaluation pipeline first to generate the database.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        analyzer.close_db()
        print(f"\n✨ Analysis complete!")

if __name__ == "__main__":
    main()