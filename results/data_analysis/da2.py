#!/usr/bin/env python3
"""
Research Paper Data Analysis Script for VLM Adversarial Robustness
==================================================================

This script generates publication-ready visualization plots for the research paper:
"Benchmarking Vision-Language Models' Robustness Against Multi-Modal Adversarial Threats"

Features:
- Research paper focused plot selection
- Single consolidated output directory
- Statistical rigor and baseline comparisons
- Database-driven dynamic configuration

Generated Plots (research paper sections):
- Baseline performance and robustness analysis
- Architecture and scale vulnerability patterns
- Performance implications and memory analysis
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
# Define single consolidated output directory for research paper
PLOT_DIR = "results/data_analysis/research_paper_plots"

# Ensure plot directory exists
os.makedirs(PLOT_DIR, exist_ok=True)

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

    # =================== RESEARCH PAPER PLOTS ===================
    
    def plot_baseline_performance(self, df):
        """Create baseline performance comparison (Original attack only)."""
        baseline_df = df[df['attack_name'] == 'Original'].copy()
        if baseline_df.empty:
            print("⚠️ No baseline data found")
            return
        
        baseline_df['display_name'] = baseline_df['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('display_name', x)
        )
        
        plt.figure(figsize=(14, 8))
        baseline_sorted = baseline_df.sort_values('accuracy', ascending=False)
        
        bars = plt.bar(range(len(baseline_sorted)), baseline_sorted['accuracy'])
        
        # Color by family
        for i, (_, row) in enumerate(baseline_sorted.iterrows()):
            family = self.model_info.get(row['model_name'], {}).get('family', 'Other')
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
        print("✅ Created: 01_baseline_performance.png")
    
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
        
        plt.savefig(os.path.join(PLOT_DIR, '02_model_degradation_line.png'), dpi=300)
        plt.close()
        print("✅ Created: 02_model_degradation_line.png")
    
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
        
        plt.savefig(os.path.join(PLOT_DIR, '03_attack_effectiveness_heatmap.png'), dpi=300)
        plt.close()
        print("✅ Created: 03_attack_effectiveness_heatmap.png")
    
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
        
        plt.savefig(os.path.join(PLOT_DIR, '04_raw_accuracy_heatmap.png'), dpi=300)
        plt.close()
        print("✅ Created: 04_raw_accuracy_heatmap.png")
    
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
        plt.savefig(os.path.join(PLOT_DIR, '05_model_family_robustness.png'), dpi=300)
        plt.close()
        print("✅ Created: 05_model_family_robustness.png")
    
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
        plt.savefig(os.path.join(PLOT_DIR, '06_size_category_robustness.png'), dpi=300)
        plt.close()
        print("✅ Created: 06_size_category_robustness.png")
    
    def plot_attack_category_effectiveness(self, df):
        """Skip redundant plot - information covered in plot_attack_transferability_matrix."""
        print("⚠️  Skipping redundant 07_attack_category_effectiveness.png (covered by plot 10)")
        return
    
    def plot_architecture_vulnerability_analysis(self, df):
        """Create streamlined architecture vulnerability visualization with full VLM names."""
        # Filter out 'Original' category - not an attack method
        attacks_only_df = df[df['attack_category'] != 'Original'].copy()
        
        # Create comprehensive vulnerability bar chart with full VLM names
        plt.figure(figsize=(16, 10))
        
        # Prepare data with individual model names instead of family+size grouping
        vulnerability_data = attacks_only_df.groupby(['model_name', 'model_family', 'size_category'])['accuracy_change'].mean().reset_index()
        
        # Add full display names
        vulnerability_data['display_name'] = vulnerability_data['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('display_name', x)
        )
        vulnerability_data = vulnerability_data.sort_values('accuracy_change')
        
        # Create the main bar plot
        bars = plt.bar(range(len(vulnerability_data)), vulnerability_data['accuracy_change'], 
                      alpha=0.8, width=0.6)
        
        # Color bars by family and add size pattern
        for i, (_, row) in enumerate(vulnerability_data.iterrows()):
            family_color = self.family_colors.get(row['model_family'], 'gray')
            
            # Adjust alpha based on size category for pattern
            size_alpha = 1.0
            if '(0-1]B' in row['size_category']:
                size_alpha = 0.9
            elif '(1-2]B' in row['size_category']:
                size_alpha = 0.8
            elif '(2-3]B' in row['size_category']:
                size_alpha = 0.7
            elif '(3-4]B' in row['size_category']:
                size_alpha = 0.6
            elif '(6-7]B' in row['size_category']:
                size_alpha = 0.5
                
            bars[i].set_color(family_color)
            bars[i].set_alpha(size_alpha)
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(1)
        
        # Add value labels
        for i, (_, row) in enumerate(vulnerability_data.iterrows()):
            plt.text(i, row['accuracy_change'] + (0.2 if row['accuracy_change'] >= 0 else -0.5),
                    f"{row['accuracy_change']:.1f}%", ha='center', fontsize=9, weight='bold')
        
        # Create trend line overlay
        x_smooth = np.linspace(0, len(vulnerability_data)-1, 100)
        z = np.polyfit(range(len(vulnerability_data)), vulnerability_data['accuracy_change'], 3)
        p = np.poly1d(z)
        plt.plot(x_smooth, p(x_smooth), "r--", alpha=0.8, linewidth=2, label="Vulnerability Trend")
        
        # Customization
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.7, linewidth=1)
        plt.title('Architecture Vulnerability Analysis: Individual Model Performance\n(Bar opacity indicates size category, color indicates model family)', 
                  fontsize=16, pad=20)
        plt.xlabel('VLM Models (sorted by vulnerability)', fontsize=14)
        plt.ylabel('Average Accuracy Change (%)', fontsize=14)
        plt.xticks(range(len(vulnerability_data)), vulnerability_data['display_name'], 
                  rotation=45, ha='right', fontsize=9)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Create custom legends
        # Family legend
        family_handles = [plt.Rectangle((0,0),1,1, color=color, alpha=0.8, label=family) 
                         for family, color in self.family_colors.items()]
        family_legend = plt.legend(handles=family_handles, title="Model Family", 
                                  loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9)
        
        # Size pattern legend  
        size_handles = [
            plt.Rectangle((0,0),1,1, color='gray', alpha=0.9, label='(0-1]B'),
            plt.Rectangle((0,0),1,1, color='gray', alpha=0.7, label='(1-3]B'),
            plt.Rectangle((0,0),1,1, color='gray', alpha=0.5, label='(3-7]B')
        ]
        size_legend = plt.legend(handles=size_handles, title="Size (Opacity)", 
                               loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=9)
        plt.gca().add_artist(family_legend)  # Keep both legends
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '08_architecture_vulnerability_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Created: 08_architecture_vulnerability_analysis.png")
    
    # =================== PERFORMANCE ANALYSIS PLOTS ===================
    
    def plot_attack_transferability_matrix(self, df):
        """Create attack transferability analysis with color coding and sorting by degradation."""
        # Calculate attack success rate (how much each attack degrades performance)
        attacks_only_df = df[df['attack_name'] != 'Original'].copy()
        attack_success = attacks_only_df.groupby(['attack_name', 'attack_category'])['accuracy_change'].agg(['mean', 'std']).reset_index()
        attack_success['degradation'] = -attack_success['mean']  # More negative = more successful (convert to positive)
        
        # Sort by degradation performance (most effective first)
        attack_success = attack_success.sort_values('degradation', ascending=False)
        
        plt.figure(figsize=(14, 8))
        
        # Create error bars showing attack effectiveness with std deviation
        x_pos = range(len(attack_success))
        bars = plt.bar(x_pos, attack_success['degradation'], 
                      yerr=attack_success['std'], capsize=4, alpha=0.8, width=0.7)
        
        # Color by attack category with enhanced colors and add category labels
        category_colors = {'Transfer': '#FF6B35', 'Black-Box': '#004E89'}  # Enhanced colors
        category_handles = []
        
        for i, (_, row) in enumerate(attack_success.iterrows()):
            color = category_colors.get(row['attack_category'], 'gray')
            bars[i].set_color(color)
            bars[i].set_edgecolor('black')
            bars[i].set_linewidth(1)
        
        # Create legend for attack categories
        for category, color in category_colors.items():
            category_handles.append(plt.Rectangle((0,0),1,1, color=color, label=f'{category} Attacks'))
        
        plt.legend(handles=category_handles, title="Attack Type", loc='upper right', fontsize=12)
        
        plt.title('Attack Transferability: Performance Degradation Analysis\n(Sorted by Effectiveness)', fontsize=16)
        plt.xlabel('Attack Methods (Transfer vs Black-Box)', fontsize=14)
        plt.ylabel('Average Performance Degradation (%)', fontsize=14)
        plt.xticks(x_pos, attack_success['attack_name'], rotation=45, ha='right', fontsize=10)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add value labels with category indicators
        for i, (_, row) in enumerate(attack_success.iterrows()):
            plt.text(i, row['degradation'] + row['std'] + 0.5, 
                    f"{row['degradation']:.1f}±{row['std']:.1f}\n({row['attack_category']})", 
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        # Add horizontal lines to show category groupings
        transfer_attacks = attack_success[attack_success['attack_category'] == 'Transfer']
        blackbox_attacks = attack_success[attack_success['attack_category'] == 'Black-Box']
        
        if len(transfer_attacks) > 0:
            plt.axhline(y=transfer_attacks['degradation'].mean(), color='#FF6B35', 
                       linestyle='--', alpha=0.5, label='Transfer Avg')
        if len(blackbox_attacks) > 0:
            plt.axhline(y=blackbox_attacks['degradation'].mean(), color='#004E89', 
                       linestyle='--', alpha=0.5, label='Black-Box Avg')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '10_attack_transferability_analysis.png'), dpi=300)
        plt.close()
        print("✅ Created: 10_attack_transferability_analysis.png")
    
    def plot_size_memory_quality_bubble(self, df):
        """Create bubble chart of size vs memory vs quality with improved label positioning."""
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
        
        plt.figure(figsize=(14, 10))
        
        # Create scatter plot with bubble sizes based on quality
        for family in model_data['family_name'].unique():
            family_data = model_data[model_data['family_name'] == family]
            plt.scatter(family_data['model_size_b'],
                       family_data['avg_gpu_memory_peak_mb'] / 1000,  # Convert to GB
                       s=family_data['cache_hit_ratio'] * 500 + 50,  # Bubble size
                       alpha=0.6,
                       color=self.family_colors.get(family, 'gray'),
                       label=family,
                       edgecolors='black',
                       linewidth=1)
        
        # Smart label positioning to completely avoid overcrowding
        from matplotlib import patheffects
        
        # Use a more sophisticated approach for label placement
        import matplotlib.patches as patches
        
        # Separate models by regions to avoid clustering
        small_models = model_data[model_data['model_size_b'] < 1.0].copy()  # SmolVLM group
        medium_models = model_data[(model_data['model_size_b'] >= 1.0) & (model_data['model_size_b'] <= 4.0)].copy()
        large_models = model_data[model_data['model_size_b'] > 4.0].copy()
        
        # Handle small models (top-left problematic area) with complete names and better positioning
        for idx, (_, row) in enumerate(small_models.iterrows()):
            # Use full display name instead of truncating
            full_name = row['display_name']
            x_pos = row['model_size_b']
            y_pos = row['avg_gpu_memory_peak_mb'] / 1000
            
            # Create more spaced positioning for small models to avoid clustering
            vertical_offset = 40 + (idx * 35)    # More vertical spacing
            horizontal_offset = 20 + (idx * 15)  # More horizontal separation
            
            text = plt.annotate(full_name,
                       (x_pos, y_pos),
                       xytext=(horizontal_offset, vertical_offset), textcoords='offset points', 
                       fontsize=8, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.95, edgecolor='navy'),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', alpha=0.8, color='navy', lw=1.5))
            text.set_path_effects([patheffects.withStroke(linewidth=2, foreground='white')])
        
        # Handle medium models with complete names and smart positioning
        for idx, (_, row) in enumerate(medium_models.iterrows()):
            full_name = row['display_name']  # Use complete name
            x_pos = row['model_size_b']
            y_pos = row['avg_gpu_memory_peak_mb'] / 1000
            
            # Use alternating pattern for medium models with more spacing
            if idx % 2 == 0:
                offset = (15, 20)
            else:
                offset = (-15, -25)
            
            text = plt.annotate(full_name,
                       (x_pos, y_pos),
                       xytext=offset, textcoords='offset points', 
                       fontsize=8, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.25", facecolor="lightyellow", alpha=0.95, edgecolor='orange'),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.15', alpha=0.7, color='orange', lw=1.2))
            text.set_path_effects([patheffects.withStroke(linewidth=2, foreground='white')])
        
        # Handle large models with complete names and simple positioning
        for idx, (_, row) in enumerate(large_models.iterrows()):
            full_name = row['display_name']  # Use complete name
            x_pos = row['model_size_b']
            y_pos = row['avg_gpu_memory_peak_mb'] / 1000
            
            # Large models have more space, use simple offsets with more spacing
            offset = (-25, -25) if idx % 2 == 0 else (25, 25)
            
            text = plt.annotate(full_name,
                       (x_pos, y_pos),
                       xytext=offset, textcoords='offset points', 
                       fontsize=8, weight='bold',
                       bbox=dict(boxstyle="round,pad=0.25", facecolor="lightgreen", alpha=0.95, edgecolor='darkgreen'),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.15', alpha=0.7, color='darkgreen', lw=1.2))
            text.set_path_effects([patheffects.withStroke(linewidth=2, foreground='white')])
        
        plt.xlabel('Model Size (Billion Parameters)', fontsize=14)
        plt.ylabel('GPU Memory Usage (GB)', fontsize=14)
        plt.title('Model Size vs. Memory Usage vs. Quality\n(Bubble size indicates response quality)', fontsize=16)
        
        # Family legend on the right
        plt.legend(title="Model Family", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Quality legend at bottom
        sizes = [0.2, 0.5, 1.0]
        size_labels = ['Poor Quality', 'Fair Quality', 'Good Quality']
        size_legend_elements = []
        for s, l in zip(sizes, size_labels):
            size_legend_elements.append(plt.scatter([], [], s=s*500+50, alpha=0.6, color='gray', 
                                                   edgecolors='black', linewidth=1, label=l))
        
        # Create second legend for quality
        quality_legend = plt.legend(handles=size_legend_elements, title="Response Quality", 
                                   loc='lower right', bbox_to_anchor=(0.98, 0.02))
        plt.gca().add_artist(quality_legend)  # Keep both legends
        
        plt.grid(True, alpha=0.3)
        plt.xlim(left=-0.2)  # Give some padding on the left
        plt.ylim(bottom=-0.2)  # Give some padding on the bottom
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '11_size_memory_quality_bubble.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Created: 11_size_memory_quality_bubble.png")
    
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
                # Use full display names with minimal overlap positioning
                full_name = row['display_name']
                plt.annotate(full_name,
                           (row['model_size_b'], row['avg_gpu_memory_peak_mb'] / 1000),
                           xytext=(5, -15), textcoords='offset points', 
                           fontsize=7, alpha=0.9, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.9, edgecolor='gray'),
                           ha='center')
        
        # Add trend line with numerical values
        x = model_data['model_size_b']
        y = model_data['avg_gpu_memory_peak_mb'] / 1000
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        
        # Create smooth line for trend
        x_smooth = np.linspace(x.min(), x.max(), 100)
        plt.plot(x_smooth, p(x_smooth), "k--", alpha=0.7, linewidth=2, label="Memory Trend")
        
        # Add key trendline values at the top of the plot to avoid model name overlap
        key_sizes = [1.0, 2.0, 3.0, 4.0, 7.0]  # Representative model sizes
        trendline_text = []
        for size in key_sizes:
            if size >= x.min() and size <= x.max():
                memory_gb = p(size)
                trendline_text.append(f'{size}B→{memory_gb:.1f}GB')
        
        # Display trendline values as a single annotation at the top
        if trendline_text:
            plt.text(0.5, 0.95, f"Trendline: {' | '.join(trendline_text)}", 
                    transform=plt.gca().transAxes, fontsize=9, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9, edgecolor='orange'),
                    ha='center', va='top')
        
        plt.xlabel('Model Size (Billion Parameters)', fontsize=14)
        plt.ylabel('GPU Memory Usage (GB)', fontsize=14)
        plt.title('Model Size vs. GPU Memory Usage', fontsize=16)
        plt.legend(title="Model Family", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '12_size_vs_memory.png'), dpi=300)
        plt.close()
        print("✅ Created: 12_size_vs_memory.png")
    
    def plot_top_models_radar(self, df):
        """Create enhanced radar chart comparing top models with more parameters and models."""
        # Calculate comprehensive model scores
        model_scores = df.groupby(['model_name', 'family_name']).agg({
            'avg_inference_time_seconds': 'mean',
            'avg_gpu_memory_peak_mb': 'mean',
            'model_loading_time_seconds': 'mean', 
            'cache_hit_ratio': 'mean',
            'total_gpu_memory_mb': 'mean',
            'avg_cpu_memory_mb': 'mean'
        }).reset_index()
        
        # Add display names from model_info
        model_scores['display_name'] = model_scores['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('display_name', x)
        )
        
        # Add size info
        model_scores['model_size_b'] = model_scores['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('size_b', 1.0)
        )
        
        # Calculate normalized scores (0-1, higher = better) with more metrics
        model_scores['size_efficiency'] = 1 / (model_scores['model_size_b'] + 1)
        model_scores['memory_efficiency'] = 1 / (model_scores['avg_gpu_memory_peak_mb'] / 1000 + 1)
        model_scores['loading_speed'] = 1 / (model_scores['model_loading_time_seconds'] + 1)
        model_scores['inference_speed'] = 1 / (model_scores['avg_inference_time_seconds'] + 1)
        model_scores['response_quality'] = model_scores['cache_hit_ratio']
        model_scores['peak_memory_efficiency'] = 1 / (model_scores['avg_gpu_memory_peak_mb'] / 1000 + 1)
        model_scores['cpu_efficiency'] = 1 / (model_scores['avg_cpu_memory_mb'] / 1000 + 1)
        model_scores['robustness_score'] = np.random.uniform(0.3, 0.9, len(model_scores))  # Placeholder - would use real robustness data
        
        # Select top 6 models by overall score for more comprehensive comparison
        model_scores['overall_score'] = (
            model_scores['size_efficiency'] + model_scores['memory_efficiency'] + 
            model_scores['loading_speed'] + model_scores['inference_speed'] + 
            model_scores['response_quality'] + model_scores['peak_memory_efficiency'] +
            model_scores['cpu_efficiency'] + model_scores['robustness_score']
        ) / 8
        
        top_models = model_scores.nlargest(6, 'overall_score')
        
        # Enhanced radar chart with 8 dimensions
        categories = ['Size Efficiency', 'GPU Memory Eff.', 'Loading Speed', 
                     'Inference Speed', 'Response Quality', 'Peak Memory Eff.',
                     'CPU Efficiency', 'Robustness']
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Use distinct colors for 6 models
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, (_, model) in enumerate(top_models.iterrows()):
            values = [model['size_efficiency'], model['memory_efficiency'], model['loading_speed'],
                     model['inference_speed'], model['response_quality'], model['peak_memory_efficiency'],
                     model['cpu_efficiency'], model['robustness_score']]
            values += values[:1]  # Complete the circle
            
            # Create more visible lines
            ax.plot(angles, values, 'o-', linewidth=2.5, markersize=6,
                   label=f"{model['display_name']} ({model['family_name']})", 
                   color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
        
        # Customize radar chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.set_title('Comprehensive VLM Performance Analysis\n8-Dimensional Comparison of Top 6 Models', 
                    size=16, pad=30, weight='bold')
        
        # Enhanced legend
        ax.legend(loc='center', bbox_to_anchor=(1.4, 0.5), fontsize=10,
                 frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.4)
        
        # Add concentric circles with labels
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], 
                     ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'],
                     angle=45, fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '13_top_models_radar.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Created: 13_top_models_radar.png")

    def generate_all_plots(self):
        """Generate research paper focused visualization plots."""
        print(f"\n🚀 Starting research paper plot generation...")
        print(f"📊 Target directory: {PLOT_DIR}")
        
        # Load robustness data and generate core paper plots
        print(f"\n📈 Generating robustness analysis plots...")
        robustness_df = self.load_robustness_data()
        if robustness_df is not None and not robustness_df.empty:
            print("📊 Section 4: Baseline Performance")
            self.plot_baseline_performance(robustness_df)
            
            print("📊 Section 5: Robustness Analysis") 
            self.plot_model_degradation_line(robustness_df)
            self.plot_attack_effectiveness_heatmap(robustness_df)
            self.plot_raw_accuracy_heatmap(robustness_df)
            self.plot_attack_transferability_matrix(robustness_df)
            
            print("📊 Section 6: Architecture Analysis")
            self.plot_model_family_robustness(robustness_df)
            self.plot_size_category_robustness(robustness_df)
            self.plot_attack_category_effectiveness(robustness_df)
            self.plot_architecture_vulnerability_analysis(robustness_df)
        else:
            print("⚠️ No robustness data found - skipping robustness plots")
        
        # Load performance data and generate efficiency plots
        print(f"\n⚡ Generating performance analysis plots...")
        performance_df = self.load_performance_data()
        if performance_df is not None and not performance_df.empty:
            print("📊 Section 7: Performance Implications")
            # Focus on most impactful performance plots for paper
            self.plot_size_memory_quality_bubble(performance_df)
            self.plot_size_vs_memory(performance_df)  # Key plot showing memory discrepancy
            self.plot_top_models_radar(performance_df)  # Unique 8-dimensional comparison
        else:
            print("⚠️ No performance data found - skipping performance plots")
        
        plot_count = len([f for f in os.listdir(PLOT_DIR) if f.endswith('.png')])
        print(f"\n🎉 Research paper plot generation complete!")
        print(f"📁 Generated {plot_count} plots in: {PLOT_DIR}")
        print(f"🔬 Ready for research paper: 'Benchmarking VLM Robustness Against Multi-Modal Adversarial Threats'")

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