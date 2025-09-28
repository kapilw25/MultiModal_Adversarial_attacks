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
PLOT_DIR = "results/research_plots"

# Ensure plot directory exists
os.makedirs(PLOT_DIR, exist_ok=True)

class DatabaseNotFoundError(Exception):
    """Custom exception for missing database."""
    pass

# GLOBAL AUTO-SCALING FUNCTIONS
def global_cluster_aware_scaling(vals, target_range):
    """Cluster-aware scaling that compresses empty gaps and expands within clusters"""
    from scipy import stats, interpolate
    from sklearn.cluster import KMeans
    import numpy as np
    
    # Detect clusters in the data
    vals_reshaped = vals.reshape(-1, 1)
    n_clusters = min(4, len(vals))  # Max 4 clusters, but not more than data points
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(vals_reshaped)
    cluster_centers = kmeans.cluster_centers_.flatten()
    
    # Sort clusters by center value
    cluster_order = np.argsort(cluster_centers)
    
    # Create transformation mapping
    val_min, val_max = vals.min(), vals.max()
    target_min, target_max = target_range
    
    # Allocate space: 80% for clusters, 20% for gaps
    cluster_space = (target_max - target_min) * 0.8
    gap_space = (target_max - target_min) * 0.2
    
    space_per_cluster = cluster_space / n_clusters
    space_per_gap = gap_space / max(1, n_clusters - 1)
    
    # Build transformation function
    val_test = np.linspace(val_min, val_max, 500)
    val_transformed_test = np.zeros_like(val_test)
    
    current_target = target_min
    
    for i, cluster_idx in enumerate(cluster_order):
        # Find data points in this cluster
        cluster_mask = cluster_labels == cluster_idx
        cluster_vals = vals[cluster_mask]
        
        if len(cluster_vals) == 0:
            continue
            
        cluster_min, cluster_max = cluster_vals.min(), cluster_vals.max()
        
        # Map this cluster's range to allocated space
        cluster_start = current_target
        cluster_end = current_target + space_per_cluster
        
        # Transform values in this cluster's range
        in_cluster = (val_test >= cluster_min) & (val_test <= cluster_max)
        if cluster_max > cluster_min:
            val_transformed_test[in_cluster] = cluster_start + (
                (val_test[in_cluster] - cluster_min) / (cluster_max - cluster_min)
            ) * space_per_cluster
        else:
            val_transformed_test[in_cluster] = cluster_start + space_per_cluster / 2
        
        current_target = cluster_end + space_per_gap
    
    # Handle gaps between clusters with linear interpolation
    for i in range(len(val_test) - 1):
        if val_transformed_test[i] == 0 and val_transformed_test[i+1] == 0:
            # Find nearest non-zero values
            prev_idx = i
            while prev_idx >= 0 and val_transformed_test[prev_idx] == 0:
                prev_idx -= 1
            next_idx = i + 1
            while next_idx < len(val_transformed_test) and val_transformed_test[next_idx] == 0:
                next_idx += 1
            
            if prev_idx >= 0 and next_idx < len(val_transformed_test):
                # Linear interpolation
                alpha = (i - prev_idx) / (next_idx - prev_idx)
                val_transformed_test[i] = (1 - alpha) * val_transformed_test[prev_idx] + alpha * val_transformed_test[next_idx]
    
    # Create interpolation function
    transform_func = interpolate.interp1d(val_test, val_transformed_test, kind='linear',
                                        bounds_error=False, fill_value='extrapolate')
    vals_scaled = transform_func(vals)
    
    return vals_scaled, transform_func

def global_density_based_scaling(vals, target_range, expansion_strength):
    """Density-based scaling for sharp distributions"""
    from scipy import stats, interpolate
    import numpy as np
    
    density = stats.gaussian_kde(vals, bw_method='scott')
    val_min, val_max = vals.min(), vals.max()
    val_test = np.linspace(val_min, val_max, 300)
    density_vals = density(val_test)
    
    density_norm = (density_vals - density_vals.min()) / (density_vals.max() - density_vals.min() + 1e-10)
    expansion_factors = 1.0 + density_norm * expansion_strength
    
    dval = np.diff(val_test)
    dval = np.append(dval, dval[-1])
    stretched_intervals = dval * expansion_factors
    val_transformed_test = np.cumsum(stretched_intervals)
    
    target_min, target_max = target_range
    val_transformed_test = target_min + (val_transformed_test / val_transformed_test[-1]) * (target_max - target_min)
    
    transform_func = interpolate.interp1d(val_test, val_transformed_test, kind='cubic',
                                        bounds_error=False, fill_value='extrapolate')
    vals_scaled = transform_func(vals)
    
    return vals_scaled, transform_func

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
        # Map internal model names to clean research paper model names (no organization prefixes)
        # NOTE: Database stores names with exact case (e.g., "Qwen25_VL_3B")
        name_mappings = {
            # Qwen models - Clean model names
            'Qwen25_VL_3B': 'Qwen2.5-VL-3B-Instruct',
            'Qwen25_VL_7B': 'Qwen2.5-VL-7B-Instruct',
            'Qwen2_VL_2B': 'Qwen2-VL-2B-Instruct',
            
            # Google models - Clean model names  
            'Gemma3_VL_4B': 'Gemma-3-4B-IT',
            'PaliGemma_VL_3B': 'PaliGemma-3B-Mix-224',
            
            # DeepSeek models - Clean model names
            'DeepSeek1_VL_1pt3B': 'DeepSeek-VL-1.3B-Chat',
            'DeepSeek1_VL_7B': 'DeepSeek-VL-7B-Chat',
            
            # SmolVLM2 models - Clean model names
            'SmolVLM2_pt25B': 'SmolVLM2-256M-Video-Instruct',
            'SmolVLM2_pt5B': 'SmolVLM2-500M-Video-Instruct',
            'SmolVLM2_2pt2B': 'SmolVLM2-2.2B-Instruct',
            
            # InternVL models - Clean model names
            'InternVL3_1B': 'InternVL3-1B',
            'InternVL3_2B': 'InternVL3-2B',
            'InternVL25_4B': 'InternVL2.5-4B',
            
            # Other models - Clean model names
            'Moondream2_2B': 'Moondream2',
            'LLAVA_1pt5_7B': 'LLaVA-1.5-7B',
            'LLAVA_v1pt6_Mistral_7B': 'LLaVA-v1.6-Mistral-7B',
            
            # Legacy mappings for backward compatibility
            'gpt4o': 'GPT-4o',
            'phi3pt5_vision_4b': 'Phi-3.5-Vision-4B',
            'florence2_pt23b': 'Florence-2-Base',
            'florence2_pt77b': 'Florence-2-Large',
            'glmedge_2b': 'GLM-Edge-V-2B'
        }
        
        if model_name in name_mappings:
            return name_mappings[model_name]
        else:
            # Generate display name by cleaning up underscores and capitalizing
            return model_name.replace('_', '-').upper()
    
    def _estimate_model_size(self, model_name):
        """Get official model size in billions of parameters from HuggingFace specifications."""
        
        # Official model sizes from HuggingFace (matching database_manager.py)
        official_sizes = {
            'Qwen25_VL_3B': 3.75,        # 3.75B params
            'Qwen25_VL_7B': 8.29,        # 8.29B params  
            'Qwen2_VL_2B': 2.21,         # 2.21B params
            'Gemma3_VL_4B': 4.3,         # 4.3B params
            'PaliGemma_VL_3B': 2.92,     # 2.92B params
            'DeepSeek1_VL_1pt3B': 1.98,  # 1.98B params
            'DeepSeek1_VL_7B': 7.34,     # 7.34B params
            'SmolVLM2_pt25B': 0.256,     # 0.256B params
            'SmolVLM2_pt5B': 0.507,      # 0.507B params
            'SmolVLM2_2pt2B': 2.25,      # 2.25B params
            'InternVL3_1B': 0.938,       # 0.938B params
            'InternVL3_2B': 2.09,        # 2.09B params
            'InternVL25_4B': 3.71,       # 3.71B params
            'Moondream2_2B': 1.93,       # 1.93B params
            'LLAVA_1pt5_7B': 7.06,       # 7.06B params
            'LLAVA_v1pt6_Mistral_7B': 7.57,  # 7.57B params
            
            # Legacy models
            'gpt4o': 175.0,              # Estimated
            'florence2_pt23b': 0.23,     # 0.23B params
            'florence2_pt77b': 0.77,     # 0.77B params
            'phi3pt5_vision_4b': 4.15,   # 4.15B params
            'glmedge_2b': 2.0            # Estimated
        }
        
        # Use exact model name match for official sizes
        if model_name in official_sizes:
            return official_sizes[model_name]
        
        # Fallback: Legacy regex-based estimation for unknown models
        import re
        name_lower = model_name.lower()
        
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
                    if 'pt' in name_lower and 'pt' not in match.group(0):
                        pt_match = re.search(r'pt(\d+)', name_lower)
                        if pt_match:
                            return float(f"0.{pt_match.group(1)}")
                    return float(match.group(1))
                elif len(match.groups()) == 2:
                    return float(f"{match.group(1)}.{match.group(2)}")
        
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
            "White-Box": "#FF7F0E",
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
        """REMOVED: Redundant plot - information covered by other visualizations."""
        print("⚠️  Skipping redundant 02_model_degradation_line.png (redundant)")
        return
    
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
    
    # =================== ATTACK GENERATION ANALYSIS PLOTS ===================
    
    def load_attack_comprehensive_data(self):
        """Load attack generation comprehensive analysis data from database."""
        try:
            query = """
            SELECT 
                attack_name,
                attack_category,
                avg_attack_time,
                avg_ssim,
                avg_mean_perturbation,
                avg_accuracy_change,
                param_executions
            FROM attack_comprehensive_analysis
            WHERE param_success_rate = 1.0
            """
            
            df = pd.read_sql_query(query, self.conn)
            print(f"✅ Loaded attack comprehensive data: {len(df)} records")
            return df
            
        except Exception as e:
            print(f"❌ Error loading attack comprehensive data: {e}")
            return None
    
    def plot_attack_generation_efficiency_analysis(self):
        """REMOVED: Redundant plot - information covered by plot_attack_raw_metrics_comparison."""
        print("⚠️  Skipping redundant 14_attack_generation_efficiency_analysis.png (covered by plot 15)")
        return

    
    def plot_attack_raw_metrics_comparison(self):
        """Create focused single-panel visualization for tier-1 research standards."""
        attack_data = self.load_attack_comprehensive_data()
        if attack_data is None or attack_data.empty:
            print("⚠️ No attack comprehensive analysis data found - skipping raw metrics plot")
            return
        
        # Create single panel focusing on most critical research insight
        plt.figure(figsize=(12, 8))
        
        # Enhanced color mapping
        colors = {'White-Box': '#FF6B35', 'Black-Box': '#004E89'}
        
        # Stealth vs Impact Analysis - The core research contribution
        for category in attack_data['attack_category'].unique():
            cat_data = attack_data[attack_data['attack_category'] == category]
            scatter = plt.scatter(cat_data['avg_mean_perturbation'], abs(cat_data['avg_accuracy_change']),
                                s=cat_data['avg_attack_time'] * 12,  # Size = generation time
                                alpha=0.7, color=colors[category],
                                label=f'{category} Attacks', edgecolors='black', linewidth=1.5)
        
        # RESEARCH-STANDARD: Non-uniform scaling to resolve clustering
        import numpy as np
        
        # Get original perturbation values
        x_original = attack_data['avg_mean_perturbation'].values
        
        # UNIVERSAL AUTO-SCALING FUNCTION: Handles both sharp and flat distributions
        def universal_auto_scale(vals, target_range=(0, 20), expansion_strength=2.5):
            """
            Universal automatic scaling using adaptive method selection:
            - For sharp density distributions: Use density-based expansion
            - For flat distributions: Use percentile-based expansion
            
            Automatically detects which method to use based on density contrast.
            """
            from scipy import stats
            from scipy.interpolate import interp1d
            
            if len(vals) < 2:
                return vals, lambda x: x
            
            val_min, val_max = vals.min(), vals.max()
            if val_min == val_max:
                return vals, lambda x: x
            
            # Step 1: Analyze density contrast to choose method
            try:
                density = stats.gaussian_kde(vals, bw_method='scott')
                val_test = np.linspace(val_min, val_max, 300)
                density_vals = density(val_test)
                density_contrast = density_vals.max() / (density_vals.min() + 1e-10)
            except:
                density_contrast = 1.0
            
            # Step 2: Choose method based on density contrast
            if density_contrast > 5.0:  # Sharp distribution - use density method
                return density_based_scaling(vals, target_range, expansion_strength)
            else:  # Flat distribution - use cluster-aware method for Y-axis
                return cluster_aware_scaling(vals, target_range)
        
        def density_based_scaling(vals, target_range, expansion_strength):
            """Density-based scaling for sharp distributions (like X-axis)"""
            from scipy import stats, interpolate
            
            density = stats.gaussian_kde(vals, bw_method='scott')
            val_min, val_max = vals.min(), vals.max()
            val_test = np.linspace(val_min, val_max, 300)
            density_vals = density(val_test)
            
            density_norm = (density_vals - density_vals.min()) / (density_vals.max() - density_vals.min() + 1e-10)
            expansion_factors = 1.0 + density_norm * expansion_strength
            
            dval = np.diff(val_test)
            dval = np.append(dval, dval[-1])
            stretched_intervals = dval * expansion_factors
            val_transformed_test = np.cumsum(stretched_intervals)
            
            target_min, target_max = target_range
            val_transformed_test = target_min + (val_transformed_test / val_transformed_test[-1]) * (target_max - target_min)
            
            transform_func = interpolate.interp1d(val_test, val_transformed_test, kind='cubic',
                                                bounds_error=False, fill_value='extrapolate')
            vals_scaled = transform_func(vals)
            
            return vals_scaled, transform_func
        
        def cluster_aware_scaling(vals, target_range):
            """Cluster-aware scaling that compresses empty gaps and expands within clusters"""
            from scipy import stats, interpolate
            from sklearn.cluster import KMeans
            
            # Detect clusters in the data
            vals_reshaped = vals.reshape(-1, 1)
            n_clusters = min(4, len(vals))  # Max 4 clusters, but not more than data points
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(vals_reshaped)
            cluster_centers = kmeans.cluster_centers_.flatten()
            
            # Sort clusters by center value
            cluster_order = np.argsort(cluster_centers)
            
            # Create transformation mapping
            val_min, val_max = vals.min(), vals.max()
            target_min, target_max = target_range
            
            # Allocate space: 80% for clusters, 20% for gaps
            cluster_space = (target_max - target_min) * 0.8
            gap_space = (target_max - target_min) * 0.2
            
            space_per_cluster = cluster_space / n_clusters
            space_per_gap = gap_space / max(1, n_clusters - 1)
            
            # Build transformation function
            val_test = np.linspace(val_min, val_max, 500)
            val_transformed_test = np.zeros_like(val_test)
            
            current_target = target_min
            
            for i, cluster_idx in enumerate(cluster_order):
                # Find data points in this cluster
                cluster_mask = cluster_labels == cluster_idx
                cluster_vals = vals[cluster_mask]
                
                if len(cluster_vals) == 0:
                    continue
                    
                cluster_min, cluster_max = cluster_vals.min(), cluster_vals.max()
                
                # Map this cluster's range to allocated space
                cluster_start = current_target
                cluster_end = current_target + space_per_cluster
                
                # Transform values in this cluster's range
                in_cluster = (val_test >= cluster_min) & (val_test <= cluster_max)
                if cluster_max > cluster_min:
                    val_transformed_test[in_cluster] = cluster_start + (
                        (val_test[in_cluster] - cluster_min) / (cluster_max - cluster_min)
                    ) * space_per_cluster
                else:
                    val_transformed_test[in_cluster] = cluster_start + space_per_cluster / 2
                
                current_target = cluster_end + space_per_gap
            
            # Handle gaps between clusters with linear interpolation
            for i in range(len(val_test) - 1):
                if val_transformed_test[i] == 0 and val_transformed_test[i+1] == 0:
                    # Find nearest non-zero values
                    prev_idx = i
                    while prev_idx >= 0 and val_transformed_test[prev_idx] == 0:
                        prev_idx -= 1
                    next_idx = i + 1
                    while next_idx < len(val_transformed_test) and val_transformed_test[next_idx] == 0:
                        next_idx += 1
                    
                    if prev_idx >= 0 and next_idx < len(val_transformed_test):
                        # Linear interpolation
                        alpha = (i - prev_idx) / (next_idx - prev_idx)
                        val_transformed_test[i] = (1 - alpha) * val_transformed_test[prev_idx] + alpha * val_transformed_test[next_idx]
            
            # Create interpolation function
            transform_func = interpolate.interp1d(val_test, val_transformed_test, kind='linear',
                                                bounds_error=False, fill_value='extrapolate')
            vals_scaled = transform_func(vals)
            
            return vals_scaled, transform_func
        
        # Apply universal auto-scaling to BOTH axes
        y_original = abs(attack_data['avg_accuracy_change']).values
        
        # Transform X-axis (perturbation) - automatic method selection
        x_transformed, x_transform_func = universal_auto_scale(x_original, target_range=(0, 20), expansion_strength=2.5)
        
        # Transform Y-axis (accuracy) - automatic method selection (will use percentile-based)
        y_transformed, y_transform_func = universal_auto_scale(y_original, target_range=(0, 12), expansion_strength=2.5)
        
        # Clear and create new plot with dual-transformed coordinates
        plt.clf()
        plt.figure(figsize=(12, 8))
        
        # Plot with both axes auto-scaled
        for category in attack_data['attack_category'].unique():
            cat_data = attack_data[attack_data['attack_category'] == category]
            cat_indices = cat_data.index
            cat_x_transformed = np.array([x_transformed[attack_data.index.get_loc(idx)] for idx in cat_indices])
            cat_y_transformed = np.array([y_transformed[attack_data.index.get_loc(idx)] for idx in cat_indices])
            
            scatter = plt.scatter(cat_x_transformed, cat_y_transformed,
                                s=(cat_data['avg_ssim'] - 0.85) * 2000 + 50,  # Enhanced scaling: Low SSIM=50, High SSIM=340
                                alpha=0.7, color=colors[category],
                                label=f'{category} Attacks', edgecolors='black', linewidth=1.5)
        
        # Add labels close to bubbles using transformed coordinates
        for i, (_, row) in enumerate(attack_data.iterrows()):
            x_pos = x_transformed[i]
            y_pos = y_transformed[i]
            
            plt.annotate(row['attack_name'], (x_pos, y_pos),
                        xytext=(6, 6), textcoords='offset points', 
                        fontsize=9, weight='bold',
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.95, 
                                 edgecolor=colors.get(row['attack_category'], 'black'), linewidth=1),
                        color=colors.get(row['attack_category'], 'black'))
        
        # AUTOMATIC X-axis ticks
        x_min, x_max = x_original.min(), x_original.max()
        x_auto_ticks = np.linspace(x_min, x_max, 8)
        x_transformed_ticks = x_transform_func(x_auto_ticks)
        plt.xticks(x_transformed_ticks, [f'{orig:.1f}' for orig in x_auto_ticks])
        
        # AUTOMATIC Y-axis ticks  
        y_min, y_max = y_original.min(), y_original.max()
        y_auto_ticks = np.linspace(y_min, y_max, 7)
        y_transformed_ticks = y_transform_func(y_auto_ticks)
        plt.yticks(y_transformed_ticks, [f'{orig:.1f}' for orig in y_auto_ticks])
        
        plt.xlabel('Average Mean Perturbation (pixel changes)', fontsize=16)
        plt.ylabel('Average Accuracy Change (%)', fontsize=16)
        plt.title('Average Mean Perturbation vs Average Accuracy Change', fontsize=18, weight='bold', pad=20)
        
        # Professional legend without clutter
        legend_elements = []
        for category, color in colors.items():
            legend_elements.append(plt.scatter([], [], color=color, s=120, alpha=0.7, 
                                             edgecolors='black', linewidth=1.5, label=f'{category} Attacks'))
        
        # Auto-calculate SSIM ranges from data for bubble size legend
        ssim_min = attack_data['avg_ssim'].min()
        ssim_max = attack_data['avg_ssim'].max()
        ssim_33 = attack_data['avg_ssim'].quantile(0.33)
        ssim_67 = attack_data['avg_ssim'].quantile(0.67)
        
        time_sizes = [50, 150, 340]  # Clear visual distinction
        time_labels = [f'Low avg_ssim ({ssim_min:.3f}-{ssim_33:.3f})', 
                      f'Medium avg_ssim ({ssim_33:.3f}-{ssim_67:.3f})', 
                      f'High avg_ssim ({ssim_67:.3f}-{ssim_max:.3f})']
        for size, label in zip(time_sizes, time_labels):
            legend_elements.append(plt.scatter([], [], s=size, alpha=0.7, color='gray', 
                                             edgecolors='black', linewidth=1, label=label))
        
        plt.legend(handles=legend_elements, title="Attack Properties", 
                  fontsize=11, title_fontsize=12, loc='upper right')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '15_attack_raw_metrics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Created: 15_attack_raw_metrics_comparison.png")
    
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
        category_colors = {'White-Box': '#FF6B35', 'Black-Box': '#004E89'}  # Enhanced colors
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
        
        plt.title('Attack Effectiveness: Performance Degradation Analysis\n(Sorted by Effectiveness)', fontsize=16)
        plt.xlabel('Attack Methods (White-Box vs Black-Box)', fontsize=14)
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
        whitebox_attacks = attack_success[attack_success['attack_category'] == 'White-Box']
        blackbox_attacks = attack_success[attack_success['attack_category'] == 'Black-Box']
        
        if len(whitebox_attacks) > 0:
            plt.axhline(y=whitebox_attacks['degradation'].mean(), color='#FF6B35', 
                       linestyle='--', alpha=0.5, label='White-Box Avg')
        if len(blackbox_attacks) > 0:
            plt.axhline(y=blackbox_attacks['degradation'].mean(), color='#004E89', 
                       linestyle='--', alpha=0.5, label='Black-Box Avg')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '10_attack_transferability_analysis.png'), dpi=300)
        plt.close()
        print("✅ Created: 10_attack_transferability_analysis.png")
    
    def plot_size_memory_quality_bubble(self, df):
        """REMOVED: Redundant plot - information covered by plot_size_vs_memory with trend analysis."""
        print("⚠️  Skipping redundant 11_size_memory_quality_bubble.png (covered by plot 12)")
        return

    
    def plot_size_vs_memory(self, df):
        """Create scatter plot using database variables with cluster-aware scaling."""
        model_data = df.groupby(['model_name', 'family_name']).agg({
            'avg_gpu_memory_peak_mb': 'mean'
        }).reset_index()
        
        # Add display names and model sizes from model_info
        model_data['display_name'] = model_data['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('display_name', x)
        )
        model_data['model_size_b'] = model_data['model_name'].apply(
            lambda x: self.model_info.get(x, {}).get('size_b', 1.0)
        )
        
        plt.figure(figsize=(14, 10))
        
        # Apply cluster-aware scaling to both axes
        x_original = model_data['model_size_b'].values
        y_original = (model_data['avg_gpu_memory_peak_mb'] / 1000).values  # Convert to GB
        
        x_transformed, x_transform_func = global_cluster_aware_scaling(x_original, target_range=(0, 20))
        y_transformed, y_transform_func = global_cluster_aware_scaling(y_original, target_range=(0, 12))
        
        # Create scatter plot with transformed coordinates
        for i, (_, row) in enumerate(model_data.iterrows()):
            family = row['family_name']
            plt.scatter(x_transformed[i], y_transformed[i],
                       color=self.family_colors.get(family, 'gray'),
                       s=120, alpha=0.8, 
                       label=family if family not in plt.gca().get_legend_handles_labels()[1] else "",
                       edgecolors='black', linewidth=0.5)
        
        # Add labels without arrows (direct positioning) - rotated 90 degrees
        for i, (_, row) in enumerate(model_data.iterrows()):
            plt.annotate(row['display_name'], (x_transformed[i], y_transformed[i]),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, weight='bold', rotation=90,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, 
                                 edgecolor='gray', linewidth=1))
        
        # Add trend line using original values
        x = model_data['model_size_b']
        y = model_data['avg_gpu_memory_peak_mb'] / 1000
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        r_squared = 1 - (sum((y - p(x))**2) / ((len(y) - 1) * np.var(y, ddof=1)))
        
        # Transform trend line points
        x_trend_orig = np.linspace(x.min(), x.max(), 100)
        y_trend_orig = p(x_trend_orig)
        x_trend_trans = x_transform_func(x_trend_orig)
        y_trend_trans = y_transform_func(y_trend_orig)
        plt.plot(x_trend_trans, y_trend_trans, "k--", alpha=0.8, linewidth=2.5, label="Memory Trend")
        
        # Set axis ticks using original values
        x_min, x_max = x_original.min(), x_original.max()
        x_auto_ticks = np.linspace(x_min, x_max, 6)
        x_transformed_ticks = x_transform_func(x_auto_ticks)
        plt.xticks(x_transformed_ticks, [f'{orig:.1f}' for orig in x_auto_ticks])
        
        y_min, y_max = y_original.min(), y_original.max()
        y_auto_ticks = np.linspace(y_min, y_max, 6)
        y_transformed_ticks = y_transform_func(y_auto_ticks)
        plt.yticks(y_transformed_ticks, [f'{orig:.1f}' for orig in y_auto_ticks])
        
        # Trend info
        plt.text(0.98, 0.02, f"Trend: y={p.coefficients[0]:.2f}x+{p.coefficients[1]:.2f}\\nR²={r_squared:.3f}", 
                transform=plt.gca().transAxes, fontsize=10, weight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.95, edgecolor='orange'),
                ha='right', va='bottom')
        
        plt.xlabel('Model Size (Billion Parameters)', fontsize=14)
        plt.ylabel('Average GPU Memory Peak (GB)', fontsize=14)
        plt.title('Model Size vs Average GPU Memory Peak', fontsize=16)
        plt.legend(title="Model Family", bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, '12_size_vs_memory.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Created: 12_size_vs_memory.png")
    
    def _smart_label_placement(self, model_data):
        """Fallback manual smart label placement when adjustText is not available."""
        # Divide plot into regions and distribute labels
        for i, (_, row) in enumerate(model_data.iterrows()):
            full_name = row['display_name']
            x_pos = row['model_size_b']
            y_pos = row['avg_gpu_memory_peak_mb'] / 1000
            
            # Use radial positioning to spread labels around points
            import math
            angle = (i * 360 / len(model_data)) * math.pi / 180  # Distribute around circle
            radius = 0.3  # Distance from point
            
            offset_x = radius * math.cos(angle) * 1.5  # Stretch horizontally
            offset_y = radius * math.sin(angle)
            
            plt.annotate(full_name,
                        (x_pos, y_pos),
                        xytext=(x_pos + offset_x, y_pos + offset_y), 
                        fontsize=8, weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                alpha=0.9, edgecolor='darkgray', linewidth=1),
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.6, lw=1),
                        ha='center', va='center')
    
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
        # Calculate real robustness scores from actual attack results
        robustness_df = self.load_robustness_data()
        if robustness_df is None or robustness_df.empty:
            print("❌ No robustness data found - CANNOT generate radar chart without real data")
            return
        
        # Filter out baseline 'Original' attacks to get actual adversarial results
        attacks_only = robustness_df[robustness_df['attack_name'] != 'Original']
        if attacks_only.empty:
            print("❌ No adversarial attack data found - CANNOT generate radar chart")
            return
        
        # Calculate mean accuracy degradation per model (more negative = less robust)
        model_robustness = attacks_only.groupby('model_name')['accuracy_change'].mean()
        
        # Convert to 0-1 robustness score: less degradation = higher score
        min_degradation = model_robustness.min()  # Most negative (least robust)
        max_degradation = model_robustness.max()  # Least negative (most robust)
        
        if min_degradation == max_degradation:
            print("❌ All models have identical robustness - CANNOT differentiate")
            return
        
        # Normalize: (value - min) / (max - min) where less negative = higher score
        robustness_normalized = (model_robustness - min_degradation) / (max_degradation - min_degradation)
        
        # Only keep models that exist in both performance and robustness data
        performance_models = set(model_scores['model_name'])
        robustness_models = set(robustness_normalized.index)
        common_models = performance_models & robustness_models
        
        if len(common_models) < 3:
            print(f"❌ Only {len(common_models)} models have both performance and robustness data - need at least 3")
            return
        
        # Filter to common models only
        model_scores = model_scores[model_scores['model_name'].isin(common_models)]
        model_scores['robustness_score'] = model_scores['model_name'].map(robustness_normalized)
        
        print(f"✅ Using real robustness data for {len(common_models)} models")
        print(f"   Robustness range: {min_degradation:.1f}% to {max_degradation:.1f}% avg degradation")
        
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
        
        # Load and generate attack generation analysis
        print(f"\n🔧 Generating attack generation analysis plots...")
        print("📊 Section 8: Attack Generation Analysis")
        self.plot_attack_generation_efficiency_analysis()
        self.plot_attack_raw_metrics_comparison()
        
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