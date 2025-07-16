#!/usr/bin/env python3
"""
VLM Performance Analysis for Research Paper
Generates publication-quality graphs from VLM performance data
Organizes models by size and family for better visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# Set the style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# Create output directory if it doesn't exist
output_dir = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(output_dir, exist_ok=True)

# Define the data
data = {
    "Model": [
        "Qwen2.5-VL-3B", "Qwen2.5-VL-7B", "Qwen2-VL-2B",
        "Gemma-3-4b-it", "PaliGemma-3B",
        "DeepSeek-VL-1.3B", "DeepSeek-VL-7B",
        "SmolVLM2-256M", "SmolVLM2-500M", "SmolVLM2-2.2B",
        "Florence-2-base", "Florence-2-large",
        "Moondream2-2B",
        "GLM-Edge-V-2B",
        "InternVL3-1B", "InternVL3-2B", "InternVL2.5-4B",
        "Phi-3.5-vision-instruct"
    ],
    "Size_B": [
        3, 7, 2,
        4, 3,
        1.3, 7,
        0.256, 0.5, 2.2,
        0.23, 0.77,
        1.93,
        2,
        1, 2, 4,
        4.15
    ],
    "GPU_Memory_GB": [
        2.3, 5.7, 1.5,
        3.1, 2.2,
        1.6, 4.8,
        1.0, 1.9, 1.4,
        0.52, 1.59,
        3.7,
        3.7,
        0.87, 1.72, 2.73,
        2.34
    ],
    "Loading_Time_s": [
        17.66, 33.21, 9.78,
        19.58, 28.00,
        7.26, 29.62,
        2.28, 3.90, 16.27,
        31.72, 155.34,
        6.57,
        7.48,
        5.14, 9.51, 14.56,
        16.77
    ],
    "Inference_Time_s": [
        3.70, 2.24, 2.99,
        4.80, 0.49,
        2.96, 3.54,
        1.76, 1.37, 2.92,
        0.60, 0.66,
        2.64,
        30.58,
        6.47, 6.85, 5.89,
        64.15
    ],
    "Quantization_Strategy": [
        "float16, NF4", "float16, NF4", "float16, NF4",
        "bfloat16, NF4", "bfloat16, NF4",
        "Optimized 4-bit", "Extreme 4-bit",
        "float32", "float32", "float16, 4-bit",
        "float16", "float16",
        "float16",
        "bfloat16",
        "bfloat16, NF4", "bfloat16, NF4", "bfloat16, NF4",
        "bfloat16, NF4"
    ],
    "Response_Quality": [
        "Complete, accurate ✅", "Complete, accurate ✅", "Complete, detailed ✅",
        "Complete, concise ✅", "Brief, minimal ❌",
        "Complete, accurate ✅", "Incomplete response ❌",
        "Incomplete response ❌", "Brief, accurate ✅", "Repetitive content ❌",
        "Complete, accurate ✅", "Complete, detailed ✅",
        "Concise, accurate ✅",
        "Brief, accurate ✅",
        "Good, minor errors ⚠️", "Complete, accurate ✅", "Complete, accurate ✅",
        "Complete, accurate ✅"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Add derived columns
df["Quality_Score"] = df["Response_Quality"].apply(
    lambda x: 2 if "✅" in x else (1 if "⚠️" in x else 0)
)
df["Size_Category"] = pd.cut(
    df["Size_B"], 
    bins=[0, 1, 2, 4, 8], 
    labels=["<1B", "1-2B", "2-4B", "4-8B"]
)

# Add model family column
def get_family(model_name):
    if "Qwen" in model_name:
        return "Qwen"
    elif "Gemma" in model_name or "PaliGemma" in model_name:
        return "Google"
    elif "DeepSeek" in model_name:
        return "DeepSeek"
    elif "SmolVLM" in model_name:
        return "SmolVLM"
    elif "Florence" in model_name:
        return "Microsoft Florence"
    elif "Moondream" in model_name:
        return "Moondream"
    elif "GLM" in model_name:
        return "GLM"
    elif "InternVL" in model_name:
        return "InternVL"
    elif "Phi" in model_name:
        return "Microsoft Phi"
    else:
        return "Other"

df["Family"] = df["Model"].apply(get_family)

# Sort by size for better visualization
df = df.sort_values("Size_B")

# Define color palette for families
family_colors = {
    "Qwen": "#FF5733",
    "Google": "#33A8FF",
    "DeepSeek": "#33FF57",
    "SmolVLM": "#D133FF",
    "Microsoft Florence": "#FFD133",
    "Moondream": "#33FFEC",
    "GLM": "#FF33A8",
    "InternVL": "#8C33FF",
    "Microsoft Phi": "#33FFA8"
}

# Define quality markers
quality_markers = {0: "X", 1: "s", 2: "o"}

# 1. Size vs. GPU Memory with Family Color Coding
plt.figure(figsize=(12, 8))
for family in df["Family"].unique():
    subset = df[df["Family"] == family]
    plt.scatter(
        subset["Size_B"], 
        subset["GPU_Memory_GB"],
        s=100, 
        color=family_colors[family], 
        label=family,
        alpha=0.8,
        marker='o'
    )

# Add model names as annotations
for i, row in df.iterrows():
    plt.annotate(
        row["Model"].split("-")[0],
        (row["Size_B"], row["GPU_Memory_GB"]),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=8,
        alpha=0.8
    )

# Add trend line
z = np.polyfit(df["Size_B"], df["GPU_Memory_GB"], 1)
p = np.poly1d(z)
x_trend = np.linspace(df["Size_B"].min(), df["Size_B"].max(), 100)
plt.plot(x_trend, p(x_trend), "k--", alpha=0.5)

plt.title("Model Size vs. GPU Memory Usage", fontsize=16)
plt.xlabel("Model Size (Billion Parameters)", fontsize=14)
plt.ylabel("GPU Memory Usage (GB)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(title="Model Family", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "size_vs_memory.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(output_dir, "size_vs_memory.pdf"), bbox_inches="tight")

# 2. Size vs. Loading Time with Family Color Coding
plt.figure(figsize=(12, 8))
for family in df["Family"].unique():
    subset = df[df["Family"] == family]
    plt.scatter(
        subset["Size_B"], 
        subset["Loading_Time_s"],
        s=100, 
        color=family_colors[family], 
        label=family,
        alpha=0.8,
        marker='o'
    )

# Add model names as annotations
for i, row in df.iterrows():
    if row["Loading_Time_s"] < 50:  # Only annotate models with reasonable loading times
        plt.annotate(
            row["Model"].split("-")[0],
            (row["Size_B"], row["Loading_Time_s"]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8
        )

plt.title("Model Size vs. Loading Time", fontsize=16)
plt.xlabel("Model Size (Billion Parameters)", fontsize=14)
plt.ylabel("Loading Time (seconds)", fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(title="Model Family", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Set y-axis limit to exclude extreme outliers for better visualization
plt.ylim(0, df["Loading_Time_s"].quantile(0.9) * 1.5)
plt.savefig(os.path.join(output_dir, "size_vs_loading_time.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(output_dir, "size_vs_loading_time.pdf"), bbox_inches="tight")

# 3. Size vs. Inference Time with Quality Markers
plt.figure(figsize=(12, 8))
for quality in sorted(df["Quality_Score"].unique(), reverse=True):
    subset = df[df["Quality_Score"] == quality]
    for family in subset["Family"].unique():
        family_subset = subset[subset["Family"] == family]
        plt.scatter(
            family_subset["Size_B"], 
            family_subset["Inference_Time_s"],
            s=100, 
            color=family_colors[family],
            marker=quality_markers[quality],
            label=f"{family} ({'Good' if quality == 2 else 'Fair' if quality == 1 else 'Poor'})" if family not in [s.split()[0] for s in plt.gca().get_legend_handles_labels()[1]] else "",
            alpha=0.8
        )

# Add model names as annotations
for i, row in df.iterrows():
    if row["Inference_Time_s"] < 20:  # Only annotate models with reasonable inference times
        plt.annotate(
            row["Model"].split("-")[0],
            (row["Size_B"], row["Inference_Time_s"]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            alpha=0.8
        )

plt.title("Model Size vs. Inference Time with Response Quality", fontsize=16)
plt.xlabel("Model Size (Billion Parameters)", fontsize=14)
plt.ylabel("Inference Time (seconds)", fontsize=14)
plt.grid(True, alpha=0.3)

# Create custom legend for quality markers
quality_labels = {0: "Poor Quality", 1: "Fair Quality", 2: "Good Quality"}
quality_handles = [plt.Line2D([0], [0], marker=marker, color='gray', 
                             label=label, markersize=10, linestyle='None')
                  for quality, (marker, label) in 
                  zip(sorted(quality_markers.keys()), 
                      [(quality_markers[q], quality_labels[q]) for q in sorted(quality_markers.keys())])]

# Add both legends
handles, labels = plt.gca().get_legend_handles_labels()
first_legend = plt.legend(handles, labels, title="Model Family", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().add_artist(first_legend)
plt.legend(handles=quality_handles, title="Response Quality", bbox_to_anchor=(1.05, 0.5), loc='center left')

plt.tight_layout()
plt.ylim(0, df["Inference_Time_s"].quantile(0.9) * 1.5)
plt.savefig(os.path.join(output_dir, "size_vs_inference_time.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(output_dir, "size_vs_inference_time.pdf"), bbox_inches="tight")

# 4. Grouped Bar Chart by Size Category
plt.figure(figsize=(14, 10))

# Prepare data for grouped bar chart
size_categories = df["Size_Category"].unique()
families = df["Family"].unique()

# Calculate average metrics by size category and family
metrics_by_size_family = df.groupby(["Size_Category", "Family"]).agg({
    "GPU_Memory_GB": "mean",
    "Loading_Time_s": "mean",
    "Inference_Time_s": "mean",
    "Quality_Score": "mean"
}).reset_index()

# Filter to include only categories with data
metrics_by_size_family = metrics_by_size_family.dropna()

# Set up the plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
metrics = ["GPU_Memory_GB", "Loading_Time_s", "Inference_Time_s", "Quality_Score"]
titles = ["GPU Memory Usage (GB)", "Loading Time (s)", "Inference Time (s)", "Quality Score (0-2)"]

for i, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[i//2, i%2]
    
    # Create pivot table for easier plotting
    pivot_data = metrics_by_size_family.pivot_table(
        index="Size_Category", 
        columns="Family", 
        values=metric
    )
    
    # Plot
    pivot_data.plot(kind="bar", ax=ax, colormap=plt.cm.tab10)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Model Size Category", fontsize=12)
    ax.set_ylabel(title, fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Only show legend in the first subplot
    if i == 0:
        ax.legend(title="Model Family", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.get_legend().remove()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "metrics_by_size_category.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(output_dir, "metrics_by_size_category.pdf"), bbox_inches="tight")

# 5. Heatmap of Performance Metrics
plt.figure(figsize=(16, 12))

# Normalize metrics for fair comparison
metrics_for_heatmap = df[["Model", "Size_B", "GPU_Memory_GB", "Loading_Time_s", "Inference_Time_s", "Quality_Score"]]
metrics_for_heatmap["Size_B_norm"] = metrics_for_heatmap["Size_B"] / metrics_for_heatmap["Size_B"].max()
metrics_for_heatmap["GPU_Memory_GB_norm"] = metrics_for_heatmap["GPU_Memory_GB"] / metrics_for_heatmap["GPU_Memory_GB"].max()
metrics_for_heatmap["Loading_Time_s_norm"] = metrics_for_heatmap["Loading_Time_s"] / metrics_for_heatmap["Loading_Time_s"].max()
metrics_for_heatmap["Inference_Time_s_norm"] = metrics_for_heatmap["Inference_Time_s"] / metrics_for_heatmap["Inference_Time_s"].max()
metrics_for_heatmap["Quality_Score_norm"] = metrics_for_heatmap["Quality_Score"] / 2  # Normalize to 0-1 range

# For time metrics, lower is better, so invert the normalization
metrics_for_heatmap["Loading_Time_s_norm"] = 1 - metrics_for_heatmap["Loading_Time_s_norm"]
metrics_for_heatmap["Inference_Time_s_norm"] = 1 - metrics_for_heatmap["Inference_Time_s_norm"]

# Create heatmap data
heatmap_data = metrics_for_heatmap[["Model", "Size_B_norm", "GPU_Memory_GB_norm", 
                                   "Loading_Time_s_norm", "Inference_Time_s_norm", 
                                   "Quality_Score_norm"]]
heatmap_data = heatmap_data.set_index("Model")
heatmap_data.columns = ["Size Efficiency", "Memory Efficiency", 
                        "Loading Speed", "Inference Speed", 
                        "Response Quality"]

# Sort by overall performance
heatmap_data["Overall"] = heatmap_data.mean(axis=1)
heatmap_data = heatmap_data.sort_values("Overall", ascending=False)
heatmap_data = heatmap_data.drop("Overall", axis=1)

# Create custom colormap (green for good, red for bad)
cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#FF5533", "#FFFF99", "#33FF57"])

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(heatmap_data, annot=True, cmap=cmap, linewidths=0.5, fmt=".2f", vmin=0, vmax=1)
plt.title("VLM Performance Metrics (Higher is Better)", fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "performance_heatmap.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(output_dir, "performance_heatmap.pdf"), bbox_inches="tight")

# 6. Bubble Chart: Size vs Memory vs Inference Time
plt.figure(figsize=(14, 10))

# Create bubble chart
for family in df["Family"].unique():
    subset = df[df["Family"] == family]
    
    # Size of bubble represents quality score
    sizes = subset["Quality_Score"].apply(lambda x: (x + 1) * 100)
    
    plt.scatter(
        subset["Size_B"], 
        subset["GPU_Memory_GB"],
        s=sizes, 
        color=family_colors[family], 
        label=family,
        alpha=0.7
    )
    
    # Add model names
    for i, row in subset.iterrows():
        plt.annotate(
            row["Model"].split("-")[0],
            (row["Size_B"], row["GPU_Memory_GB"]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8
        )

plt.title("Model Size vs. Memory Usage vs. Quality", fontsize=16)
plt.xlabel("Model Size (Billion Parameters)", fontsize=14)
plt.ylabel("GPU Memory Usage (GB)", fontsize=14)
plt.grid(True, alpha=0.3)

# Create custom legend for bubble sizes
size_handles = [
    plt.scatter([], [], s=(q+1)*100, color='gray', alpha=0.7) 
    for q in sorted(df["Quality_Score"].unique())
]
size_labels = ["Poor Quality", "Fair Quality", "Good Quality"]

# Add both legends
handles, labels = plt.gca().get_legend_handles_labels()
first_legend = plt.legend(handles, labels, title="Model Family", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().add_artist(first_legend)
plt.legend(handles=size_handles, labels=size_labels, title="Response Quality", 
           bbox_to_anchor=(1.05, 0.5), loc='center left', scatterpoints=1)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "size_memory_quality_bubble.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(output_dir, "size_memory_quality_bubble.pdf"), bbox_inches="tight")

# 7. Radar Chart for Top Models
plt.figure(figsize=(14, 12))

# Select top models from each size category
top_models = []
for size_cat in df["Size_Category"].unique():
    subset = df[df["Size_Category"] == size_cat]
    if not subset.empty:
        # Get top model by quality score, then by inference time
        top_model = subset.sort_values(["Quality_Score", "Inference_Time_s"], 
                                      ascending=[False, True]).iloc[0]
        top_models.append(top_model["Model"])

# Filter dataframe to include only top models
top_df = df[df["Model"].isin(top_models)]

# Normalize metrics for radar chart
radar_metrics = ["Size_B", "GPU_Memory_GB", "Loading_Time_s", "Inference_Time_s", "Quality_Score"]
radar_df = top_df[["Model", "Family"] + radar_metrics].copy()

# For radar chart, we want higher values to be better
for metric in radar_metrics:
    if metric != "Quality_Score":  # For these metrics, lower is better
        max_val = radar_df[metric].max()
        radar_df[f"{metric}_norm"] = 1 - (radar_df[metric] / max_val)
    else:  # For quality score, higher is better
        radar_df[f"{metric}_norm"] = radar_df[metric] / 2  # Normalize to 0-1

# Set up radar chart
categories = ["Size Efficiency", "Memory Efficiency", "Loading Speed", 
              "Inference Speed", "Response Quality"]
N = len(categories)

# Create angle for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Close the loop

# Create subplot with polar projection
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

# Plot each model
for i, row in radar_df.iterrows():
    values = [
        row["Size_B_norm"],
        row["GPU_Memory_GB_norm"],
        row["Loading_Time_s_norm"],
        row["Inference_Time_s_norm"],
        row["Quality_Score_norm"]
    ]
    values += values[:1]  # Close the loop
    
    # Plot values
    ax.plot(angles, values, linewidth=2, label=row["Model"], 
            color=family_colors[row["Family"]])
    ax.fill(angles, values, alpha=0.1, color=family_colors[row["Family"]])

# Set category labels
plt.xticks(angles[:-1], categories, size=12)

# Draw y-axis labels
ax.set_rlabel_position(0)
plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=10)
plt.ylim(0, 1)

# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.title("Top Model Performance Comparison", size=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_models_radar.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(output_dir, "top_models_radar.pdf"), bbox_inches="tight")

# 8. Efficiency Score Calculation and Visualization
# Calculate efficiency score: (Quality / (Size * Memory * Loading Time * Inference Time))^(1/4)
df["Efficiency_Score"] = (
    (df["Quality_Score"] + 1) / 
    (df["Size_B"] * df["GPU_Memory_GB"] * df["Loading_Time_s"] * df["Inference_Time_s"])
)**(1/4)

# Normalize to 0-100 scale
df["Efficiency_Score"] = 100 * df["Efficiency_Score"] / df["Efficiency_Score"].max()

# Sort by efficiency score
df_sorted = df.sort_values("Efficiency_Score", ascending=False)

# Plot efficiency scores
plt.figure(figsize=(14, 10))
bars = plt.bar(
    df_sorted["Model"], 
    df_sorted["Efficiency_Score"],
    color=[family_colors[family] for family in df_sorted["Family"]]
)

# Add family color patches for legend
patches = [mpatches.Patch(color=family_colors[family], label=family) 
           for family in df_sorted["Family"].unique()]

plt.title("VLM Efficiency Score", fontsize=16)
plt.xlabel("Model", fontsize=14)
plt.ylabel("Efficiency Score (higher is better)", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.grid(axis='y', alpha=0.3)
plt.legend(handles=patches, title="Model Family", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "efficiency_score.png"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(output_dir, "efficiency_score.pdf"), bbox_inches="tight")

print(f"Generated 8 visualization plots in {output_dir}")
print("Analysis complete!")
