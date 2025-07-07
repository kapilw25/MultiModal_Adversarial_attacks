import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

# Create output directory if it doesn't exist
output_dir = '/home/016649880@SJSUAD/Multi-modal-Self-instruct/scripts/data_analysis/plots'
os.makedirs(output_dir, exist_ok=True)

# Create a DataFrame from the data
data = {
    'Attack Name': ['Original', 'PGD', 'FGSM', 'CW-L2', 'CW-L0', 'CW-L∞', 'L-BFGS', 'JSMA', 'DeepFool', 
                   'Square', 'HopSkipJump', 'Pixel', 'SimBA', 'Spatial', 'Query-Efficient BB', 'ZOO', 'Boundary', 'GeoDA'],
    'Attack Type': ['-', 'Transfer', 'Transfer', 'Transfer', 'Transfer', 'Transfer', 'Transfer', 'Transfer', 'Transfer',
                   'Black-Box', 'Black-Box', 'Black-Box', 'Black-Box', 'Black-Box', 'Black-Box', 'Black-Box', 'Black-Box', 'Black-Box'],
    'Original Accuracy': [82.35] * 18,
    'Attack Accuracy': [82.35, 70.59, 35.29, 35.29, 47.06, 29.41, 35.29, 82.35, 47.06, 
                       76.47, 47.06, 29.41, 41.18, 52.94, 76.47, 41.18, 41.18, 88.24],
    'Change': [0.00, -11.76, -47.06, -47.06, -35.29, -52.94, -47.06, 0.00, -35.29, 
              -5.88, -35.29, -52.94, -41.18, -29.41, -5.88, -41.18, -41.18, 5.88]
}

df = pd.DataFrame(data)

# Sort by Attack Accuracy for better visualization
df_sorted = df.sort_values('Attack Accuracy')

# Set up the figure with a larger size
plt.figure(figsize=(14, 8))

# Plot 1: Attack Accuracy Comparison
plt.subplot(1, 2, 1)
sns.barplot(x='Attack Accuracy', y='Attack Name', data=df_sorted, hue='Attack Name', palette='viridis', legend=False)
plt.title('Attack Accuracy Comparison', fontsize=14)
plt.xlabel('Accuracy (%)', fontsize=12)
plt.ylabel('Attack Name', fontsize=12)
plt.axvline(x=82.35, color='red', linestyle='--', label='Original Accuracy')
plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Plot 2: Accuracy Change by Attack Type
plt.subplot(1, 2, 2)
# Create a categorical color map
colors = {'Transfer': 'blue', 'Black-Box': 'green', '-': 'gray'}
attack_colors = [colors[t] for t in df['Attack Type']]

# Sort by Change for better visualization
df_change_sorted = df.sort_values('Change')
sns.barplot(x='Change', y='Attack Name', data=df_change_sorted, hue='Attack Name', palette='RdYlGn', legend=False)
plt.title('Accuracy Change by Attack', fontsize=14)
plt.xlabel('Change in Accuracy (%)', fontsize=12)
plt.ylabel('Attack Name', fontsize=12)
plt.axvline(x=0, color='black', linestyle='--')
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'attack_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
print(f"Saved plot 1 to {os.path.join(output_dir, 'attack_accuracy_comparison.png')}")

# Plot 3: Comparison between Transfer-based and Black-box attacks
plt.figure(figsize=(12, 6))

# Group by attack type and calculate mean accuracy
attack_type_avg = df[df['Attack Name'] != 'Original'].groupby('Attack Type')['Attack Accuracy'].mean().reset_index()
attack_type_std = df[df['Attack Name'] != 'Original'].groupby('Attack Type')['Attack Accuracy'].std().reset_index()

# Create a bar plot with error bars
sns.barplot(x='Attack Type', y='Attack Accuracy', data=attack_type_avg, hue='Attack Type', palette='Set2', legend=False)
plt.errorbar(x=range(len(attack_type_avg)), 
             y=attack_type_avg['Attack Accuracy'], 
             yerr=attack_type_std['Attack Accuracy'],
             fmt='none', color='black', capsize=5)

plt.title('Average Accuracy by Attack Type', fontsize=14)
plt.xlabel('Attack Type', fontsize=12)
plt.ylabel('Average Accuracy (%)', fontsize=12)
plt.axhline(y=82.35, color='red', linestyle='--', label='Original Accuracy')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'average_accuracy_by_attack_type.png'), dpi=300, bbox_inches='tight')
print(f"Saved plot 2 to {os.path.join(output_dir, 'average_accuracy_by_attack_type.png')}")

# Plot 4: Distribution of accuracy changes
plt.figure(figsize=(10, 6))
sns.histplot(data=df[df['Attack Name'] != 'Original'], x='Change', bins=10, kde=True)
plt.title('Distribution of Accuracy Changes', fontsize=14)
plt.xlabel('Change in Accuracy (%)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.axvline(x=0, color='red', linestyle='--', label='No Change')
plt.legend()
plt.grid(linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'distribution_of_accuracy_changes.png'), dpi=300, bbox_inches='tight')
print(f"Saved plot 3 to {os.path.join(output_dir, 'distribution_of_accuracy_changes.png')}")

# Plot 5: Grouped bar chart comparing Transfer vs Black-Box attacks
plt.figure(figsize=(16, 10))

# Create a new column for categorizing attacks
df['Effectiveness'] = pd.cut(
    df['Change'],
    bins=[-60, -40, -20, -0.001, 10],
    labels=['High Degradation', 'Moderate Degradation', 'Low Degradation', 'Improvement']
)

# Count attacks in each category by type
effectiveness_by_type = pd.crosstab(df['Effectiveness'], df['Attack Type'])
effectiveness_by_type = effectiveness_by_type.reindex(['High Degradation', 'Moderate Degradation', 'Low Degradation', 'Improvement'])

# Plot the grouped bar chart
effectiveness_by_type.plot(kind='bar', figsize=(12, 6), color=['blue', 'green'])
plt.title('Attack Effectiveness by Type', fontsize=14)
plt.xlabel('Effectiveness Category', fontsize=12)
plt.ylabel('Number of Attacks', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Attack Type')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'attack_effectiveness_by_type.png'), dpi=300, bbox_inches='tight')
print(f"Saved plot 4 to {os.path.join(output_dir, 'attack_effectiveness_by_type.png')}")

print("All plots have been saved successfully!")
