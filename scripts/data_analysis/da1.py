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

# Create separate dataframes for transfer and black-box attacks
df_transfer = df[df['Attack Type'] == 'Transfer'].sort_values('Attack Accuracy')
df_blackbox = df[df['Attack Type'] == 'Black-Box'].sort_values('Attack Accuracy')
df_original = df[df['Attack Type'] == '-']

# Set up the figure with a larger size for separated attack types
plt.figure(figsize=(18, 10))

# Plot 1: Attack Accuracy Comparison - Separated by Attack Type
plt.subplot(1, 2, 1)

# Define color palettes for different attack types
transfer_palette = sns.color_palette("Blues_d", len(df_transfer))
blackbox_palette = sns.color_palette("Greens_d", len(df_blackbox))
original_palette = ['gray']

# Plot transfer attacks
sns.barplot(x='Attack Accuracy', y='Attack Name', data=df_transfer, 
            palette=transfer_palette, label='Transfer Attacks')

# Plot black-box attacks
sns.barplot(x='Attack Accuracy', y='Attack Name', data=df_blackbox, 
            palette=blackbox_palette, label='Black-Box Attacks')

# Plot original baseline
sns.barplot(x='Attack Accuracy', y='Attack Name', data=df_original, 
            color='lightgray', label='Original')

plt.title('Attack Accuracy by Type', fontsize=16)
plt.xlabel('Accuracy (%)', fontsize=14)
plt.ylabel('Attack Name', fontsize=14)
plt.axvline(x=82.35, color='red', linestyle='--', label='Original Accuracy')
plt.legend(loc='lower right')
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Plot 2: Accuracy Change by Attack Type
plt.subplot(1, 2, 2)

# Sort by Change for better visualization
df_transfer_change = df_transfer.sort_values('Change')
df_blackbox_change = df_blackbox.sort_values('Change')

# Create a custom color map based on change values
transfer_cmap = plt.cm.RdYlGn(np.linspace(0.15, 0.75, len(df_transfer_change)))
blackbox_cmap = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(df_blackbox_change)))

# Plot transfer attacks change
ax1 = plt.barh(df_transfer_change['Attack Name'], df_transfer_change['Change'], 
              color=transfer_cmap, alpha=0.8, label='Transfer Attacks')

# Plot black-box attacks change
ax2 = plt.barh(df_blackbox_change['Attack Name'], df_blackbox_change['Change'], 
              color=blackbox_cmap, alpha=0.8, label='Black-Box Attacks')

# Plot original baseline
plt.barh(df_original['Attack Name'], df_original['Change'], 
        color='lightgray', label='Original')

plt.title('Accuracy Change by Attack Type', fontsize=16)
plt.xlabel('Change in Accuracy (%)', fontsize=14)
plt.ylabel('Attack Name', fontsize=14)
plt.axvline(x=0, color='black', linestyle='--')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.legend(loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'attack_accuracy_comparison_separated.png'), dpi=300, bbox_inches='tight')
print(f"Saved separated plot to {os.path.join(output_dir, 'attack_accuracy_comparison_separated.png')}")

print("Plot has been saved successfully!")
