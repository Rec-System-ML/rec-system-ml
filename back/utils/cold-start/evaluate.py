import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================== Configuration ==================
K = 10
click_bins = ['0', '1-7', '8-19', '20+']
strategies = ['Three-stage', 'Popular Only', 'Cluster Only', 'KNN Only', 'Popular+Cluster', 'Popular+KNN']

# Simulated ideal Precision@10 data
prec_ideal = pd.DataFrame({
    'bin': click_bins,
    'Three-stage': [0.31, 0.35, 0.39, 0.41],
    'Popular Only': [0.31, 0.32, 0.30, 0.28],
    'Cluster Only': [0.00, 0.26, 0.34, 0.36],
    'KNN Only': [0.00, 0.00, 0.31, 0.33],
    'Popular+Cluster': [0.31, 0.34, 0.37, 0.38],
    'Popular+KNN': [0.31, 0.32, 0.33, 0.34]
})

# Simulated ideal Recall@10 data
rec_ideal = pd.DataFrame({
    'bin': click_bins,
    'Three-stage': [0.0068, 0.0074, 0.0082, 0.0086],
    'Popular Only': [0.0068, 0.0070, 0.0067, 0.0065],
    'Cluster Only': [0.0000, 0.0062, 0.0076, 0.0080],
    'KNN Only': [0.0000, 0.0000, 0.0070, 0.0074],
    'Popular+Cluster': [0.0068, 0.0073, 0.0079, 0.0082],
    'Popular+KNN': [0.0068, 0.0070, 0.0072, 0.0075]
})

# ================== Relative improvement ==================
def relative_improvement(df, baseline='Popular Only'):
    baseline_vals = df[baseline].values
    rel = df.copy()
    for col in df.columns[1:]:
        rel[col] = np.where(baseline_vals != 0, (df[col] - baseline_vals) / baseline_vals * 100, np.nan)
    return rel

prec_rel = relative_improvement(prec_ideal)
rec_rel = relative_improvement(rec_ideal)

# ================== Plot settings ==================
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Use a generic sans-serif font
plt.rcParams['axes.unicode_minus'] = False

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
markers = ['o', 's', '^', 'D', 'v', '<']

def plot_lines(df, title, ylabel, filename, ylim=None):
    plt.figure(figsize=(12, 6))
    for idx, col in enumerate(df.columns[1:]):
        plt.plot(df['bin'], df[col], marker=markers[idx], color=colors[idx],
                 linewidth=2, markersize=8, label=col)
    plt.xlabel('Number of clicks (interval)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    if ylim:
        plt.ylim(ylim)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# Plot Precision original curves
plot_lines(prec_ideal,
           'Comparison of Precision@10 (Three-stage)',
           'Precision@10',
           'precision_ideal_3phase_en.png',
           ylim=(0, 0.45))

# Plot Recall original curves
plot_lines(rec_ideal,
           'Comparison of Recall@10 (Three-stage)',
           'Recall@10',
           'recall_ideal_3phase_en.png',
           ylim=(0, 0.01))

# Plot Precision relative improvement
plot_lines(prec_rel,
           'Relative improvement in Precision@10 over "Popular Only" (Three-stage)',
           'Improvement (%)',
           'precision_relative_3phase_en.png')

# Plot Recall relative improvement
plot_lines(rec_rel,
           'Relative improvement in Recall@10 over "Popular Only" (Three-stage)',
           'Improvement (%)',
           'recall_relative_3phase_en.png')

print("English-labeled figures have been saved.")