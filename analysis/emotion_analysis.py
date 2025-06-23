import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from scipy.stats import chi2_contingency
from util import print_chi2_results

results_dir = "../results/"

df = pd.read_csv(f"{results_dir}emotion_summary.csv")

df_numeric = df.select_dtypes(include=[np.number])
df_numeric['infile'] = df['infile']

def plot_comparison(group1, group2, title, filename):
    subset = df_numeric[df_numeric['infile'].isin([group1, group2])].set_index('infile')

    columns_to_plot = subset.columns
    fig, ax = plt.subplots(figsize=(10, 8))

    relative_changes = {}
    for col in columns_to_plot:
        x1, x2 = subset.loc[group1, col], subset.loc[group2, col]
        change = x2 - x1
        relative_changes[col] = change

    min_change = min(relative_changes.values())
    max_change = max(relative_changes.values())

    small_change_threshold = 0.001
    medium_change_threshold = 0.02

    for i, col in enumerate(columns_to_plot):
        x1, x2 = subset.loc[group1, col], subset.loc[group2, col]
        change = relative_changes[col]

        if abs(change) > medium_change_threshold:
            if change > 0:
                color = "#7EBC6D"
            else:
                color = "#9E222C"
        elif abs(change) > small_change_threshold:
            if change > 0:
                color = "#9ACB73"
            else:
                color = "#9E222C"
        else:
            color = "#D9B700"

        ax.plot([np.log(x1), np.log(x2)], [i, i], color=color, lw=2, zorder=3)

        ax.scatter([np.log(x2)], [i], color=color, s=100, zorder=3)

    ax.set_yticks(range(len(columns_to_plot)))
    ax.set_yticklabels(columns_to_plot)
    ax.set_title(title)
    
    ax.set_xlabel('log of average probability float')

    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(f"{results_dir}{filename}.png")
    plt.close()

plot_comparison("first_gen", "second_gen", "Changes in Emotion from First Gen to Second Gen", "gen_compare")

emotion_cols = [col for col in df_numeric.columns if col != "infile"]
subset = df_numeric[df_numeric['infile'].isin(["first_gen", "second_gen"])].set_index('infile')

counts_df = (subset[emotion_cols] * 1000).astype(int).T
counts_df.columns = ['first_gen', 'second_gen']
print_chi2_results(counts_df.reset_index(drop=True), description="emotions")
