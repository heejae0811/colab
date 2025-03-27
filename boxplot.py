import seaborn as sns
import matplotlib.pyplot as plt

for metric in metrics:
    plt.figure(figsize=(5, 4))

    ax = sns.boxplot(
        x='Group',
        y=metric,
        data=df,
        showfliers=True,
        linewidth=1.5,
        width=0.5
    )

    # 박스 중앙에 중앙값 표시 (배경 없음)
    group_labels = df['Group'].unique()
    for i, group in enumerate(group_labels):
        median_val = df[df['Group'] == group][metric].median()
        ax.text(
            i, median_val,
            f'{median_val:.2f}',
            ha='center', va='center', color='white', fontsize=10, fontweight='bold'
        )

    plt.title(f'{metric} (Boxplot with Median)')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()