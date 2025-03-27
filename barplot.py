import seaborn as sns
import matplotlib.pyplot as plt

for metric in metrics:
    plt.figure(figsize=(5, 4))

    ax = sns.barplot(
        x='Group',
        y=metric,
        data=df,
        errorbar='sd',
        capsize=0.1,
        err_kws={'linewidth': 1}
    )

    # 막대 안쪽 가운데에 평균값 표시
    for bar in ax.patches:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height / 2,  # 막대의 중간 위치
            f'{height:.2f}',
            ha='center', va='center', color='white', fontsize=10, fontweight='bold'
        )

    plt.title(f'{metric} (Mean ± SD)')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()