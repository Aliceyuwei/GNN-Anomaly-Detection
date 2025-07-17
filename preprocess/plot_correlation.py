import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_heatmap(df, features, save_path=None):
    corr_matrix = df[features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title("Final Feature Correlation Heatmap")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"✅ 熱圖已儲存到 {save_path}")
    else:
        plt.show()


def plot_feature_correlation_bar(corr_series, save_path=None):
    plt.figure(figsize=(12, max(6, 0.3 * len(corr_series))))
    sns.barplot(x=corr_series.values, y=corr_series.index,
                palette='coolwarm', orient='h')
    plt.title("Filtered Feature Correlation with Label_Binary")
    plt.xlabel("Pearson Correlation")
    plt.ylabel("Feature")
    plt.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 條狀圖已儲存到 {save_path}")
    else:
        plt.show()
