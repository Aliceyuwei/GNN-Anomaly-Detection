# ### 特徵選擇 ###

import os                        # 用於處理檔案與資料夾路徑，例如組合絕對路徑、讀取目前程式所在位置等
import pandas as pd              # 用於資料讀取與處理（表格型資料，如 CSV），提供 DataFrame 結構
import numpy as np               # 提供數值運算功能，例如矩陣操作、統計計算
import matplotlib.pyplot as plt  # 用來繪圖（條狀圖、線圖等），是 matplotlib 套件的核心繪圖工具
import seaborn as sns            # 基於 matplotlib 的進階繪圖套件，擅長統計視覺化（例如熱圖、分類圖）


def load_data():
    """
    讀取預處理好的資料
    """

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, '..', 'data',
                             'processed', 'preprocessed.csv')
    file_path = os.path.abspath(file_path)

    df = pd.read_csv(file_path)
    return df


def add_label_binary(df):
    """
    將 Label 轉成二元(0 = BENIGN, 1 = 其他）
    """

    df['Label_Binary'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    # print(df.columns.tolist(), '所有欄位')
    return df


def select_numeric_features(df):
    """
    選出數值型態的特徵欄位（不包含 Label & Label_Binary & Label_enc)
    """

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 排除目標欄:['Label', 'Label_Binary', 'Label_enc']
    exclude_cols = [col for col in [
        'Label', 'Label_Binary', 'Label_enc'] if col in df.columns]

    features = [col for col in numeric_cols if col not in exclude_cols]

    return features


def compute_feature_correlation(df, features):
    """
    計算每個數值特徵與 Label_Binary 的 Pearson 相關係數
    回傳一個 Series，index 是特徵名稱，值是相關係數
    並依絕對值由大到小排序。
    """
    corr_with_label = {}
    for feature in features:
        corr = df[feature].corr(df['Label_Binary'])
        corr_with_label[feature] = corr
    corr_series = pd.Series(corr_with_label).sort_values(
        key=abs, ascending=False)
    return corr_series


def plot_correlation_heatmap(df, features, save_path=None):
    """
    畫特徵相關係數熱圖(視覺化輔助)
    """
    corr_matrix = df[features + ['Label_Binary']].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title("Feature Correlation Heatmap")
    # plt.show()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"圖已儲存到 {save_path}")
        plt.close()
    else:
        plt.show()


def plot_feature_correlation_bar(corr_series, save_path=None):
    """
    畫出所有與 Label_Binary 的相關係數條狀圖
    """
    plt.figure(figsize=(12, max(6, 0.3 * len(corr_series))))  # 根據特徵數自動調整高度
    sns.barplot(
        x=corr_series.values,
        y=corr_series.index,
        palette='coolwarm',
        orient='h'
    )
    plt.title("Feature Correlation with Label_Binary")
    plt.xlabel("Pearson Correlation")
    plt.ylabel("Feature")
    plt.axvline(0, color='black', linewidth=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"條狀圖已儲存到 {save_path}")
        plt.close()
    else:
        plt.show()


def filter_by_correlation(corr_series, threshold=0.05):
    """
    根據與 Label_Binary 的相關係數大小篩選特徵
    只保留絕對值 ≥ threshold 的特徵
    """
    selected = corr_series[abs(corr_series) >= threshold].index.tolist()
    return selected


def remove_highly_correlated_features(df, features, threshold=0.9):
    """
    刪除兩兩相關係數大於 threshold 的特徵，只保留其中一個
    """
    corr_matrix = df[features].corr().abs()
    to_keep = set(features)

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1 = features[i]
            f2 = features[j]
            if corr_matrix.loc[f1, f2] > threshold:
                # 兩特徵相關度太高，移除 f2
                if f2 in to_keep:
                    to_keep.remove(f2)
    return list(to_keep)


def run_feature_selection(corr_threshold=0.05, high_corr_threshold=0.9, plot_heatmap=False, heatmap_output_path=None, barplot_output_path=None):
    """
    整合以上步驟的主程式
    """
    df = load_data()

    # 步驟 1：加入二元 Label
    df = add_label_binary(df)

    # 步驟 2：選數值特徵
    numeric_features = select_numeric_features(df)

    # 步驟 3：計算相關係數
    corr_series = compute_feature_correlation(df, numeric_features)
    print("與 Label_Binary 的相關係數（絕對值由大到小）:")
    print(corr_series)

    # 步驟 4：畫圖：相關係數熱圖 (Correlation Heatmap) & 相關係數條狀圖
    if plot_heatmap and heatmap_output_path:
        plot_correlation_heatmap(df, numeric_features,
                                 save_path=heatmap_output_path)

    if barplot_output_path:
        plot_feature_correlation_bar(
            corr_series, save_path=barplot_output_path)

    # 步驟 5：根據門檻篩選特徵
    selected_features = filter_by_correlation(
        corr_series, threshold=corr_threshold)
    print(f"\n篩選後的特徵數量（相關係數門檻 {corr_threshold}）：{len(selected_features)}")

    # 步驟 6：剔除高度相關特徵
    selected_features_no_corr = remove_highly_correlated_features(
        df, selected_features, threshold=high_corr_threshold)
    print(
        f"剔除高度相關（> {high_corr_threshold}）後剩餘特徵數量：{len(selected_features_no_corr)}")

    return selected_features_no_corr


if __name__ == "__main__":
    # 取得目前執行的 .py 檔案的絕對路徑，並取得該檔案所在資料夾路徑
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 設定輸出圖片路徑
    heatmap_output_path = os.path.join(
        BASE_DIR, "../data/processed/correlation_heatmap.png")
    barplot_output_path = os.path.join(
        BASE_DIR, "../data/processed/correlation_bar_all.png")

    print(f"熱圖輸出路徑: {heatmap_output_path}")
    print(f"條狀圖輸出路徑: {barplot_output_path}")

    selected_features = run_feature_selection(
        plot_heatmap=True,
        heatmap_output_path=heatmap_output_path,
        barplot_output_path=barplot_output_path,
    )
    print("\n最終選擇的特徵清單：")
    print(selected_features)
