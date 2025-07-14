# ### 特徵選擇 ###
import json
import os                        # 用於處理檔案與資料夾路徑，例如組合絕對路徑、讀取目前程式所在位置等
import pandas as pd              # 用於資料讀取與處理（表格型資料，如 CSV），提供 DataFrame 結構
import numpy as np               # 提供數值運算功能，例如矩陣操作、統計計算
import matplotlib.pyplot as plt  # 用來繪圖（條狀圖、線圖等），是 matplotlib 套件的核心繪圖工具
import seaborn as sns            # 基於 matplotlib 的進階繪圖套件，擅長統計視覺化（例如熱圖、分類圖）


def load_data(input_csv_path):
    """
    讀取預處理好的資料
    """
    return pd.read_csv(input_csv_path)


def add_label_binary(df):
    """
    將 Label 轉成二元(0 = BENIGN, 1 = 其他）
    """

    df['Label_Binary'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    return df


def select_numeric_features(df):
    """
    選出數值型態的特徵欄位（不包含 Label & Label_Binary & Label_enc)
    """

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 排除目標欄:['Label', 'Label_Binary', 'Label_enc']
    exclude_cols = [col for col in [
        'Label', 'Label_Binary', 'Label_enc'] if col in df.columns]

    return [col for col in numeric_cols if col not in exclude_cols]


def compute_feature_correlation(df, features):
    """
    計算每個數值特徵與 Label_Binary 的 Pearson 相關係數
    回傳一個 Series，index 是特徵名稱，值是相關係數（float）
    並依絕對值由大到小排序。
    """
    corr_with_label = {}
    for feature in features:
        try:
            corr = df[feature].corr(df['Label_Binary'])
            corr_with_label[feature] = float(corr)
        except Exception as e:
            print(f"⚠️ 特徵 {feature} 計算相關係數失敗：{e}")
            corr_with_label[feature] = np.nan

    corr_series = pd.Series(corr_with_label).astype(float).dropna()

    return corr_series.sort_values(key=lambda x: abs(x), ascending=False)


def plot_correlation_heatmap(df, features, save_path=None):
    """
    畫特徵之間的相關係數熱圖（輸入為最終保留特徵）
    """
    corr_matrix = df[features].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title("Final Feature Correlation Heatmap")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"圖已儲存到 {save_path}")
        plt.close()
    else:
        plt.show()


def plot_feature_correlation_bar(corr_series, save_path=None):
    """
    畫出與 Label_Binary 的相關係數條狀圖
    """
    plt.figure(figsize=(12, max(6, 0.3 * len(corr_series))))
    sns.barplot(x=corr_series.values, y=corr_series.index,
                palette='coolwarm', orient='h')
    plt.title("Filtered Feature Correlation with Label_Binary")
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
    return corr_series[abs(corr_series) >= threshold].index.tolist()


def remove_highly_correlated_features(df, features, threshold=0.9):
    """
    刪除兩兩相關係數大於 threshold 的特徵，只保留其中一個
    """
    corr_matrix = df[features].corr().abs()
    to_keep = set(features)

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            f1, f2 = features[i], features[j]
            if corr_matrix.loc[f1, f2] > threshold and f2 in to_keep:
                to_keep.remove(f2)
    return list(to_keep)


def run_feature_selection(
    input_csv_path,
    feature_output_path=None,
    corr_threshold=0.05,
    high_corr_threshold=0.9,
    barplot_output_path=None,
    heatmap_output_path=None
):
    """
    主流程：執行特徵篩選，並視覺化（條狀圖＋熱圖）
    並將選出特徵寫入 JSON
    """
    df = load_data(input_csv_path)

    # 步驟 1：加入二元 Label
    df = add_label_binary(df)

    # 步驟 2：選數值特徵
    numeric_features = select_numeric_features(df)

    # 步驟 3：計算每個數值特徵與 Label_Binary 的相關係數
    try:
        corr_series = compute_feature_correlation(df, numeric_features)
        corr_series = corr_series.dropna()
    except Exception as e:
        print("計算相關係數時發生錯誤：", e)
        return []

    print("與 Label_Binary 的相關係數（絕對值由大到小）:")
    print(corr_series)

    # 找出哪些特徵的相關係數為 NaN，這代表這些特徵可能資料有問題或無法計算相關係數
    nan_features = [f for f in numeric_features if df[f].corr(
        df['Label_Binary']) is np.nan]

    # 如果有相關係數為 NaN 的特徵，印出警告訊息
    if nan_features:
        print("⚠️ 以下欄位的相關係數為 NaN，將被排除：", nan_features)

    # 步驟 4：條狀圖（前 0.05 篩選）
    filtered_corr_series = corr_series[pd.to_numeric(
        corr_series, errors="coerce").abs() >= corr_threshold]

    if barplot_output_path:
        plot_feature_correlation_bar(
            filtered_corr_series, save_path=barplot_output_path)

    # 步驟 5：根據門檻篩選特徵
    selected_features = filter_by_correlation(
        corr_series, threshold=corr_threshold)
    print(f"\n篩選後的特徵數量（相關係數門檻 {corr_threshold}）：{len(selected_features)}")

    # 步驟 6：剔除高度相關特徵
    final_features = remove_highly_correlated_features(
        df, selected_features, threshold=high_corr_threshold)

    # 熱圖（用最終保留特徵）
    if heatmap_output_path:
        plot_correlation_heatmap(
            df, final_features, save_path=heatmap_output_path)

    if feature_output_path:
        with open(feature_output_path, "w") as f:
            json.dump(final_features, f, indent=2)
        print(f"特徵清單已儲存到 {feature_output_path}")

    return final_features


if __name__ == "__main__":
    # 取得目前執行的 .py 檔案的絕對路徑，並取得該檔案所在資料夾路徑
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 設定輸出圖片路徑
    heatmap_output_path = os.path.join(
        BASE_DIR, "../data/processed/correlation_heatmap_final.png")
    barplot_output_path = os.path.join(
        BASE_DIR, "../data/processed/correlation_bar_filtered.png")
    selected_feature_path = os.path.join(
        BASE_DIR, "../data/processed/selected_features.json")
    input_csv_path = os.path.join(
        BASE_DIR, "../data/processed/preprocessed.csv")

    print(f"熱圖輸出路徑: {heatmap_output_path}")
    print(f"條狀圖輸出路徑: {barplot_output_path}")
    print(f"特徵 JSON 輸出路徑: {selected_feature_path}")

    selected_features = run_feature_selection(
        input_csv_path=input_csv_path,
        feature_output_path=selected_feature_path,
        corr_threshold=0.05,
        high_corr_threshold=0.9,
        barplot_output_path=barplot_output_path,
        heatmap_output_path=heatmap_output_path
    )

    print("\n最終選擇的特徵清單：")
    print(selected_features)
