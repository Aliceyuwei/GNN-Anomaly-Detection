# ### 特徵選擇 ###
import json
import os                        # 用於處理檔案與資料夾路徑，例如組合絕對路徑、讀取目前程式所在位置等
import pandas as pd              # 用於資料讀取與處理（表格型資料，如 CSV），提供 DataFrame 結構
import numpy as np               # 提供數值運算功能，例如矩陣操作、統計計算

from preprocess.plot_correlation import (
    plot_correlation_heatmap,
    plot_feature_correlation_bar,
)


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
        corr_series = compute_feature_correlation(
            df, numeric_features).dropna()
    except Exception as e:
        print("計算相關係數時發生錯誤：", e)
        return []

    print("與 Label_Binary 的相關係數（絕對值由大到小）:")
    print(corr_series)

    # 步驟 4：條狀圖（前 0.05 篩選）
    # 篩選後的相關係數條狀圖
    filtered_corr_series = corr_series[abs(corr_series) >= corr_threshold]

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

    return final_features, df[numeric_features].corr()
