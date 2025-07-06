# 功能： 讀取原始合併檔，做缺值處理、欄位挑選、標籤編碼，輸出清理乾淨但未標準化的 CSV

import os  # 匯入 os 模組，用來處理檔案與路徑相關操作
import pandas as pd  # 匯入 pandas 做資料處理
from sklearn.preprocessing import LabelEncoder  # 用於標籤編碼


def clean_data(input_path, output_path):
    """
    清理資料主函式
    參數:
        input_path: str, 原始合併後的 CSV 檔案路徑
        output_path: str, 清理後（但未標準化）的資料儲存路徑
    """

    # 讀取資料
    df = pd.read_csv(input_path)
    print(f"讀取資料，shape: {df.shape}")

    # 欄位挑選（可依需求修改）
    # selected_columns = ['Feature1', 'Feature2', 'Label']
    # df = df[selected_columns]

    # 缺值處理：數值欄位以中位數補值
    missing_ratio = df.isnull().sum() / len(df)
    print("各欄位缺值比例:")
    print(missing_ratio[missing_ratio > 0])

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"{col} 補中位數 {median_val}")

    # 清理 Inf / -Inf，再次移除缺失值
    df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
    before_drop = len(df)
    df.dropna(inplace=True)
    after_drop = len(df)
    print(f"移除 NaN/Inf 後剩餘筆數: {after_drop} (移除 {before_drop - after_drop} 筆)")

    # 斷言：最後安全檢查
    assert df.isnull().sum().sum() == 0, "尚有 NaN 存在"
    assert not df.isin([float('inf'), float('-inf')]).values.any(), "尚有 Inf 存在"
    print("確認無 NaN / Inf")

    # 標籤編碼：將 'Label' 欄位轉為整數形式
    label_encoder = LabelEncoder()
    df['Label_enc'] = label_encoder.fit_transform(df['Label'])
    print("標籤編碼對照表:")
    print(dict(zip(label_encoder.classes_,
          label_encoder.transform(label_encoder.classes_))))

    # 儲存清理後的資料
    df.to_csv(output_path, index=False)
    print(f"清理後資料儲存至 {output_path}")


if __name__ == "__main__":
    # 設定檔案路徑（使用絕對路徑方便跨目錄執行）
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(BASE_DIR, "../data/processed/merged.csv")
    output_csv = os.path.join(BASE_DIR, "../data/processed/cleaned.csv")

    # 執行清理流程
    clean_data(input_csv, output_csv)
