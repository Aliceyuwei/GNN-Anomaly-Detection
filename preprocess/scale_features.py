# 功能： 讀取清理好的檔案，對數值欄位做標準化，並輸出最終模型輸入檔

import os  # 匯入 os 模組，用來處理檔案與路徑相關操作
import pandas as pd  # 匯入 pandas 用來處理資料
from sklearn.preprocessing import StandardScaler  # 匯入標準化工具


def scale_features(input_path, output_path):
    """
    對資料進行數值標準化（z-score），輸出最終模型訓練用檔案。
    參數:
        input_path: str, 清理後的 CSV 檔案路徑
        output_path: str, 標準化後資料儲存路徑
    """

    # 讀取清理後的資料（已無缺值、已做 Label Encoding）
    df = pd.read_csv(input_path)
    print(f"讀取資料，shape: {df.shape}")

    # 特徵欄位：去除 'Label' 與 'Label_enc'（非數值特徵或已編碼的目標欄）
    features = df.drop(columns=['Label', 'Label_enc'])

    # 使用 StandardScaler 進行 Z-score 標準化：每個欄位轉換為平均 0、標準差 1
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 將標準化後資料轉回 DataFrame（保留欄位名稱）
    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)

    # 把編碼後的目標欄加回來（Label_enc）
    df_scaled['Label_enc'] = df['Label_enc'].values

    # 儲存結果至新的 CSV 檔案
    df_scaled.to_csv(output_path, index=False)
    print(f"標準化後資料儲存至 {output_path}")


if __name__ == "__main__":
    # 設定絕對路徑（避免不同工作目錄導致找不到檔案）
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(BASE_DIR, "../data/processed/cleaned.csv")
    output_csv = os.path.join(BASE_DIR, "../data/processed/preprocessed.csv")

    # 執行標準化流程
    scale_features(input_csv, output_csv)
