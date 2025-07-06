# 資料清理、缺值處理、標籤編碼、標準化

import os  # 匯入 os 模組，用來處理檔案與路徑相關操作
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(input_path, output_path):
    """
    資料前處理主函式
    參數:
        input_path: str, 合併後的 CSV 路徑
        output_path: str, 處理完成後資料的儲存路徑
    """

    # 1. 讀取資料
    df = pd.read_csv(input_path)
    print(f"讀取資料，shape: {df.shape}")

    # 2. 欄位挑選（可根據需要調整，這裡保留全部數值欄位與 Label）
    # selected_columns = [ 'Destination Port', 'Flow Duration', ..., 'Label']
    # df = df[selected_columns]

    # 3. 缺值處理
    missing_ratio = df.isnull().sum() / len(df)
    print("各欄位缺值比例:")
    print(missing_ratio[missing_ratio > 0])

    # 先補齊每個數值欄位的缺值（中位數）
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"{col} 補中位數 {median_val}")

    # 補完缺值後，統一處理 Inf/-Inf 並丟棄 NaN（包含之前未補的非數值欄位缺值）
    inf_count = ((df == float('inf')) | (df == float('-inf'))).sum().sum()
    print(f"含有 Inf 或 -Inf 的值共 {inf_count} 個")
    df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

    before_drop = len(df)
    df.dropna(inplace=True)
    after_drop = len(df)
    print(f"已清理 Inf/NaN，移除資料筆數: {before_drop - after_drop}")

    # 斷言：最後安全檢查
    assert df.isnull().sum().sum() == 0, "尚有 NaN 存在"
    assert not df.isin([float('inf'), float('-inf')]).values.any(), "尚有 Inf 存在"
    print("確認無 NaN / Inf")

    # 4. 標籤編碼(Label Encoding)
    label_encoder = LabelEncoder()
    df['Label_enc'] = label_encoder.fit_transform(df['Label'])
    print("標籤編碼對照表:")
    print(dict(zip(label_encoder.classes_,
          label_encoder.transform(label_encoder.classes_))))

    # 5. 數值標準化
    # 除了 Label 和 Label_enc，其他數值欄位做標準化
    features = df.drop(columns=['Label', 'Label_enc'])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(features_scaled, columns=features.columns)

    # 把標籤欄位加回去
    df_scaled['Label_enc'] = df['Label_enc'].values

    # 6. 儲存前處理後的資料
    df_scaled.to_csv(output_path, index=False)
    print(f"處理完成資料已儲存至 {output_path}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(BASE_DIR, "../data/processed/merged.csv")
    output_csv = os.path.join(BASE_DIR, "../data/processed/preprocessed.csv")
    preprocess_data(input_csv, output_csv)
