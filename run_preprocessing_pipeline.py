# 資料前處理流程總控腳本
# 執行整個前處理流程（清理 + 標準化）
# 可直接在專案根目錄執行：python run_preprocessing_pipeline.py

import os
from preprocess.clean_data import clean_data
from preprocess.scale_features import scale_features

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_csv = os.path.join(BASE_DIR, "data/processed/merged.csv")
    cleaned_csv = os.path.join(BASE_DIR, "data/processed/cleaned.csv")
    preprocessed_csv = os.path.join(
        BASE_DIR, "data/processed/preprocessed.csv")

    print("Step 1: 清理資料...")
    clean_data(input_csv, cleaned_csv)

    print("Step 2: 數值標準化...")
    scale_features(cleaned_csv, preprocessed_csv)

    print("前處理流程完成！")
