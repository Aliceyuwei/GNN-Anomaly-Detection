# 資料前處理流程總控腳本
# 執行整個前處理流程：
# 1. 合併多個原始 CSV 檔案
# 2. 資料清理
# 3. 數值標準化
# 4. 特徵選擇（並輸出選定特徵清單 JSON）
# 5. 使用選定特徵切分資料集（訓練/驗證/測試）
#
# 可直接在專案根目錄執行：
# python run_preprocessing_pipeline.py

import os
from preprocess.merge_csvs import merge_csvs
from preprocess.clean_data import clean_data
from preprocess.scale_features import scale_features
from preprocess.feature_selection import run_feature_selection
from preprocess.data_split import split_data_and_save  # 假設你有寫這函式包切分存檔

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 路徑設定
    raw_dir = os.path.join(BASE_DIR, "data/raw")
    merged_csv = os.path.join(BASE_DIR, "data/processed/merged.csv")
    cleaned_csv = os.path.join(BASE_DIR, "data/processed/cleaned.csv")
    preprocessed_csv = os.path.join(
        BASE_DIR, "data/processed/preprocessed.csv")
    selected_feature_json = os.path.join(
        BASE_DIR, "data/processed/selected_features.json")
    output_dir = os.path.join(BASE_DIR, "data/processed")

    # 1. 合併多個 raw CSV 成 merged.csv
    print("Step 1: 合併 CSV 檔案...")
    merge_csvs(raw_dir, merged_csv)

    # 2. 清理資料
    print("Step 2: 清理資料...")
    clean_data(merged_csv, cleaned_csv)

    # 3. 數值標準化
    print("Step 3: 數值標準化...")
    scale_features(cleaned_csv, preprocessed_csv)

    # 4. 特徵選擇
    print("Step 4: 特徵選擇...")
    selected_features = run_feature_selection(
        input_csv_path=preprocessed_csv,
        feature_output_path=selected_feature_json,
        corr_threshold=0.05,
        high_corr_threshold=0.9,
        barplot_output_path=os.path.join(
            BASE_DIR, "data/processed/correlation_bar_filtered.png"),
        heatmap_output_path=os.path.join(
            BASE_DIR, "data/processed/correlation_heatmap_final.png")
    )

    # 5. 切分資料集（train/val/test）
    print("Step 5: 切分資料集...")
    split_data_and_save(preprocessed_csv, selected_feature_json, output_dir)

    print("前處理整合流程完成！")
