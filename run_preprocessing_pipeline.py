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
import shutil
import json
import time
from datetime import datetime
import pandas as pd
from preprocess.merge_csvs import merge_csvs
from preprocess.clean_data import clean_data
from preprocess.scale_features import scale_features
from preprocess.feature_selection import run_feature_selection
from preprocess.data_split import split_data_and_save

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 設定處理與備份參數
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    processed_dir = os.path.join(BASE_DIR, "data/processed")
    raw_dir = os.path.join(BASE_DIR, "data/raw")
    version_dir = os.path.join(
        BASE_DIR, "data/versions", f"{timestamp}_preproc")
    corr_threshold = 0.05
    high_corr_threshold = 0.9

    # 圖表儲存位置
    figures_dir = os.path.join(processed_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    barplot_path = os.path.join(figures_dir, "correlation_bar_filtered.png")
    heatmap_path = os.path.join(figures_dir, "correlation_heatmap_final.png")

    # 檔案路徑
    merged_csv = os.path.join(processed_dir, "merged.csv")
    cleaned_csv = os.path.join(processed_dir, "cleaned.csv")
    preprocessed_csv = os.path.join(processed_dir, "preprocessed.csv")
    selected_feature_json = os.path.join(
        processed_dir, "selected_features.json")

    # 紀錄預處理所耗費時間
    start_time = time.time()

    # 1. 合併多個 raw CSV 成 merged.csv
    print("Step 1: 合併 CSV 檔案...")
    merge_csvs(raw_dir, merged_csv)
    merged_rows = len(pd.read_csv(merged_csv))

    # 2. 清理資料
    print("Step 2: 清理資料...")
    clean_data(merged_csv, cleaned_csv)

    # 3. 數值標準化
    print("Step 3: 數值標準化...")
    scale_features(cleaned_csv, preprocessed_csv)

    # 4. 特徵選擇
    print("Step 4: 特徵選擇...")
    selected_features, corr_matrix = run_feature_selection(
        input_csv_path=preprocessed_csv,
        feature_output_path=selected_feature_json,
        corr_threshold=corr_threshold,
        high_corr_threshold=high_corr_threshold,
        barplot_output_path=barplot_path,
        heatmap_output_path=heatmap_path
    )

    # 5. 切分資料集（train/val/test）
    print("Step 5: 切分資料集...")
    split_data_and_save(preprocessed_csv, selected_feature_json, processed_dir)

    # 計時結束
    end_time = time.time()
    elapsed_seconds = round(end_time - start_time, 2)

    # 6 備份到版本資料夾（含 figures）
    print(f"🗂️ 備份處理後資料到 {version_dir}")
    shutil.copytree(processed_dir, version_dir)

    # 7 寫入 version.json
    version_info = {
        "timestamp": timestamp,
        "merged_rows": merged_rows,
        "selected_features": selected_features,
        "corr_threshold": corr_threshold,
        "high_corr_threshold": high_corr_threshold,
        "elapsed_time_seconds": elapsed_seconds,
        "figures": {
            "barplot": "figures/correlation_bar_filtered.png",
            "heatmap": "figures/correlation_heatmap_final.png"
        }
    }
    with open(os.path.join(version_dir, "version.json"), "w") as f:
        json.dump(version_info, f, indent=2)

    print(f"✅ 前處理流程完成！共耗時 {elapsed_seconds} 秒，version.json 已建立")
