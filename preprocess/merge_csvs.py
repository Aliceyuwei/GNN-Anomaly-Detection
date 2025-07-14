# 負責合併多天 CSV 檔

# 載入 pandas 套件，這是 Python 中最常用的資料處理工具（用來讀 .csv、做分析、表格處理）
import pandas as pd
# 載入 os 模組，主要用來處理 檔案與路徑的操作（例如：組路徑、取得目前資料夾位置等）
import os
# 載入 os 模組，主要用來處理 檔案與路徑的操作（例如：組路徑、取得目前資料夾位置等）
from glob import glob


def merge_csvs(data_folder="data/raw", output_path="data/processed/merged.csv"):
    """
    合併多個 CSV 檔案成一個，並儲存至指定路徑
    """

    # 放寬匹配條件：符合含有 WorkingHours 的所有檔案
    csv_files = sorted(
        glob(os.path.join(data_folder, "*WorkingHours*.pcap_ISCX.csv")))

    # 印出找到的檔案
    print("找到的 CSV 檔案：")
    for file in csv_files:
        print(" -", file)

    # 建立空的 DataFrame
    all_data = []

    # 讀取每一個 CSV 並加到 all_data 裡
    for file in csv_files:
        print(f"讀取中：{file}")
        df = pd.read_csv(file, encoding='ISO-8859-1')

        # 移除所有欄位名稱的前後空格
        df.columns = df.columns.str.strip()

        # 檢查是否有 'Label' 欄位
        if 'Label' not in df.columns:
            print(f"❗ 檔案 {file} 缺少 Label 欄位，跳過")
            continue

        all_data.append(df)

    # 合併所有 DataFrame
    merged_df = pd.concat(all_data, ignore_index=True)

    # 顯示資料量與標籤分佈
    print("\n✅ 合併完成")
    print("總筆數：", len(merged_df))
    print("標籤分佈：")
    print(merged_df['Label'].value_counts())
    # print("所有欄位：")
    # print(merged_df.columns.tolist())
    # print("看前五筆資料")
    # print(merged_df.head())

   # 存檔到 data/processed 資料夾
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    print(f"已將合併後資料存成 {output_path}")
