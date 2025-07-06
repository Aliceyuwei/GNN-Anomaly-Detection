# 顯示與分析 Label 分布情況（例如類別不平衡）

import matplotlib.pyplot as plt   # 匯入繪圖套件 matplotlib.pyplot，方便畫圖
import matplotlib                 # 匯入 matplotlib，用來設定字體或其他全域參數
import pandas as pd               # 處理表格資料
import matplotlib.pyplot as plt   # 繪製圖表
import seaborn as sns             # 美化統計圖表
import os                         # 處理檔案路徑

# 設定字體（以下示範 macOS 系統常用的蘋果系字體）
matplotlib.rcParams['font.family'] = [
    'Apple LiGothic', 'Arial Unicode MS', 'DejaVu Sans']


def analyze_label_distribution(input_path, output_path=None):
    """
    分析 CSV 檔案中的 Label 分布情況：
    - 印出各類別的數量與比例
    - 畫出 Label 分布柱狀圖
    - 指定 output_path，則儲存圖片檔
    參數:
        input_path: str, 清理好的 CSV 檔案路徑（需含 'Label' 欄位）
        output_path: str or None, 圖片儲存路徑（預設 None，不儲存）
    """

    # 讀取資料
    df = pd.read_csv(input_path)

    # 顯示各類別的樣本數量
    print("Label 數量統計:")
    print(df['Label'].value_counts())
    # 顯示各類別的樣本比例（%）
    print("Label 比例 (%):")
    print(df['Label'].value_counts(normalize=True) * 100)

    # 繪製類別分布圖
    plt.figure(figsize=(10, 5))
    sns.countplot(
        data=df,
        x='Label',
        order=df['Label'].value_counts().index  # 依照數量排序
    )
    plt.title("Label 分布圖")
    plt.xticks(rotation=45)  # 旋轉 X 軸文字避免重疊
    plt.tight_layout()

    if output_path:
        # 確保資料夾存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Label 分布圖已儲存至: {output_path}")

    plt.show()


if __name__ == "__main__":
    # 取得目前執行的 .py 檔案的絕對路徑，並取得該檔案所在資料夾路徑
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # 設定輸入檔案路徑：往上兩層進入 data/processed 資料夾，讀取 cleaned.csv
    input_csv = os.path.join(BASE_DIR, "../data/processed/cleaned.csv")
    # 設定輸出圖片路徑：同樣在 data/processed，命名為 label_distribution.png
    output_png = os.path.join(
        BASE_DIR, "../data/processed/label_distribution.png")
    # 印出圖片輸出路徑，方便確認儲存位置
    print(f"輸出圖片路徑: {output_png}")

    # 呼叫自訂義的分析函式 analyze_label_distribution，
    # 傳入輸入檔案路徑和圖片輸出路徑
    # 函式會分析 Label 分布、繪圖，並將圖片存至 output_png
    analyze_label_distribution(input_csv, output_png)
