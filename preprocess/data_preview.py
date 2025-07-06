import pandas as pd
import os


# 查看當前工作目錄，確認是否在正確位置
print("目前位置：", os.getcwd())

# 用相對路徑讀取資料
# df = pd.read_csv(
#     "data/raw/Monday-WorkingHours.pcap_ISCX.csv", encoding="ISO-8859-1")
df = pd.read_csv("data/raw/Monday-WorkingHours.pcap_ISCX.csv",
                 encoding="ISO-8859-1", skipinitialspace=True)


# 看前幾筆資料
print(df.head())

# 看有哪些欄位
print(df.columns)

# 看分類標籤分布
print(df['Label'].value_counts())
