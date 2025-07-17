# GNN-Anomaly-Detection

Graph Neural Network-based Intrusion Detection System (IDS) using the CICIDS2017 dataset.  
本專案透過圖神經網路（GNN）模型進行異常偵測，包含 GCN、GAT 與 GraphSAGE 實作。

---

## 🧠 Project Overview

本專案將網路流量資料建模為圖，藉由 GNN 模型偵測異常行為。

- 節點（Node）：IP 或主機
- 邊（Edge）：彼此之間的通訊行為
- 特徵（Feature）：網路封包統計與協定資訊
- 標籤（Label）：正常流量 (BENIGN) 或攻擊行為

我們使用 CICIDS2017 作為資料來源，並針對其連續天的原始流量進行資料清理、標準化、建圖與特徵選擇等處理。

---

## 🚀 Quick Start

### 1. 安裝相依套件

```
pip install -r requirements.txt
```

### 2. 放置原始資料

下載並解壓 [CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html) 到：

```
data/raw/
```

建議命名為：

```
data/raw/Wednesday-workingHours.pcap_ISCX.csv
data/raw/Thursday-workingHours-Morning-WebAttacks.pcap_ISCX.csv
...
```

### 3. 執行資料預處理（含版本備份）

```
python run_preprocessing_pipeline.py
```

會完成以下動作：

- 合併多日原始 CSV
- 清理與補齊欄位
- 編碼標籤、標準化數值欄位
- 特徵篩選與視覺化
- 自動備份處理後資料與圖表至 `data/versions/` 資料夾

範例輸出路徑：

```
data/versions/20250717_1745_preproc/
├── merged.csv
├── cleaned.csv
├── preprocessed.csv
├── selected_features.json
├── figures/
│   ├── correlation_bar_filtered.png
│   └── correlation_heatmap_final.png
├── version.json
└── README.md
```

### 4. 訓練模型

```
python train/train_gcn.py
```

或切換至其他模型：

```
python train/train_gat.py
```

訓練設定可透過 `config.yaml` 調整。

---

## 🧪 Features

- ✅ Graph-based anomaly detection
- ✅ 多種 GNN 模型：GCN / GAT / GraphSAGE
- ✅ 自動化預處理流程與版本控制
- ✅ 特徵篩選視覺化（熱圖與條狀圖）
- ✅ 評估指標完整：Accuracy, F1, Recall, Precision
- ✅ 完整模組化程式架構（EDA / Preprocess / Training / Evaluation）

---

## 📁 Folder Structure

```
GNN-Anomaly-Detection/
│
├── data/                      # 資料目錄
│   ├── raw/                   # 原始 CSV（如 CICIDS2017 原始檔）
│   ├── processed/             # 最新一輪清理與標準化後的資料
│   └── versions/              # 每次預處理流程的版本化備份（含 metadata、圖表等）
│       └── 20250717_1745_preproc/
│           ├── merged.csv
│           ├── cleaned.csv
│           ├── preprocessed.csv
│           ├── selected_features.json
│           ├── figures/      # 相關圖表（如熱圖、條狀圖）
│           ├── version.json  # 預處理參數與摘要 metadata
│           └── README.md     # 說明此版本目的與內容
│
├── eda/                      # 資料探索（EDA）模組
│   └── label_distribution.py
│
├── evaluate/                 # 模型評估工具
│   └── metrics.py
│
├── experiments/              # 實驗記錄、模型快照等
│
├── models/                   # GNN 模型定義
│   ├── gcn.py
│   ├── gat.py
│   └── graphsage.py
│
├── preprocess/               # 前處理模組
│   ├── merge_csvs.py
│   ├── clean_data.py
│   ├── scale_features.py
│   ├── feature_selection.py
│   ├── plot_correlation.py   # 視覺化繪圖模組（條狀圖、熱圖）
│   ├── data_split.py
│   └── evaluate.py           # 評估用（如訓練集 / 測試集分布）
│
├── train/                    # 訓練腳本
│   ├── train_gcn.py
│   └── train_gat.py
│
├── utils/                    # 工具函式（通用工具）
│   └── helpers.py
│
├── run_preprocessing_pipeline.py  # 預處理流程主控腳本（含版本備份、自動儲存圖與 JSON）
├── requirements.txt
├── config.yaml               # 超參與訓練設定（可選）
└── README.md

```

---

## 📚 Dataset

使用 [CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html)，包含正常與多種網路攻擊（如 DoS、Brute Force、Botnet 等）。

⚠️ 注意：需至官方網站註冊帳號取得下載權限。

---

## 🔧 Requirements

- Python 3.8–3.10
- torch
- torch-geometric
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- pyyaml

---

## 📌 TODO

- [x] 完成 GCN 基線模型
- [x] 加入特徵篩選與視覺化
- [x] 預處理流程版本化（含 json + 圖表備份）
- [ ] 加入 GAT 與 GraphSAGE 支援
- [ ] 支援多日訓練與跨日測試實驗

---

## 📄 License

MIT License
© 2025 Alice Lin
