# GNN-Anomaly-Detection

GNN-based IDS using CICIDS2017

# GNN-Anomaly-Detection

Graph Neural Network-based Intrusion Detection System (IDS) using CICIDS2017.  
This project aims to detect network anomalies through graph modeling and GNN architectures such as GCN, GAT, and GraphSAGE.

---

## 🧠 Project Description

This project uses Graph Neural Networks to perform anomaly detection on network traffic data.  
Network flows from the CICIDS2023 dataset are modeled as graphs, where:

- **Nodes** represent IPs or hosts
- **Edges** represent communication between them
- **Features** represent statistical or protocol-based attributes
- **Labels** indicate normal vs. malicious behavior

The primary goal is to evaluate the effectiveness of GNNs for intrusion detection tasks.

---

## 📁 Folder Structure

GNN-Anomaly-Detection/
│
├── data/ # 資料目錄
│ ├── raw/ # 原始 CSV（如 CICIDS2017 原始檔）
│ ├── processed/ # 清理與標準化後的資料
│ └── make_graph.py # 將原始資料轉成 PyG graph 格式的腳本
│
├── eda/ # 資料探索（Exploratory Data Analysis）模組
│ └── label_distribution.py # 顯示與分析 Label 分布情況（例如類別不平衡）
│
├── evaluate/ # 評估相關腳本
│ └── metrics.py # 評估指標與工具函式
│
├── experiments/ # 實驗結果、模型檔、訓練紀錄等
│
├── models/ # GNN 模型程式碼
│ ├── gcn.py # 基線模型 GCN
│ ├── gat.py # GAT 模型 (可選)
│ └── graphsage.py # GraphSAGE 模型 (可選)
│
├── preprocess/ # 資料前處理相關腳本
│ ├── data_preview.py # 當前資料匯入測試
│ ├── valuate.py # 模型評估
│ ├── train.py # 模型訓練
│ ├── merge_csvs.py # 合併多天 CSV
│ ├── clean_data.py # 欄位清理、缺值補齊、標籤編碼
│ └── scale_features.py # 數值標準化
│
├── train/ # 訓練相關腳本
│ ├── train_gcn.py # 訓練 GCN 的腳本
│ └── train_gat.py # 訓練 GAT 的腳本 (可選)
│
├── utils/ # 工具函式（資料讀寫、視覺化等）
│ └── helpers.py # 例如資料載入、繪圖函式
│
├── run_preprocessing_pipeline.py # 資料前處理流程總控腳本
├── README.md # 專案說明文件
├── requirements.txt # Python 依賴套件
└── config.yaml # 訓練及參數設定（可選）

## 📚 Dataset

We use the **[CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html)**, which contains labeled real-world network traffic, including benign behavior and multiple attack types (e.g., DDoS, brute force, botnets).

> 💡 You need to request access and download the dataset manually from the official website.

---

## 🧪 Quick Start

1. **Install required packages**

pip install -r requirements.txt

2. **Preprocess the dataset and build the graph**

python data/make_graph.py

3. **Train the GCN baseline mode**

python train/train_gcn.py

4. **Evaluate the model and visualize results**

## Features

⏳ Graph-based anomaly detection

⏳ GCN model using PyTorch & PyTorch Geometric

⏳ Extendable to GAT, GraphSAGE

⏳ Configurable hyperparameters

⏳ Evaluation with accuracy, precision, recall, F1-score

⏳ Logging and results saving

## To Do

Define project structure

Select CICIDS2023 as dataset

Implement graph construction from flows

Build GCN model baseline
ｓａ
Add evaluation metrics

Add GAT & GraphSAGE comparisons

## Add the following to requirements

torch
torch-geometric
pandas
numpy
scikit-learn
pyyaml
💡 Compatible with Python 3.8–3.10

## License

MIT License
© 2025 Alice Lin
