# GNN-Anomaly-Detection

GNN-based IDS using CICIDS2023

# GNN-Anomaly-Detection

Graph Neural Network-based Intrusion Detection System (IDS) using CICIDS2023.  
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
├── README.md # Project documentation
├── requirements.txt # Python dependencies
├── config.yaml # Optional config for training
│
├── data/ # Raw and processed data
│ ├── raw/ # Original CSVs from CICIDS2023
│ ├── processed/ # Graph data in PyG format
│ └── make_graph.py # Script to build graph from raw data
│
├── models/ # GNN models (GCN, GAT, etc.)
│ └── gcn.py # Baseline model: GCN
│
├── train/
│ └── train_gcn.py # Training script for GCN
│
├── experiments/ # Saved models, logs, results
└── utils/
└── metrics.py # Evaluation metrics

## 📚 Dataset

We use the **[CICIDS2023 dataset](https://www.unb.ca/cic/datasets/ids-2023.html)**, which contains labeled real-world network traffic, including benign behavior and multiple attack types (e.g., DDoS, brute force, botnets).

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
