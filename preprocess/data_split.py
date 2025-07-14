# 切分訓練、驗證、測試集

import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split


def load_selected_features(json_path):
    """
    讀取已選好的特徵清單
    """
    with open(json_path, "r") as f:
        selected_features = json.load(f)
    return selected_features


def add_label_binary(df):
    """
    自動加 Label_Binary 欄位
    """
    df['Label_Binary'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    return df


def load_data(preprocessed_path, selected_features):
    """
    讀取處理好的資料，只保留選定的特徵與 Label_Binary 欄位
    """
    df = pd.read_csv(preprocessed_path)

    if 'Label_Binary' not in df.columns:
        df = add_label_binary(df)

    return df[selected_features + ['Label_Binary']]


def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    切分成訓練、驗證、測試集（比例 70/10/20）
    """
    # 先切出測試集
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['Label_Binary'])

    # 再從剩下的切出驗證集
    val_relative_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_relative_size, random_state=random_state, stratify=train_val_df['Label_Binary'])

    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, output_dir):
    """
    存成 train.csv、val.csv、test.csv
    """
    os.makedirs(output_dir, exist_ok=True)
    # 訓練資料（Training Set）→ 模型學習用
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    # 驗證資料（Validation Set）→ 調整超參數用
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    # 測試資料（Test Set）→ 最終評估模型泛化能力用
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"✅ 已將訓練、驗證、測試資料存到：{output_dir}")


def split_data_and_save(preprocessed_path, selected_features_path, output_dir):
    """
    封裝好的主函式：讀入資料 → 加欄位 → 切分資料 → 存檔
    """
    selected_features = load_selected_features(selected_features_path)
    df = load_data(preprocessed_path, selected_features)
    train_df, val_df, test_df = split_data(df)
    print(
        f"📊 切分結果：訓練 {len(train_df)} 筆、驗證 {len(val_df)} 筆、測試 {len(test_df)} 筆")
    save_splits(train_df, val_df, test_df, output_dir)
