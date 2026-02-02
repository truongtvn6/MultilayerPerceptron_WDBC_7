# -*- coding: utf-8 -*-
"""
Data Preprocessing Module - Tiền xử lý dữ liệu WDBC
===================================================
Module chứa các hàm tiền xử lý dữ liệu cho bài toán phân loại WDBC.

Pipeline tiền xử lý:
1. Load dữ liệu từ CSV
2. Loại bỏ cột nhiễu (id, Unnamed: 32)
3. Mã hóa nhãn (M → 0, B → 1)
4. Tách features và labels
5. Chia train/test
6. Chuẩn hóa Z-Score (StandardScaler)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

from .config import (
    DATA_PATH, 
    COLUMNS_TO_DROP, 
    LABEL_MAPPING, 
    TEST_SIZE, 
    RANDOM_STATE
)


def load_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load dữ liệu WDBC từ file CSV.
    
    Parameters
    ----------
    filepath : str, optional
        Đường dẫn file CSV. Mặc định dùng DATA_PATH từ config.
    
    Returns
    -------
    pd.DataFrame
        DataFrame chứa dữ liệu WDBC.
    """
    if filepath is None:
        filepath = DATA_PATH
    
    df = pd.read_csv(filepath)
    print(f"Đã load dữ liệu: {df.shape[0]} mẫu, {df.shape[1]} cột")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch dữ liệu: loại bỏ cột nhiễu.
    
    Loại bỏ:
    - Cột 'id': không phải feature
    - Cột 'Unnamed: 32': cột rỗng/NaN do format CSV
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame gốc.
    
    Returns
    -------
    pd.DataFrame
        DataFrame đã làm sạch.
    """
    df_clean = df.copy()
    
    # Drop các cột không cần thiết
    cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df_clean.columns]
    if cols_to_drop:
        df_clean.drop(columns=cols_to_drop, inplace=True)
        print(f"Đã loại bỏ các cột: {cols_to_drop}")
    
    # Kiểm tra và loại bỏ cột rỗng (tên rỗng hoặc toàn NaN)
    empty_cols = [col for col in df_clean.columns if col == '' or df_clean[col].isna().all()]
    if empty_cols:
        df_clean.drop(columns=empty_cols, inplace=True)
        print(f"Đã loại bỏ cột rỗng: {empty_cols}")
    
    print(f"Shape sau khi làm sạch: {df_clean.shape}")
    return df_clean


def encode_labels(df: pd.DataFrame, column: str = 'diagnosis') -> pd.DataFrame:
    """
    Mã hóa nhãn diagnosis: M → 0 (Malignant), B → 1 (Benign).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame chứa cột diagnosis.
    column : str
        Tên cột nhãn.
    
    Returns
    -------
    pd.DataFrame
        DataFrame với nhãn đã mã hóa.
    """
    df_encoded = df.copy()
    df_encoded[column] = df_encoded[column].map(LABEL_MAPPING)
    
    # Kiểm tra mã hóa thành công
    unique_labels = df_encoded[column].unique()
    print(f"Nhãn sau mã hóa: {unique_labels}")
    print(f"Phân bố nhãn:\n{df_encoded[column].value_counts().sort_index()}")
    
    return df_encoded


def split_features_labels(df: pd.DataFrame, 
                          label_column: str = 'diagnosis') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Tách features (X) và labels (y).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã tiền xử lý.
    label_column : str
        Tên cột nhãn.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X (features), y (labels)
    """
    X = df.drop(columns=[label_column])
    y = df[label_column]
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Danh sách {X.shape[1]} features: {list(X.columns)}")
    
    return X, y


def split_train_test(X: pd.DataFrame, 
                     y: pd.Series,
                     test_size: float = TEST_SIZE,
                     random_state: int = RANDOM_STATE) -> Tuple:
    """
    Chia dữ liệu thành tập huấn luyện và kiểm tra.
    
    Sử dụng stratify để đảm bảo phân bố nhãn cân bằng.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Labels.
    test_size : float
        Tỉ lệ test set (mặc định 0.2 = 20%).
    random_state : int
        Random seed để tái lặp.
    
    Returns
    -------
    Tuple
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Đảm bảo tỉ lệ nhãn cân bằng giữa train/test
    )
    
    print(f"Training set: {X_train.shape[0]} mẫu")
    print(f"Test set: {X_test.shape[0]} mẫu")
    
    return X_train, X_test, y_train, y_test


def normalize_zscore(X_train: np.ndarray, 
                     X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Chuẩn hóa dữ liệu bằng Z-Score (StandardScaler).
    
    Công thức: z = (x - μ) / σ
    - μ: mean (trung bình)
    - σ: standard deviation (độ lệch chuẩn)
    
    LƯU Ý: Fit scaler trên X_train, sau đó transform cả X_train và X_test
    để tránh data leakage.
    
    Parameters
    ----------
    X_train : np.ndarray
        Dữ liệu huấn luyện.
    X_test : np.ndarray
        Dữ liệu kiểm tra.
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, StandardScaler]
        X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()
    
    # Fit trên train, transform cả train và test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Đã chuẩn hóa Z-Score:")
    print(f"  - Mean sau chuẩn hóa (train): {X_train_scaled.mean():.6f}")
    print(f"  - Std sau chuẩn hóa (train): {X_train_scaled.std():.6f}")
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(filepath: Optional[str] = None) -> dict:
    """
    Pipeline tiền xử lý đầy đủ từ file CSV đến dữ liệu sẵn sàng cho training.
    
    Các bước:
    1. Load dữ liệu
    2. Làm sạch (drop cột nhiễu)
    3. Mã hóa nhãn
    4. Tách features/labels
    5. Chia train/test
    6. Chuẩn hóa Z-Score
    
    Parameters
    ----------
    filepath : str, optional
        Đường dẫn file CSV.
    
    Returns
    -------
    dict
        Dictionary chứa tất cả dữ liệu đã xử lý.
    """
    print("=" * 60)
    print("PIPELINE TIỀN XỬ LÝ DỮ LIỆU WDBC")
    print("=" * 60)
    
    # Bước 1: Load
    print("\n[1/6] Load dữ liệu...")
    df = load_data(filepath)
    
    # Bước 2: Clean
    print("\n[2/6] Làm sạch dữ liệu...")
    df = clean_data(df)
    
    # Bước 3: Encode
    print("\n[3/6] Mã hóa nhãn...")
    df = encode_labels(df)
    
    # Bước 4: Split features/labels
    print("\n[4/6] Tách features và labels...")
    X, y = split_features_labels(df)
    
    # Bước 5: Train/test split
    print("\n[5/6] Chia train/test...")
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    
    # Bước 6: Normalize
    print("\n[6/6] Chuẩn hóa Z-Score...")
    X_train_scaled, X_test_scaled, scaler = normalize_zscore(X_train, X_test)
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH TIỀN XỬ LÝ!")
    print("=" * 60)
    
    return {
        'df_original': df,
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'feature_names': list(X.columns)
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Tạo tóm tắt thống kê về dữ liệu.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame cần tóm tắt.
    
    Returns
    -------
    dict
        Dictionary chứa các thông tin tóm tắt.
    """
    summary = {
        'n_samples': df.shape[0],
        'n_features': df.shape[1] - 1,  # Trừ cột diagnosis
        'n_missing': df.isnull().sum().sum(),
        'class_distribution': df['diagnosis'].value_counts().to_dict() if 'diagnosis' in df.columns else None,
        'dtypes': df.dtypes.to_dict()
    }
    return summary


def check_data_quality(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Kiểm tra chất lượng dữ liệu.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Labels.
    
    Returns
    -------
    dict
        Báo cáo chất lượng dữ liệu.
    """
    report = {
        'missing_features': X.isnull().sum().sum(),
        'missing_labels': y.isnull().sum(),
        'duplicate_rows': X.duplicated().sum(),
        'infinite_values': np.isinf(X.values).sum() if np.issubdtype(X.values.dtype, np.number) else 0,
        'class_balance': y.value_counts(normalize=True).to_dict()
    }
    
    print("\n--- Báo cáo chất lượng dữ liệu ---")
    print(f"Missing values (features): {report['missing_features']}")
    print(f"Missing values (labels): {report['missing_labels']}")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print(f"Infinite values: {report['infinite_values']}")
    print(f"Class balance: {report['class_balance']}")
    
    return report
