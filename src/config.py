# -*- coding: utf-8 -*-
"""
Config Module - Cấu hình dự án MLP WDBC
=======================================
Chứa các hằng số, đường dẫn và hyperparameters cho dự án.
"""

import os
from pathlib import Path

# =============================================================================
# ĐƯỜNG DẪN
# =============================================================================
# Thư mục gốc của dự án
PROJECT_ROOT = Path(__file__).parent.parent

# Đường dẫn dữ liệu
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "data.csv"

# Đường dẫn lưu model
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "mlp_model.pkl"

# Đường dẫn lưu báo cáo/biểu đồ
REPORTS_DIR = PROJECT_ROOT / "reports"

# =============================================================================
# THAM SỐ DỮ LIỆU
# =============================================================================
# Random state để tái lặp kết quả
RANDOM_STATE = 42

# Tỉ lệ test set
TEST_SIZE = 0.2

# Cột cần loại bỏ (nhiễu)
COLUMNS_TO_DROP = ['id', 'Unnamed: 32']

# Ánh xạ nhãn: M (Malignant) = 0, B (Benign) = 1
LABEL_MAPPING = {'M': 0, 'B': 1}
LABEL_NAMES = ['Malignant (M)', 'Benign (B)']

# =============================================================================
# HYPERPARAMETERS MLP
# =============================================================================
# Cấu trúc mạng: số neurons mỗi hidden layer
# VD: (64, 32) = 2 hidden layers với 64 và 32 neurons
HIDDEN_LAYER_SIZES = (64, 32)

# Activation function cho hidden layers
# 'logistic' = Sigmoid: g(z) = 1 / (1 + e^(-z))
# (sklearn tự động dùng Softmax cho output layer khi có >=2 classes)
ACTIVATION = 'logistic'

# Optimizer
# 'adam': Adam optimizer (adaptive learning rate)
# 'sgd': Stochastic Gradient Descent
SOLVER = 'adam'

# Learning rate (cho SGD)
LEARNING_RATE_INIT = 0.001

# Số epoch tối đa
MAX_ITER = 500

# Ngưỡng hội tụ (tolerance)
TOL = 1e-4

# Batch size (cho SGD/Adam)
BATCH_SIZE = 'auto'

# Early stopping
EARLY_STOPPING = True
VALIDATION_FRACTION = 0.1
N_ITER_NO_CHANGE = 10

# =============================================================================
# CẤU HÌNH BIỂU ĐỒ
# =============================================================================
# DPI cho hình ảnh xuất ra
FIGURE_DPI = 150

# Kích thước mặc định
FIGURE_SIZE = (10, 6)
FIGURE_SIZE_LARGE = (12, 8)
FIGURE_SIZE_HEATMAP = (14, 12)

# Style
PLOT_STYLE = 'seaborn-v0_8-whitegrid'

# =============================================================================
# THÔNG TIN MÔ HÌNH
# =============================================================================
MODEL_INFO = {
    'name': 'Multilayer Perceptron (MLP)',
    'input_neurons': 30,  # 30 features từ WDBC dataset
    'hidden_layers': HIDDEN_LAYER_SIZES,
    'output_neurons': 2,  # 2 classes: Malignant, Benign
    'activation_hidden': 'Sigmoid',
    'activation_output': 'Softmax',
    'loss_function': 'Categorical Cross-Entropy',
    'normalization': 'Z-Score (StandardScaler)'
}

# =============================================================================
# CÔNG THỨC TOÁN HỌC (cho documentation)
# =============================================================================
FORMULAS = {
    'z_score': 'z = (x - μ) / σ',
    'sigmoid': 'g(z) = 1 / (1 + e^(-z))',
    'softmax': 'g(z_j) = e^(z_j) / Σ_k e^(z_k)',
    'categorical_ce': 'L = -Σ_j y_j * log(ŷ_j)'
}
