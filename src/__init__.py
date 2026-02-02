# -*- coding: utf-8 -*-
"""
MLP WDBC Classification Package
================================
Package chứa các module cho dự án phân loại ung thư WDBC sử dụng Multilayer Perceptron.

Modules:
    - config: Cấu hình hyperparameters và đường dẫn
    - data_preprocessing: Tiền xử lý dữ liệu (load, clean, normalize)
    - mlp_scratch: Triển khai MLP từ đầu
    - train: Huấn luyện mô hình
"""

from .config import *
from .data_preprocessing import *
from .mlp_scratch import MLPScratch
from .train import *

__version__ = "1.0.0"
__author__ = "WDBC Project"
