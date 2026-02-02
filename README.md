# Áp dụng Multilayer Perceptron (MLP) để phân loại ung thư trong tập dữ liệu WDBC

## Mô tả đề tài

Dự án này triển khai mạng nơ-ron **Multilayer Perceptron (MLP)** để phân loại khối u **lành tính (Benign)** và **ác tính (Malignant)** từ dữ liệu **Wisconsin Diagnostic Breast Cancer (WDBC)**.

### Thông tin Dataset

- **Nguồn:** Wisconsin Diagnostic Breast Cancer Dataset
- **Số mẫu:** 569
- **Số đặc trưng:** 30 features (các đặc điểm được tính từ hình ảnh số hóa của khối u)
- **Số lớp:** 2 (Malignant - Ác tính, Benign - Lành tính)

### Mục tiêu

- Xây dựng mô hình MLP phân loại chính xác hai loại khối u
- Triển khai theo đúng Training Pipeline từ slide MLP (GV Hoang Duc Quy)
- Tạo các biểu đồ trực quan cho báo cáo

---

## Cấu trúc thư mục

```
MultilayerPerceptron_WDBC_7/
├── data/
│   └── data.csv                 # WDBC dataset
├── notebooks/
│   └── wdbc_mlp_classification.ipynb   # Notebook chính (pipeline MLP hoàn chỉnh)
├── src/                         # Module tái sử dụng
│   ├── __init__.py
│   ├── config.py                # Cấu hình và hyperparameters
│   ├── data_preprocessing.py    # Tiền xử lý dữ liệu
│   ├── mlp_scratch.py           # MLP triển khai từ đầu
│   └── train.py                 # Huấn luyện và đánh giá
├── models/                      # Model đã train (.pkl)
├── reports/                     # Biểu đồ xuất cho báo cáo (PNG)
├── requirements.txt             # Dependencies
├── README.md                    # File này
└── WDBC.csv                     # Dataset gốc
```

---

## Training Pipeline (theo Slide MLP)

1. **Dataset curation** - Chuẩn bị dữ liệu: load, drop cột nhiễu, mã hóa nhãn
2. **Data normalization** - Chuẩn hóa Z-Score (StandardScaler)
3. **Build model** - Xây dựng MLP với Sigmoid (hidden) + Softmax (output)
4. **Optimizer selection** - Chọn optimizer Adam/SGD
5. **Parameter Initialization** - Khởi tạo tham số (Xavier)
6. **Metrics/Loss selection** - Categorical Cross-Entropy

---

## Kiến trúc MLP

```
Input Layer (30 neurons) → Hidden Layer 1 (64, Sigmoid) → Hidden Layer 2 (32, Sigmoid) → Output Layer (2, Softmax)
```

### Công thức toán học

| Thành phần | Công thức | Implementation |
|------------|-----------|----------------|
| **Chuẩn hóa** | z = (x - μ) / σ | `StandardScaler()` |
| **Activation Hidden** | g(z) = 1 / (1 + e^(-z)) | `activation='logistic'` |
| **Activation Output** | g(zⱼ) = e^(zⱼ) / Σₖ e^(zₖ) | Softmax (tự động) |
| **Loss** | L = -Σⱼ yⱼ log(ŷⱼ) | Categorical Cross-Entropy |

---

## Biểu đồ tạo ra

1. **Phân bố lớp** - Bar chart & Pie chart
2. **Ma trận tương quan** - Correlation Heatmap (QUAN TRỌNG)
3. **Boxplot đặc trưng** - So sánh features theo lớp
4. **Histogram đặc trưng** - Phân bố features theo lớp
5. **So sánh chuẩn hóa** - Trước/sau StandardScaler
6. **Confusion Matrix** - Ma trận nhầm lẫn
7. **Learning Curve** - Loss theo epoch
8. **So sánh Metrics** - Accuracy, Precision, Recall, F1
9. **ROC Curve** - Với AUC score
10. **Precision-Recall Curve** - Với Average Precision

---

## Cách chạy

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Chạy notebook

```bash
cd notebooks
jupyter notebook wdbc_mlp_classification.ipynb
```

### 3. Hoặc sử dụng modules Python

```python
from src.data_preprocessing import preprocess_pipeline
from src.train import create_mlp_sklearn, train_model, evaluate_model

# Tiền xử lý
data = preprocess_pipeline()

# Tạo và huấn luyện model
model = create_mlp_sklearn()
model = train_model(model, data['X_train_scaled'], data['y_train'])

# Đánh giá
results = evaluate_model(model, data['X_test_scaled'], data['y_test'])
```

---

## Kết quả

Mô hình MLP đạt được kết quả tốt trên tập test:

- **Accuracy:** ~96-98%
- **Precision:** ~0.96
- **Recall:** ~0.96
- **F1-Score:** ~0.96
- **ROC-AUC:** ~0.99

---

## Dependencies

- Python >= 3.8
- pandas >= 1.3.0
- numpy >= 1.21.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 1.0.0
- jupyter >= 1.0.0

---

## Tài liệu tham khảo

- Slide MLP - GV Hoang Duc Quy (AI & Applications)
- UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic)
- Scikit-learn Documentation - MLPClassifier

---

## License

MIT License
