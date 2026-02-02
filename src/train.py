# -*- coding: utf-8 -*-
"""
Training Module - Huấn luyện và đánh giá mô hình MLP
====================================================
Module chứa các hàm huấn luyện và đánh giá mô hình cho bài toán WDBC.

Bao gồm:
- Huấn luyện với sklearn MLPClassifier
- Huấn luyện với MLP tự triển khai
- Đánh giá model với nhiều metrics
- Tạo các biểu đồ cho báo cáo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay
)
from typing import Dict, Optional, Tuple
import pickle
import warnings

from .config import (
    HIDDEN_LAYER_SIZES, ACTIVATION, SOLVER, LEARNING_RATE_INIT,
    MAX_ITER, TOL, RANDOM_STATE, EARLY_STOPPING, VALIDATION_FRACTION,
    N_ITER_NO_CHANGE, MODELS_DIR, REPORTS_DIR, FIGURE_DPI,
    FIGURE_SIZE, FIGURE_SIZE_LARGE, LABEL_NAMES
)

# Suppress warnings
warnings.filterwarnings('ignore')


def create_mlp_sklearn(hidden_layer_sizes: Tuple = HIDDEN_LAYER_SIZES,
                       activation: str = ACTIVATION,
                       solver: str = SOLVER,
                       learning_rate_init: float = LEARNING_RATE_INIT,
                       max_iter: int = MAX_ITER,
                       random_state: int = RANDOM_STATE,
                       early_stopping: bool = EARLY_STOPPING,
                       verbose: bool = True) -> MLPClassifier:
    """
    Tạo mô hình MLP sử dụng sklearn MLPClassifier.
    
    Cấu hình theo slide:
    - Hidden layers: Sigmoid (activation='logistic')
    - Output layer: Softmax (sklearn tự động dùng khi có >=2 classes)
    - Loss: Categorical Cross-Entropy (mặc định của sklearn)
    
    Parameters
    ----------
    hidden_layer_sizes : Tuple
        Số neurons mỗi hidden layer.
    activation : str
        Activation function ('logistic' = Sigmoid).
    solver : str
        Optimizer ('adam' hoặc 'sgd').
    learning_rate_init : float
        Learning rate ban đầu.
    max_iter : int
        Số epoch tối đa.
    random_state : int
        Random seed.
    early_stopping : bool
        Có dùng early stopping không.
    verbose : bool
        In thông tin.
    
    Returns
    -------
    MLPClassifier
        Mô hình MLP chưa huấn luyện.
    """
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,  # 'logistic' = Sigmoid
        solver=solver,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        tol=TOL,
        random_state=random_state,
        early_stopping=early_stopping,
        validation_fraction=VALIDATION_FRACTION if early_stopping else 0.0,
        n_iter_no_change=N_ITER_NO_CHANGE,
        verbose=verbose
    )
    
    print("Đã tạo MLPClassifier với cấu hình:")
    print(f"  - Hidden layers: {hidden_layer_sizes}")
    print(f"  - Activation (hidden): {activation} (Sigmoid)")
    print(f"  - Activation (output): Softmax (tự động)")
    print(f"  - Solver: {solver}")
    print(f"  - Max iterations: {max_iter}")
    print(f"  - Early stopping: {early_stopping}")
    
    return model


def train_model(model, 
                X_train: np.ndarray, 
                y_train: np.ndarray,
                verbose: bool = True) -> MLPClassifier:
    """
    Huấn luyện mô hình MLP.
    
    Parameters
    ----------
    model : MLPClassifier
        Mô hình MLP.
    X_train : np.ndarray
        Dữ liệu huấn luyện (đã chuẩn hóa).
    y_train : np.ndarray
        Nhãn huấn luyện.
    verbose : bool
        In thông tin.
    
    Returns
    -------
    MLPClassifier
        Mô hình đã huấn luyện.
    """
    if verbose:
        print("\nBắt đầu huấn luyện mô hình...")
        print("-" * 50)
    
    model.fit(X_train, y_train)
    
    if verbose:
        print("-" * 50)
        print(f"Huấn luyện hoàn tất!")
        print(f"  - Số epochs thực tế: {model.n_iter_}")
        print(f"  - Loss cuối cùng: {model.loss_:.6f}")
        print(f"  - Số layers: {model.n_layers_}")
    
    return model


def evaluate_model(model, 
                   X_test: np.ndarray, 
                   y_test: np.ndarray,
                   label_names: list = LABEL_NAMES) -> Dict:
    """
    Đánh giá mô hình với nhiều metrics.
    
    Metrics:
    - Accuracy
    - Precision (per class và macro)
    - Recall (per class và macro)
    - F1-score (per class và macro)
    - Confusion matrix
    - ROC-AUC
    
    Parameters
    ----------
    model : MLPClassifier
        Mô hình đã huấn luyện.
    X_test : np.ndarray
        Dữ liệu test.
    y_test : np.ndarray
        Nhãn test.
    label_names : list
        Tên các class.
    
    Returns
    -------
    Dict
        Dictionary chứa tất cả metrics.
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=label_names)
    
    # ROC-AUC (cho class 1 - Benign)
    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    results = {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    
    # Print results
    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")
    print("=" * 60)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    print("=" * 60)
    
    return results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_class_distribution(y: pd.Series, 
                           label_names: list = LABEL_NAMES,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Biểu đồ 1: Phân bố lớp (Class Distribution).
    
    Parameters
    ----------
    y : pd.Series
        Labels.
    label_names : list
        Tên các class.
    save_path : str, optional
        Đường dẫn lưu hình.
    
    Returns
    -------
    plt.Figure
        Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    
    counts = y.value_counts().sort_index()
    colors = ['#ff6b6b', '#4ecdc4']
    
    # Bar chart
    axes[0].bar(label_names, counts.values, color=colors, edgecolor='black')
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Phân bố lớp (Count)', fontsize=14)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha='center', fontsize=12, fontweight='bold')
    
    # Pie chart
    axes[1].pie(counts.values, labels=label_names, autopct='%1.1f%%',
                colors=colors, explode=[0.02, 0.02], shadow=True,
                textprops={'fontsize': 12})
    axes[1].set_title('Phân bố lớp (%)', fontsize=14)
    
    plt.suptitle('Biểu đồ 1: Phân bố lớp trong dataset WDBC', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Đã lưu: {save_path}")
    
    return fig


def plot_correlation_heatmap(X: pd.DataFrame,
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Biểu đồ 2: Ma trận tương quan (Correlation Heatmap).
    
    QUAN TRỌNG: Thể hiện sự hiểu biết về dữ liệu.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features.
    save_path : str, optional
        Đường dẫn lưu hình.
    
    Returns
    -------
    plt.Figure
        Figure object.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    corr_matrix = X.corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r',
                center=0, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.8, 'label': 'Correlation'},
                ax=ax)
    
    ax.set_title('Biểu đồ 2: Ma trận tương quan giữa các đặc trưng\n(Correlation Heatmap)', 
                 fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Đã lưu: {save_path}")
    
    return fig


def plot_feature_boxplots(X: pd.DataFrame, 
                          y: pd.Series,
                          features: list = None,
                          label_names: list = LABEL_NAMES,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Biểu đồ 3: Boxplot các features theo lớp.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features.
    y : pd.Series
        Labels.
    features : list, optional
        Danh sách features cần vẽ. Mặc định lấy 6 features đầu.
    label_names : list
        Tên các class.
    save_path : str, optional
        Đường dẫn lưu hình.
    
    Returns
    -------
    plt.Figure
        Figure object.
    """
    if features is None:
        features = ['radius_mean', 'texture_mean', 'perimeter_mean', 
                    'area_mean', 'smoothness_mean', 'compactness_mean']
    
    # Chỉ lấy các features tồn tại
    features = [f for f in features if f in X.columns][:6]
    
    fig, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE_LARGE)
    axes = axes.flatten()
    
    df_plot = X[features].copy()
    df_plot['diagnosis'] = y.map({0: label_names[0], 1: label_names[1]})
    
    colors = {'Malignant (M)': '#ff6b6b', 'Benign (B)': '#4ecdc4'}
    
    for i, feature in enumerate(features):
        sns.boxplot(x='diagnosis', y=feature, data=df_plot, ax=axes[i],
                   palette=colors)
        axes[i].set_title(feature, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
    
    plt.suptitle('Biểu đồ 3: Phân bố đặc trưng theo lớp (Boxplot)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Đã lưu: {save_path}")
    
    return fig


def plot_normalization_comparison(X_train: np.ndarray,
                                  X_train_scaled: np.ndarray,
                                  feature_names: list,
                                  n_features: int = 5,
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Biểu đồ 5: So sánh trước và sau chuẩn hóa.
    
    Parameters
    ----------
    X_train : np.ndarray
        Dữ liệu trước chuẩn hóa.
    X_train_scaled : np.ndarray
        Dữ liệu sau chuẩn hóa.
    feature_names : list
        Tên các features.
    n_features : int
        Số features để hiển thị.
    save_path : str, optional
        Đường dẫn lưu hình.
    
    Returns
    -------
    plt.Figure
        Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LARGE)
    
    # Trước chuẩn hóa
    df_before = pd.DataFrame(X_train[:, :n_features], columns=feature_names[:n_features])
    axes[0].boxplot(df_before.values, labels=range(1, n_features + 1))
    axes[0].set_title('Trước chuẩn hóa (Original Scale)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Feature Index', fontsize=12)
    axes[0].set_ylabel('Value', fontsize=12)
    
    # Sau chuẩn hóa
    df_after = pd.DataFrame(X_train_scaled[:, :n_features], columns=feature_names[:n_features])
    axes[1].boxplot(df_after.values, labels=range(1, n_features + 1))
    axes[1].set_title('Sau chuẩn hóa Z-Score', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Feature Index', fontsize=12)
    axes[1].set_ylabel('Standardized Value', fontsize=12)
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Mean = 0')
    axes[1].legend()
    
    plt.suptitle('Biểu đồ 5: So sánh scale trước và sau StandardScaler (Z-Score)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Đã lưu: {save_path}")
    
    return fig


def plot_confusion_matrix(cm: np.ndarray,
                         label_names: list = LABEL_NAMES,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Biểu đồ 6: Confusion Matrix.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix.
    label_names : list
        Tên các class.
    save_path : str, optional
        Đường dẫn lưu hình.
    
    Returns
    -------
    plt.Figure
        Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                annot_kws={'size': 20}, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title('Biểu đồ 6: Confusion Matrix', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Đã lưu: {save_path}")
    
    return fig


def plot_learning_curve(loss_curve: list,
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Biểu đồ 7: Learning Curve (Loss theo Epoch).
    
    Parameters
    ----------
    loss_curve : list
        Danh sách loss qua các epoch.
    save_path : str, optional
        Đường dẫn lưu hình.
    
    Returns
    -------
    plt.Figure
        Figure object.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    
    epochs = range(1, len(loss_curve) + 1)
    
    ax.plot(epochs, loss_curve, 'b-', linewidth=2, label='Training Loss')
    ax.fill_between(epochs, loss_curve, alpha=0.3)
    
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss (Categorical Cross-Entropy)', fontsize=14)
    ax.set_title('Biểu đồ 7: Learning Curve - Loss theo Epoch', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Annotate final loss
    ax.annotate(f'Final: {loss_curve[-1]:.4f}',
                xy=(len(loss_curve), loss_curve[-1]),
                xytext=(len(loss_curve) * 0.8, loss_curve[-1] * 1.5),
                fontsize=12,
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Đã lưu: {save_path}")
    
    return fig


def plot_metrics_comparison(results: Dict,
                           label_names: list = LABEL_NAMES,
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Biểu đồ 8: So sánh các metrics.
    
    Parameters
    ----------
    results : Dict
        Kết quả đánh giá từ evaluate_model().
    label_names : list
        Tên các class.
    save_path : str, optional
        Đường dẫn lưu hình.
    
    Returns
    -------
    plt.Figure
        Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LARGE)
    
    # Macro metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [results['accuracy'], results['precision_macro'], 
              results['recall_macro'], results['f1_macro']]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    bars = axes[0].bar(metrics, values, color=colors, edgecolor='black')
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Metrics tổng hợp (Macro Average)', fontsize=14, fontweight='bold')
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.4f}', ha='center', fontsize=11, fontweight='bold')
    
    # Per-class metrics
    x = np.arange(len(label_names))
    width = 0.25
    
    bars1 = axes[1].bar(x - width, results['precision_per_class'], width, 
                        label='Precision', color='#2ecc71')
    bars2 = axes[1].bar(x, results['recall_per_class'], width, 
                        label='Recall', color='#e74c3c')
    bars3 = axes[1].bar(x + width, results['f1_per_class'], width, 
                        label='F1-Score', color='#9b59b6')
    
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(label_names)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Metrics theo từng lớp', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    
    plt.suptitle('Biểu đồ 8: So sánh các Metrics đánh giá', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Đã lưu: {save_path}")
    
    return fig


def plot_roc_curve(fpr: np.ndarray,
                   tpr: np.ndarray,
                   roc_auc: float,
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    Biểu đồ 9: ROC Curve.
    
    Parameters
    ----------
    fpr : np.ndarray
        False Positive Rate.
    tpr : np.ndarray
        True Positive Rate.
    roc_auc : float
        Area Under Curve.
    save_path : str, optional
        Đường dẫn lưu hình.
    
    Returns
    -------
    plt.Figure
        Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, color='#3498db', linewidth=2,
            label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1,
            label='Random classifier')
    ax.fill_between(fpr, tpr, alpha=0.3)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=14)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=14)
    ax.set_title('Biểu đồ 9: ROC Curve và AUC', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Đã lưu: {save_path}")
    
    return fig


def plot_precision_recall_curve(y_test: np.ndarray,
                                y_proba: np.ndarray,
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Biểu đồ 10: Precision-Recall Curve.
    
    Parameters
    ----------
    y_test : np.ndarray
        True labels.
    y_proba : np.ndarray
        Predicted probabilities.
    save_path : str, optional
        Đường dẫn lưu hình.
    
    Returns
    -------
    plt.Figure
        Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
    avg_precision = average_precision_score(y_test, y_proba[:, 1])
    
    ax.plot(recall, precision, color='#2ecc71', linewidth=2,
            label=f'PR curve (AP = {avg_precision:.4f})')
    ax.fill_between(recall, precision, alpha=0.3)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Biểu đồ 10: Precision-Recall Curve', fontsize=16, fontweight='bold')
    ax.legend(loc='lower left', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Đã lưu: {save_path}")
    
    return fig


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(model, filepath: str = None):
    """Lưu model ra file."""
    if filepath is None:
        filepath = MODELS_DIR / 'mlp_model.pkl'
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model đã lưu tại: {filepath}")


def load_model(filepath: str):
    """Load model từ file."""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"Model đã load từ: {filepath}")
    return model
