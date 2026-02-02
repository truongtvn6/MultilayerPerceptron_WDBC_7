# -*- coding: utf-8 -*-
"""
MLP Scratch Module - Triển khai Multilayer Perceptron từ đầu
============================================================
Module triển khai MLP từ đầu (không dùng sklearn) theo slide của GV Hoang Duc Quy.

Cấu trúc mạng:
- Input layer: 30 neurons (30 features WDBC)
- Hidden layer(s): Sigmoid activation
- Output layer: 2 neurons, Softmax activation (one-hot encoding)

Loss function: Categorical Cross-Entropy
Optimizer: Gradient Descent / SGD
"""

import numpy as np
from typing import List, Tuple, Optional
import pickle


class MLPScratch:
    """
    Multilayer Perceptron triển khai từ đầu.
    
    Theo slide MLP (GV Hoang Duc Quy):
    - Hidden layers: Sigmoid activation
    - Output layer: Softmax activation
    - Loss: Categorical Cross-Entropy
    
    Parameters
    ----------
    layer_sizes : List[int]
        Số neurons mỗi layer, bao gồm input, hidden(s), và output.
        VD: [30, 64, 32, 2] = 30 input, 2 hidden (64, 32), 2 output
    learning_rate : float
        Tốc độ học (learning rate).
    max_iter : int
        Số epoch tối đa.
    tol : float
        Ngưỡng hội tụ.
    random_state : int
        Random seed.
    verbose : bool
        In thông tin trong quá trình huấn luyện.
    
    Attributes
    ----------
    weights : List[np.ndarray]
        Danh sách ma trận trọng số W_i.
    biases : List[np.ndarray]
        Danh sách vector bias b_i.
    loss_history : List[float]
        Lịch sử loss qua các epoch.
    """
    
    def __init__(self,
                 layer_sizes: List[int],
                 learning_rate: float = 0.01,
                 max_iter: int = 500,
                 tol: float = 1e-4,
                 random_state: int = 42,
                 verbose: bool = True):
        
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        
        # Khởi tạo weights và biases
        self.weights = []
        self.biases = []
        self.loss_history = []
        
        # Set random seed
        np.random.seed(random_state)
        
        # Khởi tạo parameters
        self._initialize_parameters()
    
    def _initialize_parameters(self):
        """
        Khởi tạo weights và biases theo Xavier/Glorot initialization.
        
        W_i ~ N(0, sqrt(2 / (n_in + n_out)))
        b_i = 0
        """
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            
            # Xavier initialization
            std = np.sqrt(2.0 / (n_in + n_out))
            W = np.random.randn(n_in, n_out) * std
            b = np.zeros((1, n_out))
            
            self.weights.append(W)
            self.biases.append(b)
        
        if self.verbose:
            print(f"Khởi tạo {len(self.weights)} layers:")
            for i, (W, b) in enumerate(zip(self.weights, self.biases)):
                print(f"  Layer {i+1}: W{W.shape}, b{b.shape}")
    
    # =========================================================================
    # ACTIVATION FUNCTIONS
    # =========================================================================
    
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation: g(z) = 1 / (1 + e^(-z))
        
        Dùng cho hidden layers.
        """
        # Clip để tránh overflow
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))
    
    @staticmethod
    def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        """
        Đạo hàm Sigmoid: g'(z) = a * (1 - a), với a = sigmoid(z)
        """
        return a * (1 - a)
    
    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """
        Softmax activation: g(z_j) = e^(z_j) / Σ_k e^(z_k)
        
        Dùng cho output layer (multi-class classification).
        """
        # Trừ max để numerical stability
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    # =========================================================================
    # FORWARD PROPAGATION
    # =========================================================================
    
    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation qua tất cả layers.
        
        Parameters
        ----------
        X : np.ndarray
            Input data, shape (n_samples, n_features)
        
        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            - activations: Output của mỗi layer sau activation
            - z_values: Giá trị trước activation (z = W.x + b)
        """
        activations = [X]
        z_values = []
        
        current_input = X
        
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            # Linear transformation: z = W.x + b
            z = np.dot(current_input, W) + b
            z_values.append(z)
            
            # Activation
            if i == len(self.weights) - 1:
                # Output layer: Softmax
                a = self.softmax(z)
            else:
                # Hidden layer: Sigmoid
                a = self.sigmoid(z)
            
            activations.append(a)
            current_input = a
        
        return activations, z_values
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán xác suất cho mỗi class.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
        
        Returns
        -------
        np.ndarray
            Xác suất, shape (n_samples, n_classes)
        """
        activations, _ = self._forward(X)
        return activations[-1]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dự đoán class labels.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
        
        Returns
        -------
        np.ndarray
            Predicted labels, shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    # =========================================================================
    # LOSS FUNCTION
    # =========================================================================
    
    @staticmethod
    def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Categorical Cross-Entropy Loss.
        
        L = -Σ_j y_j * log(ŷ_j)
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels (one-hot encoded), shape (n_samples, n_classes)
        y_pred : np.ndarray
            Predicted probabilities, shape (n_samples, n_classes)
        
        Returns
        -------
        float
            Average loss.
        """
        # Clip để tránh log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # Cross-entropy
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss
    
    # =========================================================================
    # BACKWARD PROPAGATION
    # =========================================================================
    
    def _backward(self, 
                  X: np.ndarray, 
                  y_onehot: np.ndarray,
                  activations: List[np.ndarray],
                  z_values: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward propagation để tính gradients.
        
        Parameters
        ----------
        X : np.ndarray
            Input data.
        y_onehot : np.ndarray
            True labels (one-hot encoded).
        activations : List[np.ndarray]
            Activations từ forward pass.
        z_values : List[np.ndarray]
            Z values từ forward pass.
        
        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            Gradients cho weights và biases.
        """
        m = X.shape[0]  # Số samples
        
        grad_weights = [np.zeros_like(W) for W in self.weights]
        grad_biases = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error (Softmax + Cross-Entropy có đạo hàm đẹp)
        # δ = ŷ - y
        delta = activations[-1] - y_onehot
        
        # Backprop qua các layers
        for i in reversed(range(len(self.weights))):
            # Gradient cho W và b
            grad_weights[i] = np.dot(activations[i].T, delta) / m
            grad_biases[i] = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                # Propagate error to previous layer
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(activations[i])
        
        return grad_weights, grad_biases
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    
    def _to_onehot(self, y: np.ndarray) -> np.ndarray:
        """
        Chuyển labels sang one-hot encoding.
        
        Parameters
        ----------
        y : np.ndarray
            Labels, shape (n_samples,)
        
        Returns
        -------
        np.ndarray
            One-hot encoded, shape (n_samples, n_classes)
        """
        n_samples = len(y)
        n_classes = self.layer_sizes[-1]
        
        onehot = np.zeros((n_samples, n_classes))
        onehot[np.arange(n_samples), y.astype(int)] = 1
        
        return onehot
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPScratch':
        """
        Huấn luyện mô hình MLP.
        
        Parameters
        ----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Training labels (0 hoặc 1).
        
        Returns
        -------
        self
        """
        # Convert y to one-hot
        y_onehot = self._to_onehot(y)
        
        self.loss_history = []
        prev_loss = float('inf')
        
        if self.verbose:
            print(f"\nBắt đầu huấn luyện MLP ({self.max_iter} epochs)...")
            print("-" * 50)
        
        for epoch in range(self.max_iter):
            # Forward pass
            activations, z_values = self._forward(X)
            
            # Compute loss
            loss = self.categorical_cross_entropy(y_onehot, activations[-1])
            self.loss_history.append(loss)
            
            # Check convergence
            if abs(prev_loss - loss) < self.tol:
                if self.verbose:
                    print(f"Epoch {epoch+1}: Hội tụ (loss change < {self.tol})")
                break
            
            # Backward pass
            grad_weights, grad_biases = self._backward(X, y_onehot, activations, z_values)
            
            # Update parameters (Gradient Descent)
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * grad_weights[i]
                self.biases[i] -= self.learning_rate * grad_biases[i]
            
            # Logging
            if self.verbose and (epoch + 1) % 50 == 0:
                accuracy = self.score(X, y)
                print(f"Epoch {epoch+1:4d} | Loss: {loss:.6f} | Accuracy: {accuracy:.4f}")
            
            prev_loss = loss
        
        if self.verbose:
            print("-" * 50)
            print(f"Huấn luyện hoàn tất! Final Loss: {self.loss_history[-1]:.6f}")
        
        return self
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Tính accuracy trên dữ liệu.
        
        Parameters
        ----------
        X : np.ndarray
            Features.
        y : np.ndarray
            True labels.
        
        Returns
        -------
        float
            Accuracy score.
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    # =========================================================================
    # SAVE / LOAD MODEL
    # =========================================================================
    
    def save(self, filepath: str):
        """Lưu model ra file."""
        model_data = {
            'layer_sizes': self.layer_sizes,
            'weights': self.weights,
            'biases': self.biases,
            'learning_rate': self.learning_rate,
            'loss_history': self.loss_history
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model đã lưu tại: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'MLPScratch':
        """Load model từ file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(model_data['layer_sizes'], 
                    learning_rate=model_data['learning_rate'],
                    verbose=False)
        model.weights = model_data['weights']
        model.biases = model_data['biases']
        model.loss_history = model_data['loss_history']
        
        print(f"Model đã load từ: {filepath}")
        return model
    
    def get_model_summary(self) -> str:
        """Tạo tóm tắt kiến trúc model."""
        summary = []
        summary.append("=" * 50)
        summary.append("MLP MODEL SUMMARY")
        summary.append("=" * 50)
        summary.append(f"Architecture: {' -> '.join(map(str, self.layer_sizes))}")
        summary.append(f"Total layers: {len(self.layer_sizes)}")
        summary.append(f"Hidden layers: {len(self.layer_sizes) - 2}")
        summary.append("")
        
        total_params = 0
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            layer_type = "Hidden" if i < len(self.weights) - 1 else "Output"
            activation = "Sigmoid" if i < len(self.weights) - 1 else "Softmax"
            n_params = W.size + b.size
            total_params += n_params
            summary.append(f"Layer {i+1} ({layer_type}):")
            summary.append(f"  - Weights: {W.shape}")
            summary.append(f"  - Biases: {b.shape}")
            summary.append(f"  - Activation: {activation}")
            summary.append(f"  - Parameters: {n_params:,}")
        
        summary.append("")
        summary.append(f"Total parameters: {total_params:,}")
        summary.append("=" * 50)
        
        return "\n".join(summary)
