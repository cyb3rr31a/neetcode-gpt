import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate BCE: -1/N * sum(y * log(p) + (1-y) * log(1-p))
        bce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return round(float(bce_loss), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: clip y_pred to [1e-7, 1 - 1e-7] to avoid log(0)
        # return round(your_answer, 4)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate CCE: -1/N * sum_i(sum_c(y_ic * log(p_ic)))
        cce_loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        return round(float(cce_loss), 4)
