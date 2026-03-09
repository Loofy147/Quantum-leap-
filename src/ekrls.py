import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

class EKRLS:
    """
    Extended Kernel Recursive Least Squares (EKRLS) Engine.
    Used for real-time tracking of quantum states in RKHS.
    """
    def __init__(self, kernel_sigma=1.0, lmbda=0.1, nu=0.1):
        self.sigma = kernel_sigma
        self.lmbda = lmbda  # Regularization parameter
        self.nu = nu        # Sparsification threshold (ALD)

        self.dictionary = [] # Centers in RKHS
        self.alpha = None    # Weights
        self.P = None        # Inverse covariance matrix
        self.K_inv = None    # Inverse of kernel matrix for dictionary

    def _get_kernel(self, X, Y):
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)
        gamma = 1.0 / (2 * self.sigma**2)
        return rbf_kernel(X, Y, gamma=gamma)

    def update(self, x, y):
        x = np.atleast_2d(x)
        y_val = float(y)

        if len(self.dictionary) == 0:
            self.dictionary.append(x)
            k_ii = self._get_kernel(x, x)[0, 0]
            self.K_inv = np.array([[1.0 / (k_ii + self.lmbda)]])
            self.P = np.array([[1.0]])
            self.alpha = np.array([[y_val / (k_ii + self.lmbda)]])
            return

        dict_matrix = np.vstack(self.dictionary)
        k_t = self._get_kernel(x, dict_matrix).T

        a_t = self.K_inv @ k_t
        delta = self._get_kernel(x, x)[0, 0] - (k_t.T @ a_t).item()

        if delta > self.nu:
            self.dictionary.append(x)

            new_K_inv = np.zeros((len(self.dictionary), len(self.dictionary)))
            new_K_inv[:-1, :-1] = self.K_inv + (a_t @ a_t.T) / delta
            new_K_inv[:-1, -1:] = -a_t / delta
            new_K_inv[-1:, :-1] = -a_t.T / delta
            new_K_inv[-1, -1] = 1.0 / delta
            self.K_inv = new_K_inv

            new_P = np.zeros((len(self.dictionary), len(self.dictionary)))
            new_P[:-1, :-1] = self.P
            new_P[-1, -1] = 1.0
            self.P = new_P

            e_t = y_val - (k_t.T @ self.alpha).item()
            new_alpha = np.zeros((len(self.dictionary), 1))
            new_alpha[:-1] = self.alpha - (a_t / delta) * e_t
            new_alpha[-1] = e_t / delta
            self.alpha = new_alpha

        else:
            den = 1.0 + (a_t.T @ self.P @ a_t).item()
            q_t = (self.P @ a_t) / den
            self.P = self.P - q_t @ a_t.T @ self.P
            error = y_val - (k_t.T @ self.alpha).item()
            self.alpha = self.alpha + (self.K_inv @ q_t) * error

    def predict(self, x):
        if len(self.dictionary) == 0:
            return 0.0
        x = np.atleast_2d(x)
        dict_matrix = np.vstack(self.dictionary)
        k_x = self._get_kernel(x, dict_matrix)
        return float((k_x @ self.alpha).item())

    def state_transition(self, f_func):
        if len(self.dictionary) == 0:
            return

        new_dict = []
        for x in self.dictionary:
            new_dict.append(f_func(x))
        self.dictionary = new_dict

        dict_matrix = np.vstack(self.dictionary)
        K = self._get_kernel(dict_matrix, dict_matrix) + self.lmbda * np.eye(len(self.dictionary))
        self.K_inv = np.linalg.inv(K)
