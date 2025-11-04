import numpy as np
from scipy.spatial.distance import cdist
from fastdtw import fastdtw

# ============================================================
# 1. Cosine Similarity Kernel
# ============================================================
def cosine_kernel(X, Y=None):
    """Cosine similarity kernel."""
    if Y is None:
        Y = X
    # Normalize vectors
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return X_norm @ Y_norm.T


# ============================================================
# 2. Laplacian Kernel
# ============================================================
def laplacian_kernel(X, Y=None, gamma=1.0):
    """Laplacian kernel: K(x,y) = exp(-gamma * ||x-y||_1)"""
    if Y is None:
        Y = X
    dists = cdist(X, Y, metric='cityblock')  # L1 norm
    return np.exp(-gamma * dists)


# ============================================================
# 3. Exponential Kernel
# ============================================================
def exponential_kernel(X, Y=None, gamma=1.0):
    """Exponential kernel: K(x,y) = exp(-gamma * ||x-y||_2)"""
    if Y is None:
        Y = X
    dists = cdist(X, Y, metric='euclidean')  # L2 norm
    return np.exp(-gamma * dists)


# ============================================================
# 4. Wavelet / Sinc Kernel
# ============================================================
def wavelet_kernel(X, Y=None, a=1.0, wavelet='mexican_hat'):
    """
    Wavelet kernel (Mexican Hat by default)
    K(x, y) = prod_i psi((x_i - y_i)/a)
    """
    if Y is None:
        Y = X

    def psi_mexican_hat(u):
        return (1 - u**2) * np.exp(-u**2 / 2)

    n_X, d = X.shape
    n_Y = Y.shape[0]
    K = np.zeros((n_X, n_Y))

    for i in range(n_X):
        for j in range(n_Y):
            u = (X[i] - Y[j]) / a
            K[i, j] = np.prod(psi_mexican_hat(u))
    return K


# ============================================================
# 5. DTW Kernel using fastdtw
# ============================================================
def dtw_kernel(X, Y=None, gamma=0.1):
    """
    DTW-based kernel: K(x,y) = exp(-gamma * DTW(x,y))
    Assumes X and Y are 2D arrays where each row is a flattened window
    """
    if Y is None:
        Y = X

    n_X = X.shape[0]
    n_Y = Y.shape[0]
    K = np.zeros((n_X, n_Y))

    for i in range(n_X):
        for j in range(n_Y):
            # fastdtw needs 1D sequences; if X[i] is multidim, flatten first
            dist, _ = fastdtw(X[i], Y[j])
            K[i, j] = np.exp(-gamma * dist)
    return K