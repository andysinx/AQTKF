# ============================================================
# Imports
# ============================================================
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from bworth_filter import *
from preprocessing_data import *
from qtk import *
from temporal_kernelized_fcm import *
import numpy as np
from scipy.stats import spearmanr, pearsonr, ttest_ind
from sklearn.metrics import mean_squared_error
from sklearn.svm import OneClassSVM
from sklearn.manifold import MDS
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from other_kernels import *
import time
# ============================================================
# 1. Signal Comparison Utilities
# ============================================================

def compare_signals(datalist):
    """Compare two signals (sig1 vs sig2) with normalized correlation metrics and plots."""
    sig1, sig2 = datalist[0], datalist[1]
    target_len = len(sig2)
    sig1 = sig1[:target_len, :]

    axes_labels = ['X', 'Y', 'Z']
    print("=== Normalized Metrics ===")

    for i, label in enumerate(axes_labels):
        # Normalization (zero mean, unit variance)
        x1 = (sig1[:, i] - np.mean(sig1[:, i])) / np.std(sig1[:, i])
        x2 = (sig2[:, i] - np.mean(sig2[:, i])) / np.std(sig2[:, i])

        # Compute metrics
        spearman_corr, _ = spearmanr(x1, x2)
        pearson_corr, _ = pearsonr(x1, x2)
        rmse = np.sqrt(mean_squared_error(x1, x2))

        print(f"{label}-axis:")
        print(f"  Spearman correlation: {spearman_corr:.3f}")
        print(f"  Pearson correlation : {pearson_corr:.3f}")
        print(f"  RMSE               : {rmse:.3f}\n")

    # Plot normalized signal overlay
    t = np.arange(target_len)
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

    for i, label in enumerate(axes_labels):
        x1 = (sig1[:, i] - np.mean(sig1[:, i])) / np.std(sig1[:, i])
        x2 = (sig2[:, i] - np.mean(sig2[:, i])) / np.std(sig2[:, i])
        axs[i].plot(t, x1, label='OP00_00', linewidth=1.5)
        axs[i].plot(t, x2, label='OP01_00', linewidth=1.5, alpha=0.7)
        axs[i].set_ylabel(f'{label}-axis')
        axs[i].grid()
        axs[i].legend()

    axs[-1].set_xlabel('Sample')
    fig.suptitle('Comparing Normalized Signals: sig1 vs sig2', fontsize=14)
    plt.tight_layout()
    plt.savefig('image/img6.png', dpi=300)
    plt.close()


# ============================================================
# 2. Data Preparation and Filtering
# ============================================================

def prepare_and_filter_signals(start_idx=0, end_idx=200):
    """Load datasets and apply Butterworth filter, taking only signals 0-200."""
    # Original good signals
    datalist, _ = load_tool_research_data(data_path="./data", label="good")
    # Original bad/anomalous signals
    datalist_anomalies, _ = load_tool_research_data(data_path="./data", label="bad")

    print("Total Normal Data:", len(datalist), "Total Anomalous Data:", len(datalist_anomalies))

    # Prendi solo i segnali da 0 a 200
    relevant_good = datalist[start_idx:end_idx]
    relevant_anom = datalist_anomalies[start_idx:end_idx]

    # Applica il filtro Butterworth
    bw = ButterworthFilter(fs=2000, cutoff=75, order=4)
    filtered_good = bw.process_and_plot_signals(relevant_good, start_idx=0, end_idx=len(relevant_good), plot_first=False)[0]
    filtered_anom = bw.process_and_plot_signals(relevant_anom, start_idx=0, end_idx=len(relevant_anom), plot_first=False)[0]

    return relevant_good, relevant_anom, filtered_good, filtered_anom


# ============================================================
# 3. Quantum Kernel Computation
# ============================================================

def compute_quantum_kernel(norm_windows, anom_windows):
    """
    Compute quantum kernel matrices and print detailed diagnostics.

    Parameters
    ----------
    norm_windows : np.ndarray
        Array of normal windows, shape (N_norm, window_size, C)
    anom_windows : np.ndarray
        Array of anomalous windows, shape (N_anom, window_size, C)

    Returns
    -------
    qtk : QuantumTemporalKernel
        Object containing the computed kernels
    train_windows : np.ndarray
        Normalized train windows
    test_windows : np.ndarray
        Normalized test windows
    """

    # Initialize quantum kernel
    qtk = QuantumTemporalKernel(n_qubits=3, L=norm_windows.shape[1], features=norm_windows.shape[2])

    # -----------------------------
    # Normalize input data to [-1, 1]
    # -----------------------------
    X_min = norm_windows.min(axis=(0, 1), keepdims=True)
    X_max = norm_windows.max(axis=(0, 1), keepdims=True)
    train_windows = 2 * ((norm_windows - X_min) / (X_max - X_min)) - 1

    X_min = anom_windows.min(axis=(0, 1), keepdims=True)
    X_max = anom_windows.max(axis=(0, 1), keepdims=True)
    test_windows = 2 * ((anom_windows - X_min) / (X_max - X_min)) - 1


    # Concatenate normal and anomalous windows
    all_windows = np.concatenate([train_windows, test_windows], axis=0)

    # Shuffle the concatenated windows
    rng = np.random.default_rng(seed=48)  # fissiamo il seed per riproducibilità
    shuffled_indices = rng.permutation(len(all_windows))
    all_windows_shuffled = all_windows[shuffled_indices]

    # --- KERNEL COMPUTATION ---
    qtk.compute_kernels(train_windows, all_windows_shuffled)
    K_train = qtk.K_train_dict[0]

    # --- DIAGNOSTICS ---
    K_offdiag = K_train[np.triu_indices_from(K_train, k=1)]
    print("\n--- Quantum Kernel Diagnostics ---")
    print(f"Train kernel shape: {K_train.shape}")
    print(f"Diagonal mean    : {np.diag(K_train).mean():.4f}")
    print(f"Off-diagonal mean: {K_offdiag.mean():.4f}")
    print(f"Off-diagonal std : {K_offdiag.std():.4f}")

    # --- SYMMETRY CHECK ---
    sym_err = np.linalg.norm(K_train - K_train.T, ord='fro') / np.linalg.norm(K_train, ord='fro')
    print(f"Frobenius relative asymmetry: {sym_err:.3e}")

    # --- EIGENVALUES ---
    eigvals = eigh(K_train, eigvals_only=True)
    print(f"Eigenvalues min/max: {eigvals.min():.4f} / {eigvals.max():.4f}")
    print("--------------------------\n")

    return qtk, train_windows, test_windows


# ============================================================
# Utilities for metrics and plotting
# ============================================================

def compute_metrics(kernel_name, y_true, pred, scores):
    print(f"\n--- Metrics for {kernel_name} Kernel ---")
    bal_acc = balanced_accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, pos_label=-1)
    rec = recall_score(y_true, pred, pos_label=-1)
    f1 = f1_score(y_true, pred, pos_label=-1)
    auc = roc_auc_score((y_true==-1).astype(int), scores)
    print(f"Balanced Accuracy : {bal_acc:.3f}")
    print(f"Precision (anom)  : {prec:.3f}")
    print(f"Recall (anom)     : {rec:.3f}")
    print(f"F1 Score (anom)   : {f1:.3f}")
    print(f"AUC               : {auc:.3f}")

def plot_scores(scores, kernel_name):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    plt.hist(scores, bins=30, alpha=0.6, label='Test')
    threshold = np.percentile(scores, 5)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.xlabel("Decision function score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'image/distribution_scores_{kernel_name}.png', dpi=300)
    plt.close()


# ============================================================
# 1. Quantum Temporal Kernel SVM
# ============================================================

def run_oneclass_svm_quantum(qtk, K_train, K_test, y_true, nu=0.05):
    from sklearn.svm import OneClassSVM
    import numpy as np

    oc_svm_qtk = OneClassSVM(kernel='precomputed', nu=nu)
    oc_svm_qtk.fit(K_train)
    scores = oc_svm_qtk.decision_function(K_test)
    pred = oc_svm_qtk.predict(K_test)

    totale = len(pred)
    n_normali = np.sum(pred == 1)
    n_anomale = np.sum(pred == -1)
    print(f"[Quantum] Tested: {totale}, Normali: {n_normali}, Anomale: {n_anomale}")
    plot_scores(scores, "qtk")
    compute_metrics("Quantum", y_true, pred, scores)

    return oc_svm_qtk, pred, scores


# ============================================================
# 2. Classical Kernels SVM
# ============================================================

def run_oneclass_svm_classical(train_windows, test_windows, y_true, kernel_dict, nu=0.05):

    n_train = train_windows.shape[0]
    results_pred = {}
    results_scores = {}
    oc_svms = {}

    # reshape to 2D once
    X_train_flat = train_windows.reshape(n_train, -1)
    X_test_flat = test_windows.reshape(test_windows.shape[0], -1)
    X_test_all = np.concatenate([X_train_flat, X_test_flat], axis=0)
    rng = np.random.default_rng(seed=48)
    shuffled_indices = rng.permutation(len(X_test_all))
    X_test_all_shuffled = X_test_all[shuffled_indices]

    for name, kernel in kernel_dict.items():
        if callable(kernel):
            # custom precomputed kernel
            K_train = kernel(X_train_flat, X_train_flat)
            K_test = kernel(X_test_flat, X_train_flat)
            oc = OneClassSVM(kernel='precomputed', nu=nu)
            oc.fit(K_train)
            scores = oc.decision_function(K_test)
            pred = oc.predict(K_test)
        else:
            # standard sklearn kernel
            oc = OneClassSVM(kernel=kernel, gamma='scale', nu=nu)
            oc.fit(X_train_flat)
            scores = oc.decision_function(X_test_all_shuffled)
            pred = oc.predict(X_test_all_shuffled)

        results_pred[name] = pred
        results_scores[name] = scores
        oc_svms[name] = oc

        # print metrics & plot
        plot_scores(scores, name)
        compute_metrics(name, y_true, pred, scores)

    return oc_svms, results_pred, results_scores


# ============================================================
# 5. Kernel Diagnostics and Visualization
# ============================================================

def kernel_diagnostics(K):
    """Perform spectral and normalization diagnostics on the kernel matrix."""
    print("K shape:", K.shape)
    print("K min/max:", K.min(), K.max())

    # Symmetry check
    sym_err = np.linalg.norm(K - K.T, ord='fro') / np.linalg.norm(K, ord='fro')
    print(f"Frobenius relative asymmetry: {sym_err:.3e}")

    # Eigenvalues
    eigvals = eigh(K, eigvals_only=True)
    print("Eigenvalues min/max:", eigvals.min(), eigvals.max())

    plt.figure(figsize=(5, 4))
    plt.hist(eigvals, bins=40)
    plt.title("Eigenvalues of K")
    plt.xlabel("eig")
    plt.ylabel("count")
    plt.savefig('image/img7.png', dpi=300)
    plt.close()

    # Force symmetry and project to PSD
    K_sym = (K + K.T) / 2.0
    eigvals, evecs = eigh(K_sym)
    eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None)
    K_psd = (evecs * eigvals_clipped) @ evecs.T

    # Normalize to cosine-like affinities
    diag_psd = np.diag(K_psd).copy()
    diag_psd[diag_psd <= 0] = 1e-12
    norm_factor = np.sqrt(np.outer(diag_psd, diag_psd))
    K_norm = np.clip(K_psd / norm_factor, -1.0, 1.0)

    plt.figure(figsize=(5, 5))
    plt.imshow(K_norm, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Normalized PSD-projected kernel (K_norm)")
    plt.savefig('image/img9.png', dpi=300)
    plt.close()

    return K_norm


# ============================================================
# 6. Main Execution Flow
# ============================================================

def main():
    rng = np.random.default_rng(seed=48)

    # Load and filter signals
    datalist, datalist_anomalies, _, _ = prepare_and_filter_signals()
    N_norm = len(datalist)
    N_anom = len(datalist_anomalies)

    # Split train/test indices
    all_norm_indices = np.arange(N_norm)
    train_norm_indices = rng.choice(all_norm_indices, size=90, replace=False)
    test_norm_indices = np.setdiff1d(all_norm_indices, train_norm_indices)[:90]

    all_anom_indices = np.arange(N_anom)
    train_anom_indices = rng.choice(all_anom_indices, size=10, replace=False)
    test_anom_indices = np.setdiff1d(all_anom_indices, train_anom_indices)[:10]

    # Select windows
    window_size = 100
    train_norm_windows = select_random_windows([datalist[i] for i in train_norm_indices], window_size, random_seed=48, verbose=False)
    train_anom_windows = select_random_windows([datalist_anomalies[i] for i in train_anom_indices], window_size, random_seed=48, verbose=False)
    train_windows = np.concatenate([train_norm_windows, train_anom_windows], axis=0)

    test_norm_windows = select_random_windows([datalist[i] for i in test_norm_indices], window_size, random_seed=48, verbose=False)
    test_anom_windows = select_random_windows([datalist_anomalies[i] for i in test_anom_indices], window_size, random_seed=48, verbose=False)
    test_windows = np.concatenate([test_norm_windows, test_anom_windows], axis=0)

    # True labels
    y_true = np.concatenate([np.ones(len(test_norm_windows)), -np.ones(len(test_anom_windows))])

    # Quantum Temporal Kernel
    start_time = time.time()
    
    qtk, train_qtk, test_qtk = compute_quantum_kernel(train_windows, train_windows)

    end_time = time.time()
    elapsed = end_time - start_time

    # Conversione in ore, minuti, secondi
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    oc_qtk, pred_qtk, scores_qtk = run_oneclass_svm_quantum(qtk, qtk.K_train_dict[0], qtk.K_test_dict[0], y_true)

    # Classical kernels dictionary
    kernel_dict = {
        "Linear": "linear",
        "Polynomial": "poly",
        "Sigmoid": "sigmoid",
        "RBF": "rbf",
        "Cosine": cosine_kernel,
        "Laplacian": laplacian_kernel,
        "Exponential": exponential_kernel,
        "MexicanHat": wavelet_kernel,
        "DTW": dtw_kernel
    }

    oc_svms_classic, preds_classic, scores_classic = run_oneclass_svm_classical(train_windows, test_windows, y_true, kernel_dict)

    # Kernel diagnostics
    K_norm = kernel_diagnostics(qtk.K_train_dict[0])

if __name__ == "__main__":

    start_time = time.time()

    main()
    
    end_time = time.time()
    elapsed = end_time - start_time

    # Conversione in ore, minuti, secondi
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")