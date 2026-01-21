'''# ============================================================
# Imports
# ============================================================
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc, roc_curve
from bworth_filter import *
from preprocessing_data import *
from qtk import *
from sklearn.preprocessing import MinMaxScaler
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.seasonal import STL


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
    filtered_good = bw.process_and_plot_signals(relevant_good, start_idx=0, end_idx=len(relevant_good), plot_first=False, type= 'good')[0]
    filtered_anom = bw.process_and_plot_signals(relevant_anom, start_idx=0, end_idx=len(relevant_anom), plot_first=False, type= 'bad')[0]

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
    #L=norm_windows.shape[1]
    qtk = QuantumTemporalKernel(n_qubits=3, features=norm_windows.shape[2])

    # -----------------------------
    # Normalize input data to [-1, 1]
    # -----------------------------
    X_min = norm_windows.min(axis=(0, 1), keepdims=True)
    X_max = norm_windows.max(axis=(0, 1), keepdims=True)
    train_windows = 2 * ((norm_windows - X_min) / (X_max - X_min)) - 1

    X_min = anom_windows.min(axis=(0, 1), keepdims=True)
    X_max = anom_windows.max(axis=(0, 1), keepdims=True)
    test_windows = 2 * ((anom_windows - X_min) / (X_max - X_min)) - 1


    # --- KERNEL COMPUTATION ---
    qtk.compute_kernels(train_windows, test_windows)
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

def plot_scores(scores_normal,scores_anom, nu=0.05, name='qtk'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 5))
    plt.hist(scores_normal, bins=30, alpha=0.6, label="Normal", density=True)
    plt.hist(scores_anom, bins=30, alpha=0.6, label="Anomalous", density=True)
    plt.axvline(0, color='r', linestyle='--', label='Decision boundary (0)')
    plt.xlabel("Decision function score")
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"Decision scores distribution (name={name}, nu={nu})")
    plt.savefig(f"distrib_scores_{name}_{nu}.png", dpi=300)
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
    # --- Plot distribuzione punteggi ---
    y_true = y_true if isinstance(y_true, np.ndarray) else np.array(y_true)
    scores_normal = scores[y_true == 1]
    scores_anom = scores[y_true == -1]
    plot_scores(scores_normal,scores_anom, nu, name='qtk')
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

    for name, kernel in kernel_dict.items():
        if callable(kernel):
            # custom precomputed kernel
            K_train = kernel(X_train_flat, X_train_flat)
            K_test = kernel(X_test_flat, X_train_flat)
            oc = OneClassSVM(kernel='precomputed', nu=nu)
            oc.fit(K_train)
            scores = oc.decision_function(K_test)
            pred = oc.predict(K_test)
            y_true = y_true if isinstance(y_true, np.ndarray) else np.array(y_true)
            scores_normal = scores[y_true == 1]
            scores_anom = scores[y_true == -1]
        else:
            # standard sklearn kernel
            oc = OneClassSVM(kernel=kernel, gamma='scale', nu=nu)
            oc.fit(X_train_flat)
            scores = oc.decision_function(X_test_flat)
            pred = oc.predict(X_test_flat)
            y_true = y_true if isinstance(y_true, np.ndarray) else np.array(y_true)
            scores_normal = scores[y_true == 1]
            scores_anom = scores[y_true == -1]

        results_pred[name] = pred
        results_scores[name] = scores
        oc_svms[name] = oc

        # print metrics & plot
        plot_scores(scores_normal,scores_anom, nu, name=name)
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


# --- Funzione di normalizzazione per finestra ---
def zscore_windows(windows):
    return (windows - windows.mean(axis=1, keepdims=True)) / (windows.std(axis=1, keepdims=True) + 1e-8)

# --- Funzione plot ---
def plot_scores(scores_normal,scores_anom, nu=0.05, name='lstm_torch'):
    plt.figure(figsize=(5, 5))
    plt.hist(scores_normal, bins=30, alpha=0.6, label="Normal", density=True)
    plt.hist(scores_anom, bins=30, alpha=0.6, label="Anomalous", density=True)
    plt.axvline(0, color='r', linestyle='--', label='Decision boundary (0)')
    plt.xlabel("Reconstruction error")
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"Decision scores distribution ({name}, nu={nu})")
    plt.savefig(f"distrib_scores_{name}_{nu}.png", dpi=300)
    plt.close()

# --- LSTM Autoencoder con conv1D ---
class ConvLSTMAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, conv_channels=16, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=conv_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.encoder = nn.LSTM(conv_channels, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc_latent = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, channels)
        x = x.permute(0, 2, 1)        # -> (batch, channels, seq_len) per conv1d
        x = self.relu(self.conv1(x))
        x = x.permute(0, 2, 1)        # -> (batch, seq_len, channels) per LSTM
        _, (h_n, _) = self.encoder(x)
        h_n = h_n[-1]
        h_n = torch.relu(self.fc_latent(h_n))
        h_n_repeated = h_n.unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(h_n_repeated)
        out = self.output_layer(decoded)
        return out

# --- Funzione di training e valutazione ---
def run_lstm_conv_autoencoder(train_windows, test_windows, y_true, epochs=150, batch_size=128, lr=1e-3, device='cpu'):
    device = torch.device(device)
    
    # Normalizzazione
    train_windows = zscore_windows(train_windows)
    test_windows = zscore_windows(test_windows)
    
    # Torch tensors
    X_train = torch.tensor(train_windows, dtype=torch.float32).to(device)
    X_test = torch.tensor(test_windows, dtype=torch.float32).to(device)
    y_true = np.array(y_true)
    
    N_train, window_size, C = X_train.shape
    
    model = ConvLSTMAE(input_dim=C, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_loader = DataLoader(TensorDataset(X_train, X_train), batch_size=batch_size, shuffle=True)
    
    # Training
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(X_train):.6f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        train_recon = model(X_train)
        train_errors = ((train_recon - X_train)**2).mean(dim=(1,2)).cpu().numpy()
        # Threshold basato su mean + 3*std
        threshold = np.percentile(train_errors, 10)
        
        test_recon = model(X_test)
        test_errors = ((test_recon - X_test)**2).mean(dim=(1,2)).cpu().numpy()
        pred = np.where(test_errors > threshold, -1, 1)
    
    # Metriche
    bal_acc = balanced_accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, pos_label=-1)
    rec = recall_score(y_true, pred, pos_label=-1)
    f1 = f1_score(y_true, pred, pos_label=-1)
    auc = roc_auc_score((y_true==-1).astype(int), test_errors)
    
    print("\n--- Metrics for Conv-LSTM Autoencoder ---")
    print(f"Balanced Accuracy : {bal_acc:.3f}")
    print(f"Precision (anom)  : {prec:.3f}")
    print(f"Recall (anom)     : {rec:.3f}")
    print(f"F1 Score (anom)   : {f1:.3f}")
    print(f"AUC               : {auc:.3f}")
    
    # Plot distribuzione punteggi
    scores_normal = test_errors[y_true == 1]
    scores_anom = test_errors[y_true == -1]
    plot_scores(scores_normal, scores_anom, nu=0.05, name='lstm_conv')
    
    return model, pred, test_errors

def normalize_windows(train_windows, test_windows):
    """
    Normalize windows using per-window z-score normalization.
    
    Parameters
    ----------
    train_windows : np.ndarray, shape (N_train, window_size, C)
    test_windows  : np.ndarray, shape (N_test, window_size, C)
    
    Returns
    -------
    train_norm : np.ndarray
    test_norm  : np.ndarray
    """
    # Train normalization: z-score per finestra
    train_mean = train_windows.mean(axis=1, keepdims=True)
    train_std = train_windows.std(axis=1, keepdims=True) + 1e-8
    train_norm = (train_windows - train_mean) / train_std
    
    # Test normalization: z-score per finestra
    test_mean = test_windows.mean(axis=1, keepdims=True)
    test_std = test_windows.std(axis=1, keepdims=True) + 1e-8
    test_norm = (test_windows - test_mean) / test_std
    
    return train_norm, test_norm


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
    train_norm_indices = rng.choice(all_norm_indices, size=100, replace=False)
    test_norm_indices = np.setdiff1d(all_norm_indices, train_norm_indices)[:100]

    all_anom_indices = np.arange(N_anom)
    train_anom_indices = rng.choice(all_anom_indices, size=10, replace=False)
    test_anom_indices = np.setdiff1d(all_anom_indices, train_anom_indices)[:10]

    # Select windows
    rng = np.random.default_rng(seed=48)
    window_size = 100
    train_norm_windows = select_random_windows([datalist[i] for i in train_norm_indices], window_size, random_seed=48, verbose=False)
    train_windows = train_norm_windows
    shuffled_train_indices = rng.permutation(len(train_windows))
    train_windows = train_windows[shuffled_train_indices]

    test_norm_windows = select_random_windows([datalist[i] for i in test_norm_indices], window_size, random_seed=48, verbose=False)
    test_anom_windows = select_random_windows([datalist_anomalies[i] for i in test_anom_indices], window_size, random_seed=48, verbose=False)
    test_windows = np.concatenate([test_norm_windows, test_anom_windows], axis=0)

    # True labels
    y_true = np.concatenate([np.ones(len(test_norm_windows)), -np.ones(len(test_anom_windows))])

    # Shuffle test windows and corresponding labels
    shuffled_test_indices = rng.permutation(len(test_windows))
    test_windows = test_windows[shuffled_test_indices]
    y_test = y_true[shuffled_test_indices]

    # Quantum Temporal Kernel
    start_time = time.time()

    qtk, train_qtk, test_qtk = compute_quantum_kernel(train_windows, test_windows)

    end_time = time.time()
    elapsed = end_time - start_time

    # Conversione in ore, minuti, secondi
    hours, rem = divmod(elapsed, 3600)  
    minutes, seconds = divmod(rem, 60)

    oc_qtk, pred_qtk, scores_qtk = run_oneclass_svm_quantum(qtk, qtk.K_train_dict[0], qtk.K_test_dict[0], y_true)
    
    print(f"Total QTK Computing + ONECLASS Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Classical kernels dictionary
    kernel_dict = {
        "Linear": "linear",
        "Polynomial": "poly",
        "Sigmoid": "sigmoid",
        "RBF": "rbf",
        "Cosine": cosine_kernel,
        "Laplacian": laplacian_kernel,
        "Exponential": exponential_kernel,
        "MexicanHat": wavelet_kernel
    }

    oc_svms_classic, preds_classic, scores_classic = run_oneclass_svm_classical(train_windows, test_windows, y_test, kernel_dict)

    train_windows_scaled, test_windows_scaled = normalize_windows(train_windows, test_windows)
    # --- LSTM Autoencoder PyTorch ---
    # --- Conv-LSTM Autoencoder PyTorch ---
    print("\n--- Running Conv-LSTM Autoencoder (PyTorch) ---")
    lstm_model, lstm_pred, lstm_scores = run_lstm_conv_autoencoder(train_windows_scaled, test_windows_scaled, y_test, epochs=150, batch_size=128, device='cuda' if torch.cuda.is_available() else 'cpu')
    # Kernel diagnostics
    #K_norm = kernel_diagnostics(qtk.K_train_dict[0])


# ===============================================================
# 1️⃣ STL Feature Extraction
# ===============================================================
def extract_stl_features(sig, period=200, robust=True):
    """
    Applica STL a un segnale 2D (T, C) e restituisce 9 feature:
    - Derivata RMS per asse
    - Varianza residuo per asse
    - Rapporto energia residuo/seasonal per asse
    """
    sig = np.asarray(sig)
    T, C = sig.shape
    features = []

    for axis in range(C):
        series = sig[:, axis]
        eff_period = min(period, max(3, len(series)//4))
        stl = STL(series, period=eff_period, robust=robust)
        res = stl.fit()

        trend = res.trend
        seasonal = res.seasonal
        resid = res.resid

        # 1. Derivata RMS
        diff = np.diff(series)
        deriv_rms = np.sqrt(np.mean(diff**2))

        # 2. Varianza residuo
        var_resid = np.var(resid)

        # 3. Rapporto energia residuo/seasonal
        energy_ratio = np.sum(resid**2) / (np.sum(seasonal**2) + 1e-8)

        features.extend([deriv_rms, var_resid, energy_ratio])

    return np.array(features)


# ===============================================================
# 2️⃣ Main pipeline: STL + ConvLSTM + Evaluation
# ===============================================================
def main_pipeline_stl_conv_lstm(window_size=100, downsample_factor=10, period=200):
    rng = np.random.default_rng(seed=48)

    # --- Load and filter signals
    datalist, datalist_anomalies, filtered_good, filtered_anom = prepare_and_filter_signals()
    N_norm = len(filtered_good)
    N_anom = len(filtered_anom)

    # --- Train/test split
    n_train_norm = int(0.9 * N_norm)
    n_test_norm = N_norm - n_train_norm
    n_test_anom = int(0.1 * N_anom)

    all_norm_indices = np.arange(N_norm)
    all_anom_indices = np.arange(N_anom)
    train_norm_indices = rng.choice(all_norm_indices, size=n_train_norm, replace=False)
    test_norm_indices = np.setdiff1d(all_norm_indices, train_norm_indices)
    test_anom_indices = rng.choice(all_anom_indices, size=n_test_anom, replace=False)

    # --- Extract random windows
    train_windows = select_random_windows([filtered_good[i] for i in train_norm_indices],
                                          window_size, random_seed=48, verbose=False)
    test_norm_windows = select_random_windows([filtered_good[i] for i in test_norm_indices],
                                              window_size, random_seed=48, verbose=False)
    test_anom_windows = select_random_windows([filtered_anom[i] for i in test_anom_indices],
                                              window_size, random_seed=48, verbose=False)

    # --- Extract STL features
    print("\n--- Extracting STL features ---")
    X_train = np.array([extract_stl_features(w, period=period) for w in tqdm(train_windows, desc="Train STL")])
    X_test_norm = np.array([extract_stl_features(w, period=period) for w in tqdm(test_norm_windows, desc="Test normal STL")])
    X_test_anom = np.array([extract_stl_features(w, period=period) for w in tqdm(test_anom_windows, desc="Test anomalous STL")])

    # --- Merge test sets
    X_test = np.concatenate([X_test_norm, X_test_anom], axis=0)
    y_true = np.concatenate([np.ones(len(X_test_norm)), -np.ones(len(X_test_anom))])
    shuffled_idx = rng.permutation(len(X_test))
    X_test, y_test = X_test[shuffled_idx], y_true[shuffled_idx]

    # --- Normalize
    min_val = X_train.min(axis=0, keepdims=True)
    max_val = X_train.max(axis=0, keepdims=True)
    X_train_scaled = (X_train - min_val) / (max_val - min_val + 1e-8)
    X_test_scaled = np.clip((X_test - min_val) / (max_val - min_val + 1e-8), 0, 1)

    # --- Reshape for ConvLSTM (batch, seq_len, features)
    X_train_seq = X_train_scaled[:, None, :]
    X_test_seq = X_test_scaled[:, None, :]

    # --- Train Conv-LSTM Autoencoder
    print("\n--- Running Conv-LSTM Autoencoder ---")
    lstm_model, lstm_pred, lstm_scores = run_lstm_conv_autoencoder(
        X_train_seq, X_test_seq, y_test,
        epochs=150, batch_size=64,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # --- Compute metrics
    print("\n--- Evaluating ---")
    scores = np.array(lstm_scores).ravel()

    # Normalize scores (higher = more anomalous)
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test == -1, scores_norm)
    roc_auc = auc(fpr, tpr)

    # Threshold (e.g., 95th percentile)
    thr = np.percentile(scores_norm, 5)
    y_pred = np.where(scores_norm >= thr, -1, 1)

    precision = precision_score(y_test, y_pred, pos_label=-1)
    recall = recall_score(y_test, y_pred, pos_label=-1)
    f1 = f1_score(y_test, y_pred, pos_label=-1)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    print(f"\n📊 Performance Metrics:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")
    print(f"Balanced Accuracy: {bal_acc:.3f}")
    print(f"ROC-AUC:   {roc_auc:.3f}")

    # --- Plot ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - ConvLSTM AE (STL Features)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("img/roc_curve_conv_lstm_stl.png", dpi=200)
    plt.close()

    return {
        "model": lstm_model,
        "scores": scores_norm,
        "y_test": y_test,
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_accuracy": bal_acc,
            "roc_auc": roc_auc
        }
    }


if __name__ == "__main__":

    start_time = time.time()

    main_pipeline_stl_conv_lstm(window_size=100, downsample_factor=10, period=200)
    
    end_time = time.time()
    elapsed = end_time - start_time

    # Conversione in ore, minuti, secondi
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)

    print(f"Total Experiment Execution Time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")'''


# ============================================================
# main.py — CNC MILLING MACHINES
# Quantum Temporal Kernel + EVOVAQ
# Classical Kernels
# Quantum Kernel Hardware-Efficient
# ============================================================

# =====================
# Imports
# =====================
import numpy as np
import time
import os
import json
import matplotlib.pyplot as plt

from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from bworth_filter import *
from preprocessing_data import *
from qtk import *                     
from other_kernels import *            
import evovaq.tools.operators as op
from evovaq.problem import Problem
from evovaq.DifferentialEvolution import DE
from evovaq.GeneticAlgorithm import GA
from evovaq.ParticleSwarmOptimization import PSO    
from evovaq.BigBangBigCrunch import BBBC

# ============================================================
# Utility
# ============================================================
def compute_metrics(name, y_true, y_pred, scores):
    print(f"\n--- {name} ---")
    print("Balanced Acc :", balanced_accuracy_score(y_true, y_pred))
    print("Precision    :", precision_score(y_true, y_pred, pos_label=-1))
    print("Recall       :", recall_score(y_true, y_pred, pos_label=-1))
    print("F1           :", f1_score(y_true, y_pred, pos_label=-1))
    print("ROC-AUC      :", roc_auc_score((y_true == -1).astype(int), scores))


def plot_scores(scores_n, scores_a, name):
    plt.figure(figsize=(5,5))
    plt.hist(scores_n, bins=30, alpha=0.6, label="Normal", density=True)
    plt.hist(scores_a, bins=30, alpha=0.6, label="Anomalous", density=True)
    plt.axvline(0, color="r", linestyle="--")
    plt.legend()
    plt.title(name)
    plt.savefig(f"scores_{name}.png", dpi=300)
    plt.close()


# ============================================================
# EVOVAQ Optimization for QTK
# ============================================================
def optimize_qtk_evovaq(qtk,
                        X_train,
                        optimizer_name="DE",
                        pop_size=20,
                        max_gen=10,
                        n_runs=5,
                        selection=op.sel_tournament, 
                        crossover=op.cx_two_point, 
                        mutation=op.mut_gaussian,
                        sigma=0.5,
                        mut_indpb=0.25,
                        cxpb=0.9,
                        tournsize=3,
                        param_bounds=None,
                        save_dir="./qtk_evovaq_results_cnc/"):
    """
    Optimize Hamiltonian parameters of Quantum Temporal Kernel using EvoVAQ.
    Aligned 1:1 with Hydrogen implementation.
    """

    import os
    import json
    os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------
    # Problem definition
    # --------------------------------------------------
    n_params = len(qtk.full_terms)

    def cost_function(params):
        qtk.set_H_coeffs(params)
        K_train = qtk.generate_K_train(X_train)
        K_target = qtk.K_target

        alignment = np.trace(K_train @ K_target) / (
            np.linalg.norm(K_train, 'fro') * np.linalg.norm(K_target, 'fro')
        )
        return -alignment  # maximize alignment

    n_params = len(qtk.full_terms)

    if param_bounds is None:
        param_bounds = [(-0.5, 0.5)] * n_params

    problem = Problem(
        n_params=n_params,
        param_bounds=param_bounds,
        obj_function=cost_function
    )

    # --------------------------------------------------
    # Select optimizer (EVOVAQ)
    # --------------------------------------------------
    opt = optimizer_name.upper()

    if opt == "DE":
        global_search = DE()
    elif opt == "GA":
        global_search = GA(selection=selection,
                           crossover=crossover,
                           mutation=mutation,
                           sigma=sigma,
                           mut_indpb=mut_indpb,
                           cxpb=cxpb,
                           tournsize=tournsize)
    elif opt == "PSO":
        global_search = PSO()
    elif opt == "BBBC":
        global_search = BBBC(elitism=True)
    else:
        raise ValueError(f"Unknown EVOVAQ optimizer: {optimizer_name}")

    # --------------------------------------------------
    # Optimization loop
    # --------------------------------------------------
    best_solution = None
    best_fun = np.inf
    history_all = []

    for run in tqdm(range(1, n_runs+1), desc=f"{optimizer_name} runs"):
        print(f"\n🚀 EVOVAQ run {run}/{n_runs} ({opt})")

        
        res = global_search.optimize(
            problem,
            pop_size=pop_size,
            max_gen=max_gen,
            verbose=True,
            seed=42 * run
        )

        history_all.append(res.history if hasattr(res, "history") else [])

        if res.fun < best_fun:
            best_fun = res.fun
            best_solution = res

    # --------------------------------------------------
    # Set best parameters in QTK
    # --------------------------------------------------
    qtk.set_H_coeffs(best_solution.x)

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    coeff_names = [name for name, _ in qtk.full_terms]
    params_dict = dict(zip(coeff_names, map(float, best_solution.x)))

    with open(os.path.join(save_dir, f"best_solution_{opt.lower()}.json"), "w") as f:
        json.dump(
            {"params": params_dict, "fun": float(best_fun)},
            f,
            indent=4
        )

    K_train_final = qtk.generate_K_train(X_train)
    np.save(os.path.join(save_dir, f"K_train_final_{opt.lower()}.npy"), K_train_final)

    return params_dict, history_all, K_train_final


# ============================================================
# Quantum Temporal Kernel + Classical Kernels
# ============================================================
def main():
    print("\n⚙️ MAIN — Quantum Temporal Kernel + Classical Kernels (CNC)")

    rng = np.random.default_rng(48)
    window_size = 100

    # ========================================================
    # Load CNC signals
    # ========================================================
    datalist_norm, _ = load_tool_research_data("./data", label="good")
    datalist_anom, _ = load_tool_research_data("./data", label="bad")

    # ========================================================
    # PHASE 1 — OPTIMIZATION DATA (EVOVAQ)
    # 12 normal + 3 anomalous
    # ========================================================
    print("\n🔧 Preparing optimization dataset")

    X_train_opt_norm = select_random_windows(
        datalist_norm[:12], window_size
    )
    X_train_opt_anom = select_random_windows(
        datalist_anom[:3], window_size
    )

    X_train_opt = np.concatenate([X_train_opt_norm, X_train_opt_anom])

    print(
        f"Optimization set -> {len(X_train_opt)} windows "
        f"(norm={len(X_train_opt_norm)}, anom={len(X_train_opt_anom)})"
    )

    # ========================================================
    # PHASE 2 — TEST DATA (55 windows, 5 anomalies)
    # MUST BE DIFFERENT FROM OPTIMIZATION
    # ========================================================
    print("\n🧪 Preparing test dataset")

    X_test_norm = select_random_windows(
        datalist_norm[12:62], window_size   # 50 normal
    )
    X_test_anom = select_random_windows(
        datalist_anom[3:8], window_size     # 5 anomalous (DIFFERENT)
    )

    X_test = np.concatenate([X_test_norm, X_test_anom])
    y_test = np.concatenate([
        np.ones(len(X_test_norm)),
        -np.ones(len(X_test_anom))
    ])

    print(
        f"Test set -> {len(X_test)} windows "
        f"(norm={len(X_test_norm)}, anom={len(X_test_anom)})"
    )

    # ========================================================
    # Quantum Temporal Kernel
    # ========================================================
    print("\n⚛️ Quantum Temporal Kernel")

    qtk = QuantumTemporalKernel(
        n_qubits=3,
        features=3,
        n_norm=len(X_train_opt_norm),
        n_anom=len(X_train_opt_anom),
        optimize=True
    )

    optimizer = "GA"  # oppure WITHOUT_OPT
    if optimizer != "WITHOUT_OPT":
        optimize_qtk_evovaq(qtk, X_train_opt, optimizer_name=optimizer)
        # ========================================================
        # FINAL TRAIN SET (ONLY NORMAL, REALISTIC SETUP)
        # ========================================================
    X_train = select_random_windows(
        datalist_norm[:55], window_size
    )

    # ========================================================
    # Kernel matrices
    # ========================================================
    K_train = qtk.generate_K_train(X_train)
    K_test = qtk.generate_K_test(X_train, X_test)

    # ========================================================
    # Grid search on nu — QTK (IDENTICAL TO HYDROGEN)
    # ========================================================
    print("\n🔍 Grid search on nu (QTK)")

    nu_min = 0.01
    nu_max = 0.2
    n_nu = 20

    best_nu = None
    best_acc = -1.0

    nu_values = np.linspace(nu_min, nu_max, n_nu)

    for nu in nu_values:
        oc_tmp = OneClassSVM(kernel="precomputed", nu=nu)
        oc_tmp.fit(K_train)

        y_pred_train = oc_tmp.predict(K_train)
        acc = np.mean(y_pred_train == 1)

        if acc > best_acc:
            best_acc = acc
            best_nu = nu

    print(f"✅ Best nu found: {best_nu:.4f} (train acc={best_acc:.3f})")

    # ========================================================
    # Final One-Class SVM — QTK
    # ========================================================
    oc = OneClassSVM(kernel="precomputed", nu=best_nu)
    oc.fit(K_train)

    scores = oc.decision_function(K_test)
    y_pred = oc.predict(K_test)

    compute_metrics("QTK", y_test, y_pred, scores)
    plot_scores(
        scores[y_test == 1],
        scores[y_test == -1],
        "QTK"
    )

    # ========================================================
    # Classical Kernels
    # ========================================================
    print("\n📐 Classical Kernels")

    Xtr = X_train.reshape(len(X_train), -1)
    Xte = X_test.reshape(len(X_test), -1)

    kernels = {
        "Linear": "linear",
        "RBF": "rbf",
        "Cosine": cosine_kernel,
        "Laplacian": laplacian_kernel,
        "MexicanHat": wavelet_kernel
    }

    for name, kernel in kernels.items():
        print(f"\n▶ Kernel: {name}")

        if callable(kernel):
            Ktr = kernel(Xtr, Xtr)
            Kte = kernel(Xte, Xtr)

            best_nu = None
            best_acc = -1.0

            for nu in nu_values:
                oc_tmp = OneClassSVM(kernel="precomputed", nu=nu)
                oc_tmp.fit(Ktr)
                y_pred_train = oc_tmp.predict(Ktr)
                acc = np.mean(y_pred_train == 1)

                if acc > best_acc:
                    best_acc = acc
                    best_nu = nu

            oc = OneClassSVM(kernel="precomputed", nu=best_nu)
            oc.fit(Ktr)

            scores = oc.decision_function(Kte)
            y_pred = oc.predict(Kte)

        else:
            best_nu = None
            best_acc = -1.0

            for nu in nu_values:
                oc_tmp = OneClassSVM(kernel=kernel, nu=nu)
                oc_tmp.fit(Xtr)
                y_pred_train = oc_tmp.predict(Xtr)
                acc = np.mean(y_pred_train == 1)

                if acc > best_acc:
                    best_acc = acc
                    best_nu = nu

            oc = OneClassSVM(kernel=kernel, nu=best_nu)
            oc.fit(Xtr)

            scores = oc.decision_function(Xte)
            y_pred = oc.predict(Xte)

        compute_metrics(name, y_test, y_pred, scores)
        plot_scores(
            scores[y_test == 1],
            scores[y_test == -1],
            name,
        )
    
    # --- Hardware Efficient Kernel ---
    qk = QuantumKernel(n_qubits=3)

    K_train = qk.generate_K_train(X_train)
    K_test = qk.generate_K_test(X_train, X_test)

    oc = OneClassSVM(kernel="precomputed", nu=0.05)
    oc.fit(K_train)
    scores = oc.decision_function(K_test)
    y_pred = oc.predict(K_test)

    compute_metrics("Hardware-Efficient QK", y_test, y_pred, scores)
    plot_scores(scores[y_test==1], scores[y_test==-1], "HardwareEfficient")


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\n⏱ Total time: {time.time() - start:.2f}s")

