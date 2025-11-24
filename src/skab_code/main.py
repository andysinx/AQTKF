# main.py
from preprocessing import *
import os
from qtk import *
from other_kernels import *
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import eigh
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import evovaq.tools.operators as op
from evovaq.problem import Problem
from evovaq.DifferentialEvolution import DE
from evovaq.GeneticAlgorithm import GA
from evovaq.ParticleSwarmOptimization import PSO    
from evovaq.BigBangBigCrunch import BBBC
from ChaothicEnhancedGeneticAlgorithm import CGA
from tqdm import tqdm
import json

def optimize_qtk(qtk: QuantumTemporalKernel,
                 X_train,
                 n_norm=20,
                 n_anom=5,
                 optimizer="DE",
                 n_runs=5,
                 pop_size=20,
                 max_gen=1,
                 selection=op.sel_tournament, 
                 crossover=op.cx_two_point, 
                 mutation=op.mut_gaussian,
                 sigma=0.5,
                 mut_indpb=0.25,
                 cxpb=0.9,
                 tournsize=3,
                 param_bounds=None,
                 save_dir="./qtk_evovaq_results/"):

    """
    Optimize Hamiltonian parameters of Quantum Temporal Kernel using EvoVAQ (DE, GA, PSO, BBBC, ecc).

    Args:
        optimizer: "DE", "GA", "PSO", or "BBBC"
        qtk: QuantumTemporalKernel instance
        X_train: list of windows (train)
        n_runs: number of independent runs
        pop_size: population size
        max_gen: max generations
        param_bounds: tuple (min, max) bounds for parameters
        save_dir: folder to save results
    Returns:
        best_solution: dictionary with optimized parameters and kernel
    """

    import os
    os.makedirs(save_dir, exist_ok=True)

    n_params = len(qtk.full_terms)
    if param_bounds is None:
        param_bounds = (-0.5, 0.5)
    
    # --- salva kernel iniziale ---
    '''K_train_initial = qtk.generate_K_train(X_train)
    print("K_train_old",K_train_initial)
    np.save(os.path.join(save_dir, f"K_train_initial_{optimizer}.npy"), K_train_initial)'''

    def cost_function(params):
        # Update params
        old_coeffs = [c for (_, c) in qtk.full_terms]
        print("old_coeffs",old_coeffs)
        K1 = qtk.generate_K_train(X_train)
        print("K_train_old:", K1)
        qtk.set_H_coeffs(params)
        new_coeffs_list = [c for (_, c) in qtk.full_terms]
        print("new_coeffs_list",new_coeffs_list)
        diff = np.array(new_coeffs_list) - np.array(old_coeffs)
        #print("Diff coeffs:", diff)
        K_train = qtk.generate_K_train(X_train)
        print("K_train_new:", K_train)
        #print("Diff kernel:", np.linalg.norm(K_train - K1, 'fro'))

        # Target kernel (cluster intra-normal / anom)
        K_target = qtk.K_target
        alignment = np.trace(K_train @ K_target) / (
            np.linalg.norm(K_train, 'fro') * np.linalg.norm(K_target, 'fro')
        )
        return -alignment  # Maximize alignment -> minimize negative


    problem = Problem(n_params=n_params, param_bounds=param_bounds, obj_function=cost_function)

    # --------------------------
    # Select optimizer
    # --------------------------
    optimizer_upper = optimizer.upper()
    if optimizer_upper == "DE":
        global_search = DE()
    elif optimizer_upper == "GA":
        global_search = GA(selection=selection,
                           crossover=crossover,
                           mutation=mutation,
                           sigma=sigma,
                           mut_indpb=mut_indpb,
                           cxpb=cxpb,
                           tournsize=tournsize)
    elif optimizer_upper == "PSO":
        global_search = PSO(inertia_weight=0.7298, phi1=1.49618, phi2=1.49618)
    elif optimizer_upper == "BBBC":
        global_search = BBBC(elitism=True, alpha=10.0, beta=0.25)
    elif optimizer_upper == "CGA":
        global_search = CGA(selection=selection,
                            crossover=crossover,
                            mutation=mutation,
                            cxpb=cxpb,
                            mutpb=mut_indpb,
                            elitism=True,
                            chaotic_map="logistic",
                            chaotic_alpha=0.25,
                            chaotic_iterations=10)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # --------------------------
    # Ciclo sui run
    # --------------------------
    # --- history per grafici ---
    history_all_runs = []

    # --- ciclo sulle run ---
    best_solution_overall = None
    best_fun_overall = np.inf

    for run in tqdm(range(1, n_runs+1), desc=f"{optimizer} runs"):
        # Ogni ottimizzatore dovrebbe restituire anche la history dei migliori fitness per generazione
        res = global_search.optimize(problem, pop_size=pop_size, max_gen=max_gen, verbose=True, seed=42*run)
        
        # salva evoluzione fitness per questa run (supponendo res.history contiene best per gen)
        if hasattr(res, 'history'):
            history_all_runs.append(res.history)
        else:
            history_all_runs.append([res.fun] * max_gen)  # fallback

        # aggiorna migliore soluzione globale
        if res.fun < best_fun_overall:
            best_fun_overall = res.fun
            best_solution_overall = res

    # --- salva best solution finale ---
    coeff_names = [name for name, _ in qtk.full_terms]
    params_dict = {name: float(val) for name, val in zip(coeff_names, best_solution_overall.x)}
    best_solution_info = {
        "params": params_dict,
        "fun": float(best_solution_overall.fun)
    }

    with open(os.path.join(save_dir, f"qtk_{optimizer.lower()}_best_solution.json"), "w") as f:
        json.dump(best_solution_info, f, indent=4)

    # --- salva kernel finale ---
    K_train_final = qtk.generate_K_train(X_train)
    np.save(os.path.join(save_dir, f"K_train_final_{optimizer}.npy"), K_train_final)

    return best_solution_info, history_all_runs, K_train_final#, K_train_initial


# ===================================
# 1️⃣ Metrics
# ===================================
def compute_metrics(kernel_name, y_true, pred, scores):
    print(f"\n--- Metrics for {kernel_name} Kernel ---")
    bal_acc = balanced_accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, pos_label=-1)
    rec = recall_score(y_true, pred, pos_label=-1)
    f1 = f1_score(y_true, pred, pos_label=-1)
    
    # Inverti lo score per allinearlo al fatto che -1 = anomalia
    scores_auc = -scores
    auc = roc_auc_score((y_true == -1).astype(int), scores_auc)

    print(f"Balanced Accuracy : {bal_acc:.3f}")
    print(f"Precision (anom)  : {prec:.3f}")
    print(f"Recall (anom)     : {rec:.3f}")
    print(f"F1 Score (anom)   : {f1:.3f}")
    print(f"AUC               : {auc:.3f}")

    return {
        "balanced_acc": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc
    }


# ===================================
# 2️⃣ Plot decision scores
# ===================================
def plot_scores(scores_normal, scores_anom, nu=0.05, name='qtk'):
    plt.figure(figsize=(5, 5))
    plt.hist(scores_normal, bins=30, alpha=0.6, label="Normal", density=True)
    plt.hist(scores_anom, bins=30, alpha=0.6, label="Anomalous", density=True)
    plt.axvline(0, color='r', linestyle='--', label='Decision boundary (0)')
    plt.xlabel("Decision function score")
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"Decision scores distribution (kernel={name}, nu={nu})")
    plt.tight_layout()
    plt.savefig(f"images_1/distrib_scores_{name}_{nu}.pdf", dpi=300)
    plt.close()


# ===================================
# 3️⃣ Kernel diagnostics
# ===================================
def kernel_diagnostics(K, optimizer='DE', save_prefix="images_1/kernel_diag"):
    print("\n--- Kernel Diagnostics ---")
    print("K shape:", K.shape)
    print("K min/max:", K.min(), K.max())

    sym_err = np.linalg.norm(K - K.T, ord='fro') / np.linalg.norm(K, ord='fro')
    print(f"Frobenius relative asymmetry: {sym_err:.3e}")

    eigvals = eigh(K, eigvals_only=True)
    print("Eigenvalues min/max:", eigvals.min(), eigvals.max())

    plt.figure(figsize=(5, 4))
    plt.hist(eigvals, bins=40)
    plt.title("Eigenvalues of K")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_{optimizer}_eigvals.pdf', dpi=300)
    plt.close()

    K_sym = (K + K.T) / 2.0
    eigvals, evecs = eigh(K_sym)
    eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None)
    K_psd = (evecs * eigvals_clipped) @ evecs.T

    diag_psd = np.diag(K_psd).copy()
    diag_psd[diag_psd <= 0] = 1e-12
    norm_factor = np.sqrt(np.outer(diag_psd, diag_psd))
    K_norm = np.clip(K_psd / norm_factor, -1.0, 1.0)

    plt.figure(figsize=(5, 5))
    plt.imshow(K_norm, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Normalized PSD-projected kernel (K_norm)")
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_{optimizer}_Knorm.pdf', dpi=300)
    plt.close()

    return K_norm


# ===================================
# 4️⃣ Preprocessing
# ===================================
def normalize_dataframe(df, exclude_cols=["anomaly", "changepoint"]):
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    df_norm = df.copy()
    df_norm[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_norm

def create_windows(df, window_size=900, overlap_size=30):
    windows = []

    if len(df) < window_size:
        windows.append(df)
        return windows

    if "changepoint" in df.columns:
        changepoints = df.index[df["changepoint"] == 1].tolist()
        if len(changepoints) >= 2:
            for i in range(len(changepoints) - 1):
                start = changepoints[i]
                end = changepoints[i + 1]
                segment = df.loc[start:end]
                seg_len = len(segment)
                if seg_len <= window_size:
                    windows.append(segment)
                else:
                    for j in range(0, seg_len - window_size, window_size - overlap_size):
                        windows.append(segment.iloc[j:j + window_size])
            return windows

    # Se non ci sono changepoints o ci sono pochi changepoints, fai lo sliding standard
    for i in range(0, len(df) - window_size, window_size - overlap_size):
        windows.append(df.iloc[i:i + window_size])

    return windows 

def flatten_windows(windows, window_size=900):
    X = []
    for w in windows:
        arr = w.drop(columns=["anomaly", "changepoint"], errors="ignore").values
        # taglia o pad per avere sempre shape (window_size, n_features)
        if arr.shape[0] > window_size:
            arr = arr[:window_size, :]
        elif arr.shape[0] < window_size:
            # pad con l'ultimo valore disponibile
            pad_len = window_size - arr.shape[0]
            pad = np.tile(arr[-1, :], (pad_len, 1))
            arr = np.vstack([arr, pad])
        X.append(arr.flatten())
    return np.array(X)


# ===================================
# 5️⃣ One-Class SVM Quantum/Classical
# ===================================
def run_oneclass_svm_quantum(qtk: QuantumTemporalKernel, K_train, K_test, y_true, nu_min=0.01, nu_max=0.2, n_nu=10):
    """
    Train One-Class SVM with Quantum Temporal Kernel using precomputed kernel.
    Grid search on nu to select the value that minimizes deviance (std) of scores on training set.

    Args:
        qtk: QuantumTemporalKernel instance
        K_train: precomputed training kernel
        K_test: precomputed test kernel
        y_true: ground truth labels (-1 anomaly, 1 normal)
        nu_min: minimum nu for grid search (default 0.01)
        nu_max: maximum nu for grid search (default 0.2)
        n_nu: number of nu values in grid

    Returns:
        oc_best: trained OneClassSVM
        scores_test: decision function scores on test set
        best_nu: selected nu value
        auc: roc_auc_score on test set
    """
    from sklearn.svm import OneClassSVM
    from sklearn.metrics import roc_auc_score
    import numpy as np

    # --- Griglia esponenziale di nu ---
    nus = np.logspace(np.log10(nu_min), np.log10(nu_max), n_nu)

    best_std = np.inf
    best_nu = nus[0]
    oc_best = None

    for nu in nus:
        oc = OneClassSVM(kernel="precomputed", nu=nu).fit(K_train)
        scores_train = oc.decision_function(K_train)
        score_std = np.std(scores_train)

        if score_std < best_std:
            best_std = score_std
            best_nu = nu
            oc_best = oc

    # --- Valutazione sul test set ---
    scores_test = oc_best.decision_function(K_test)
    auc = roc_auc_score((y_true == -1).astype(int), -scores_test)  # invert per allineamento

    print("\n⚛️ One-Class SVM Quantum Temporal Kernel (precomputed)")
    print(f"🔹 Selected nu: {best_nu:.5f}")
    print(f"🚀 Test AUC: {auc:.4f}")

    return oc_best, scores_test, best_nu, auc


def run_oneclass_svm_classical(X_train, X_test, y_true, kernel="rbf", nu_min=0.01, nu_max=0.2, n_nu=10, gamma="scale"):
    """
    Train a One-Class SVM with a classical kernel (RBF, linear, poly, etc.) using an exponential grid search over nu.
    Selects the nu value that minimizes the standard deviation of the training decision scores for more stable results.

    Args:
        X_train: training features
        X_test: test features
        y_true: ground truth labels (-1=anomaly, 1=normal)
        kernel: SVM kernel type
        nu_min, nu_max: exponential range for nu
        n_nu: number of nu values to evaluate
        gamma: gamma parameter for kernels that require it

    Returns:
        oc_best: trained One-Class SVM model
        scores_test: decision function scores on the test set
        best_nu: selected nu value
        auc: ROC-AUC score
    """

    # Generate nu values exponentially spaced
    nus = np.logspace(np.log10(nu_min), np.log10(nu_max), n_nu)
    best_std = np.inf
    best_nu = nus[0]
    oc_best = None

    # Grid search over nu
    for nu in nus:
        oc = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma).fit(X_train)
        scores_train = oc.decision_function(X_train)
        score_std = np.std(scores_train)
        if score_std < best_std:
            best_std = score_std
            best_nu = nu
            oc_best = oc

    # Evaluate on test set
    scores_test = oc_best.decision_function(X_test)
    auc = roc_auc_score((y_true == -1).astype(int), -scores_test)

    print(f"\n--- One-Class SVM {kernel} ---")
    print(f"🔹 Selected nu: {best_nu:.5f}")
    print(f"🚀 Test AUC: {auc:.4f}")

    return oc_best, scores_test, best_nu, auc


def analyze_and_save_kernel(K_train_opt, n_train_norm, n_train_anom, optimizer='DE', output_dir="saved_data"):
    """
        Analyze and save the optimized kernel matrix, the target matrix, and the Frobenius difference.

        Parameters
        ----------
        K_train_opt : np.ndarray
            Optimized kernel matrix.
        n_train_norm : int
            Number of normal windows in the training set.
        n_train_anom : int
            Number of anomalous windows in the training set.
        output_dir : str
            Directory where the images and matrices will be saved.
    """
    # Crea K_target
    K_target = np.zeros_like(K_train_opt)
    K_target[:n_train_norm, :n_train_norm] = 1
    K_target[n_train_norm:n_train_norm + n_train_anom, n_train_norm:n_train_norm + n_train_anom] = 1

    # Distanza Frobenius
    dist_matrix = np.abs(K_train_opt - K_target)
    fro_res = np.linalg.norm(dist_matrix, 'fro')
    print(f"📐 Frobenius distance residual K_opt vs K_target = {fro_res:.4f}")

    # Crea cartella output
    os.makedirs(output_dir, exist_ok=True)

    # Salva matrici
    np.save(os.path.join(output_dir, f"K_train_opt_{optimizer}.npy"), K_train_opt)
    np.save(os.path.join(output_dir, f"K_target_{optimizer}.npy"), K_target)
    np.save(os.path.join(output_dir, f"K_diff_{optimizer}.npy"), dist_matrix)

    # Funzione interna per plotting
    def plot_and_save(matrix, title, filename, cmap='viridis'):
        plt.figure(figsize=(5, 5))
        plt.imshow(matrix, cmap=cmap)
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.savefig(f"images_1/{filename}", dpi=300)
        plt.close()

    # Plot delle matrici
    plot_and_save(K_train_opt, "K_train_opt", f"K_train_opt_{optimizer}.pdf", cmap='viridis')
    plot_and_save(K_target, "K_target", f"K_{optimizer}_target.pdf", cmap='viridis')
    plot_and_save(dist_matrix, "|K_train_opt - K_target|", f"K_{optimizer}_diff.pdf", cmap='magma')

    print(f"✅ Kernel and target saved in {output_dir}/")

# ===================================
# 6️⃣ MAIN
# ===================================
'''def main():
    print("=== ⚙️ START QUANTUM TEMPORAL KERNEL PIPELINE ===")

    # --------------------------
    # Caricamento dataset
    # --------------------------
    valve1_df, valve2_df, other_df, anomaly_free_df = load_skab_by_folder("skab_data/")
    anomaly_free_df = normalize_dataframe(anomaly_free_df)
    valve1_df = [normalize_dataframe(df) for df in valve1_df]
    valve1_concat = pd.concat(valve1_df, ignore_index=True)

    # --------------------------
    # Creazione finestre
    # --------------------------
    window_size = 100
    overlap_size = 40
    windows_af = create_windows(anomaly_free_df, window_size=window_size, overlap_size=overlap_size)
    print(f"Number of normal windows generated: {len(windows_af)}")

    windows_v1_all = create_windows(valve1_concat, window_size=window_size, overlap_size=overlap_size)
    windows_v1_anom = [w for w in windows_v1_all if "anomaly" in w.columns and w["anomaly"].sum() > 0]
    print(f"Number of anomalous windows generated: {len(windows_v1_anom)}")

    # --------------------------
    # Selezione train/test
    # --------------------------
    n_train = 8  # totale per l'ottimizzazione
    n_train_norm = 6
    n_train_anom = 2

    train_windows = windows_af[:n_train_norm] + windows_v1_anom[:n_train_anom]
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(train_windows))
    train_windows = [train_windows[i] for i in indices]


    print(f"✅ Train windows: {len(train_windows)}")

    # Flatten windows
    X_train = [w.drop(columns=["anomaly", "changepoint"], errors="ignore").to_numpy() 
                    for w in train_windows]

    # --------------------------
    # Quantum Temporal Kernel
    # --------------------------
    print("\n⚛️ Start Quantum Temporal Kernel Experiments...")
    start_qtk = time.time()
    qtk = QuantumTemporalKernel(n_qubits=3, L=10, embedding_type="amplitude", n_norm=n_train_norm, n_anom=n_train_anom )
    end_qtk = time.time()
    print(f"🕒 Initializing time for QTK: {end_qtk - start_qtk:.2f}s")

    # --------------------------
    # EVOVAQ optimization
    # --------------------------
    print("\n🧪 Start EVOVAQ optimization on QTK...")
    start_evovaq = time.time()
    optimizer='GA'
    best_sol = optimize_qtk(qtk, X_train,
                            optimizer=optimizer,
                            n_runs=5,
                            pop_size=20,    #Commento: tutti con nruns=5,pop_size=20,maxgen=10 mentre CGA con 2,20,2
                            max_gen=5,
                            param_bounds=(-0.5, 0.5),
                            save_dir="./qtk_evovaq_results/")
    end_evovaq = time.time()
    print(f"🕒 Total time EVOVAQ: {end_evovaq - start_evovaq:.2f}s")
    print("✅ Optimal Parameter founded")


     # --------------------------
    # Selezione train/test
    # --------------------------
    n_train = 70
    n_test_norm = 30
    n_test_anom_2 = 10

    train_windows = windows_af[:n_train]
    test_windows_norm = windows_af[n_train:n_train + n_test_norm]
    test_windows_anom = windows_v1_anom[n_train_anom:n_test_anom_2]

    # Combina e mescola le finestre di test
    X_test_windows = test_windows_norm + test_windows_anom
    y_test = np.array([0]*len(test_windows_norm) + [1]*len(test_windows_anom))
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(X_test_windows))
    X_test_windows = [X_test_windows[i] for i in indices]
    y_test = y_test[indices]

    print(f"✅ Train windows: {len(train_windows)}")
    print(f"✅ Test windows: {len(X_test_windows)} = {sum(y_test==0)} normali + {sum(y_test==1)} anomale")

    X_train = flatten_windows(train_windows)
    X_test = flatten_windows(X_test_windows)
    y_true_bin = np.where(y_test == 1, -1, 1)  # 1=normale, -1=anomalia

    # Rigenera kernel con parametri ottimizzati
    print("\n⚛️ Generation of optimized kernel...")
    K_train_opt = qtk.generate_K_train(X_train)
    K_test_opt = qtk.generate_K_test(X_train, X_test)
    K_train_norm = kernel_diagnostics(K_train_opt, optimizer=optimizer)
    analyze_and_save_kernel(K_train_opt, n_train_norm, n_train_anom, optimizer=optimizer)
    # --------------------------
    # One-Class SVM Quantum
    # --------------------------
    oc_qtk, scores_qtk, best_nu, auc = run_oneclass_svm_quantum(qtk, K_train_norm, K_test_opt, y_true_bin)
    y_pred_qtk = oc_qtk.predict(K_test_opt)

    compute_metrics("Quantum Evolutionary Temporal Kernel", y_true_bin, y_pred_qtk, scores_qtk)
    plot_scores(scores_normal=scores_qtk[y_true_bin == 1],
                scores_anom=scores_qtk[y_true_bin == -1],
                name="QuantumTemporal_EVOVAQ", nu=0.05)

    # --------------------------
    # Classical Kernels
    # --------------------------
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

    print("\n🧠 Training and evaluation of One-Class SVM (classical kernels)...")
    for name, kernel in kernel_dict.items():
        print(f"\n--- {name} Kernel ---")
        start = time.time()
        oc, scores, best_nu, _ = run_oneclass_svm_classical(
            X_train, X_test, y_true_bin, kernel=kernel, nu_min=0.01, nu_max=0.2, n_nu=10
        )
        print("BEST NU: ", best_nu)
        y_pred = oc.predict(X_test)
        compute_metrics(name, y_true_bin, y_pred, scores)
        plot_scores(scores_normal=scores[y_true_bin == 1],
                    scores_anom=scores[y_true_bin == -1],
                    name=name)
    print(f"Time elapsed: {time.time() - start:.2f}s")

    print("\n=== ✅ COMPLETED PIPELINE ===")'''


def main():
    np.random.seed(100)
    X_fake = [np.random.randn(10, 8) for _ in range(4)] # 4 finestre fake
    X_fake_scaled = [(w - w.min()) / (w.max() - w.min()) for w in X_fake]
    qtk = QuantumTemporalKernel(n_qubits=3, L=5, embedding_type="amplitude", n_norm=3, n_anom=1)
    K_train = qtk.generate_K_train(X_fake_scaled)
    print('K_train before optimization:',K_train)
    print('H after', qtk.H)
    coeff = np.random.uniform(-2,2,len(qtk.full_terms))
    print('old coeffs:',[c for (_,c) in qtk.full_terms])
    print('New coeffs:',coeff)
    qtk.set_H_coeffs(coeff)
    K_train = qtk.generate_K_train(X_fake_scaled)
    print('K_train afteyr optimization:',K_train)
    print(K_train)
# Entry point
if __name__ == "__main__":
    main()

