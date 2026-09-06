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
    K_train_initial = qtk.generate_K_train(X_train)
    np.save(os.path.join(save_dir, f"K_train_initial_{optimizer}.npy"), K_train_initial)

    def cost_function(params):
        # Update params
        #old_coeffs = [c for (_, c) in qtk.full_terms]
        #print("old_coeffs",old_coeffs)
        #K1 = qtk.generate_K_train(X_train)
        #print("K_train_old:", K1)
        qtk.set_H_coeffs(params)
        #new_coeffs_list = [c for (_, c) in qtk.full_terms]
        #print("new_coeffs_list",new_coeffs_list)
        #diff = np.array(new_coeffs_list) - np.array(old_coeffs)
        #print("Diff coeffs:", diff)
        K_train = qtk.generate_K_train(X_train)
        #print("K_train_new:", K_train)
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


'''# ===================================
# 4️⃣ Preprocessing
# ===================================
def normalize_dataframe(df, exclude_cols=["anomaly", "changepoint"]):
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    df_norm = df.copy()
    df_norm[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df_norm'''

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



# ===================================
# 5️⃣ One-Class SVM Quantum/Classical
# ===================================
def run_oneclass_svm_quantum(qtk: QuantumTemporalKernel, K_train, K_test, y_true, y_true_train, nu_min=0.01, nu_max=0.2, n_nu=10):
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

    best_metric = -np.inf
    best_nu = nus[0]
    oc_best = None

    for nu in nus:
      oc = OneClassSVM(
          kernel=kernel,
          nu=nu,
          gamma=gamma
      ).fit(X_train)
  
      scores_train = oc.decision_function(X_train)
      metric = np.mean(scores_train)
  
      if metric > best_metric:
          best_metric = metric
          best_nu = nu
          oc_best = oc

    # --- Valutazione sul test set ---
    scores_test = oc_best.decision_function(K_test)
    auc = roc_auc_score((y_true == -1).astype(int), -scores_test)  # invert per allineamento

    print("\n⚛️ One-Class SVM Quantum Temporal Kernel (precomputed)")
    print(f"🔹 Selected nu: {best_nu:.5f}")
    print(f"🚀 Test AUC: {auc:.4f}")

    return oc_best, scores_test, best_nu, auc


def run_oneclass_svm_classical(X_train, X_test, y_true, y_true_train, kernel="rbf", nu_min=0.01, nu_max=0.2, n_nu=10, gamma="scale"):
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
    best_metric = -np.inf
    best_nu = nus[0]
    oc_best = None

    # Grid search over nu
    for nu in nus:
        oc = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma).fit(X_train)
        y_pred_train = oc.predict(X_train)
        metric = f1_score((y_true_train == -1).astype(int), (y_pred_train == -1).astype(int))
        if  metric > best_metric:
            best_metric = metric
            best_nu = nu
            oc_best = oc

    # Evaluate on test set
    scores_test = oc_best.decision_function(X_test)
    auc = roc_auc_score((y_true == -1).astype(int), -scores_test)

    print(f"\n--- One-Class SVM {kernel} ---")
    print(f"🔹 Selected nu: {best_nu:.5f}")
    print(f"🚀 Test AUC: {auc:.4f}")

    return oc_best, scores_test, best_nu, auc

def create_unified_dataset(anomaly_free_df, valve1_df, valve2_df, other_df):
    # --- Dati normali ---
    normal_df = anomaly_free_df.copy()
    normal_df["label"] = 1

    # Funzione helper per unire liste di dataframe e trasformare anomaly -> label
    def concat_anomalous(df_list):
        dfs = []
        for df in df_list:
            df_copy = df.drop(columns=["changepoint"], errors="ignore").copy()
            df_copy["label"] = df_copy["anomaly"].apply(lambda x: -1 if x == 1 else 1)
            df_copy = df_copy.drop(columns=["anomaly"], errors="ignore")
            dfs.append(df_copy)
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    # --- Dati anomali ---
    valve1_concat = concat_anomalous(valve1_df)
    valve2_concat = concat_anomalous(valve2_df)
    other_concat = concat_anomalous(other_df)

    # --- Dataset finale ---
    unified_df = pd.concat([normal_df, valve1_concat, valve2_concat, other_concat], ignore_index=True)

    print(f"✅ Unified dataset shape: {unified_df.shape}")
    print(f" - Normal samples: {len(normal_df)}")
    print(f" - Valve1 samples: {len(valve1_concat)}")
    print(f" - Valve2 samples: {len(valve2_concat)}")
    print(f" - Other samples: {len(other_concat)}")
    print(f" - Total anomalous samples: {(unified_df['label'] == -1).sum()}")

    return unified_df


def prepare_windows_for_qtk(unified_df, window_size=100, overlap_size=40,
                            n_train_norm=9, n_train_anom=2,
                            n_final_norm=45,
                            n_test_norm=45, n_test_anom=10,
                            seed=42):

    rng = np.random.default_rng(seed)

    # -------------------------------------------------
    # Separate normal and anomalous observations
    # -------------------------------------------------
    normal_df = unified_df[
        unified_df["label"] == 1
    ].reset_index(drop=True)

    anomalous_df = unified_df[
        unified_df["label"] == -1
    ].reset_index(drop=True)

    # -------------------------------------------------
    # Generate windows
    # -------------------------------------------------
    windows_normal = [
        w for w in create_windows(
            normal_df,
            window_size,
            overlap_size
        )
        if len(w) == window_size
    ]

    windows_anom = [
        w for w in create_windows(
            anomalous_df,
            window_size,
            overlap_size
        )
        if len(w) == window_size
    ]

    print(
        f"Available normal windows: {len(windows_normal)}, "
        f"anomalous windows: {len(windows_anom)}"
    )

    # Number of windows to skip between subsets.
    # With window_size=100 and overlap_size=40,
    # skipping one window guarantees that windows belonging
    # to different subsets do not share observations.
    stride = window_size - overlap_size
    gap_windows = int(np.ceil(overlap_size / stride))

    # =================================================
    # 1. Adaptation set for EvoVAQ
    # =================================================
    X_train = (
        windows_normal[:n_train_norm]
        + windows_anom[:n_train_anom]
    )

    y_train = np.array(
        [1] * n_train_norm
        + [-1] * n_train_anom
    )

    # =================================================
    # 2. Final OCSVM training set
    #    NORMAL WINDOWS ONLY
    # =================================================
    start_norm_final = n_train_norm + gap_windows

    X_train_final = windows_normal[
        start_norm_final:
        start_norm_final + n_final_norm
    ]

    y_train_final = np.ones(
        len(X_train_final),
        dtype=int
    )

    # =================================================
    # 3. Independent test set
    # =================================================
    start_norm_test = (
        start_norm_final
        + n_final_norm
        + gap_windows
    )

    # Anomalous windows used for adaptation must not
    # overlap anomalous windows used for testing.
    start_anom_test = n_train_anom + gap_windows

    test_normal = windows_normal[
        start_norm_test:
        start_norm_test + n_test_norm
    ]

    test_anom = windows_anom[
        start_anom_test:
        start_anom_test + n_test_anom
    ]

    X_test = test_normal + test_anom

    y_test = np.array(
        [1] * len(test_normal)
        + [-1] * len(test_anom)
    )

    # =================================================
    # Shuffle each subset independently
    # =================================================
    def shuffle_windows(X, y):
        indices = rng.permutation(len(X))
        return [X[i] for i in indices], y[indices]

    X_train, y_train = shuffle_windows(
        X_train,
        y_train
    )

    X_train_final, y_train_final = shuffle_windows(
        X_train_final,
        y_train_final
    )

    X_test, y_test = shuffle_windows(
        X_test,
        y_test
    )

    # =================================================
    # Flatten windows
    # =================================================
    def flatten_windows(X):
        return [
            np.array([
                timestep.flatten()
                for timestep in w.drop(
                    columns=["anomaly", "label"],
                    errors="ignore"
                ).to_numpy()
            ])
            for w in X
        ]

    X_train = flatten_windows(X_train)
    X_train_final = flatten_windows(X_train_final)
    X_test = flatten_windows(X_test)

    print(
        f"X_train: {len(X_train)} windows, "
        f"X_train_final: {len(X_train_final)} windows, "
        f"X_test: {len(X_test)} windows"
    )

    return (
        X_train,
        y_train,
        X_train_final,
        y_train_final,
        X_test,
        y_test
    )



# ===================================
# 6️⃣ MAIN
# ===================================
def main():
    print("=== ⚙️ START QUANTUM TEMPORAL KERNEL PIPELINE ===")

    # --------------------------
    # Caricamento dataset
    # --------------------------
    valve1_df, valve2_df, other_df, anomaly_free_df = load_skab_by_folder("skab_data/")
    print(f"✅ Loaded datasets: valve1={len(valve1_df)}, valve2={len(valve2_df)}, other={len(other_df)}, anomaly_free={len(anomaly_free_df)}")
    dataset = create_unified_dataset(anomaly_free_df, valve1_df, valve2_df, other_df)
    print(dataset)

    # --------------------------
    # Statistiche e dimensioni
    # --------------------------
    print("\n🔹 Unified dataset info:")
    print(dataset.info())
    print("\n🔹 Label distribution:")
    print(dataset['label'].value_counts())
    n_rows, n_cols = dataset.shape
    print(f"\nTotal samples: {n_rows}, Features (incl. label): {n_cols}")

    # --------------------------
    # Parametri finestra e creazione train/test per QTK
    # --------------------------
    window_size = 100
    n_train_norm=12
    n_train_anom=3
    X_train, y_train, X_train_final, y_train_final, X_test, y_test = prepare_windows_for_qtk(
      dataset,
      n_train_norm=n_train_norm,
      n_train_anom=n_train_anom
    )

    print(f"\nPrepared windows -> X_train: {len(X_train)}, X_test: {len(X_test)}")
    # labels for SVM functions
    y_true_test = np.array(y_test)
    y_final_train = np.array(y_train_final)

    # --------------------------
    # Quantum Temporal Kernel - init
    # --------------------------
    for x in range(0,1): #1,20
        print(f'RUN {x}')
        print("\n⚛️ Start Quantum Temporal Kernel Experiments...")
        start_qtk = time.time()
        qtk = QuantumTemporalKernel(n_qubits=3, L=window_size, embedding_type="amplitude", n_norm=n_train_norm, n_anom=n_train_anom)
        end_qtk = time.time()
        print(f"🕒 Initializing time for QTK: {end_qtk - start_qtk:.2f}s")

        # --------------------------
        # EVOVAQ optimization (opzionale)
        # --------------------------
        print("\n🧪 EVOVAQ optimization on QTK (opzionale)...")
        start_evovaq = time.time()

        optimizer = 'CGA'  # impostare "GA", "DE","PSO","BBBC","CGA" per eseguire l'ottimizzazione
        K_train_final = None

        if optimizer != 'WITHOUT_OPTIMIZATION':
            best_sol, history, K_train_final = optimize_qtk(
                qtk=qtk,
                X_train=X_train,
                optimizer=optimizer,
                n_runs=5,
                pop_size=20,
                max_gen=50,
                param_bounds=(-0.5, 0.5),
                save_dir="./qtk_evovaq_results/"
            )
            print(best_sol)
            print(history)
            print("✅ EVOVAQ finished. Best solution saved.")
        else:
            print("🔸 Skipping EVOVAQ (optimizer='WITHOUT_OPTIMIZATION'). Will generate K_train from current QTK parameters.")

        end_evovaq = time.time()
        print(f"🕒 Total time EVOVAQ block: {end_evovaq - start_evovaq:.2f}s")

        # --------------------------
        # Kernel generation (train già fornito dall'ottimizzatore se presente)
        # --------------------------
        
        K_train_final = qtk.generate_K_train(X_train_final)
        # genera K_test usando X_train e X_test (dipende dall'implementazione di qtk)
        # qui usiamo la firma qtk.generate_K_test(X_train, X_test) che è quella più comune nel repo
        K_test_final = qtk.generate_K_test(X_train_final, X_test)
        print(K_test_final)
        # eventualmente normalizza/diagnostica il kernel
        try:
            K_train_norm = kernel_diagnostics(K_train_final, optimizer=optimizer)
        except Exception:
            # se kernel_diagnostics non restituisce un kernel normalizzato, teniamo K_train_final
            K_train_norm = K_train_final

        # --------------------------
        # One-Class SVM Quantum
        # --------------------------
        print("\n🔮 Quantum One-Class SVM...")
        oc_qtk, scores_qtk, best_nu_qtk, _ = run_oneclass_svm_quantum(qtk, K_train_norm, K_test_final, y_true_test, y_final_train)
        y_pred_qtk = oc_qtk.predict(K_test_final)

        compute_metrics("Quantum Evolutionary Temporal", y_true_test, y_pred_qtk, scores_qtk)
        plot_scores(scores_normal=scores_qtk[y_true_test == 1], scores_anom=scores_qtk[y_true_test == -1],
                    name="QuantumTemporal_EVOVAQ", nu=best_nu_qtk if 'best_nu_qtk' in locals() else 0.05)

        # --------------------------
        # Classical Kernels
        # --------------------------
        print("\n🧠 Training and evaluation of One-Class SVM (classical kernels)...")
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

        # Prepara versioni numpy flat (non sovrascrivere X_train/X_test originali)
        X_train_arr = np.array(X_train_final)
        X_test_arr = np.array(X_test)

        if X_train_arr.ndim == 3:
            X_train_flat = X_train_arr.reshape(X_train_arr.shape[0], -1)
        else:
            X_train_flat = X_train_arr

        if X_test_arr.ndim == 3:
            X_test_flat = X_test_arr.reshape(X_test_arr.shape[0], -1)
        else:
            X_test_flat = X_test_arr

        # loop sui kernel classici
        for name, kernel in kernel_dict.items():
            print(f"\n--- {name} Kernel ---")
            start_kernel = time.time()

            oc, scores, best_nu, _ = run_oneclass_svm_classical(
                X_train_flat, X_test_flat, y_true_test, y_final_train, kernel=kernel, nu_min=0.01, nu_max=0.2, n_nu=10
            )
            print("BEST NU: ", best_nu)
            y_pred = oc.predict(X_test_flat)

            compute_metrics(name, y_true_test, y_pred, scores)
            plot_scores(scores_normal=scores[y_true_test == 1], scores_anom=scores[y_true_test == -1], name=name)

            print(f"Time elapsed ({name}): {time.time() - start_kernel:.2f}s")

        print("\n=== ✅ COMPLETED PIPELINE ===")


def run_oneclass_svm_quantum_std(qk: QuantumKernel, K_train, K_test, y_true, y_true_train, nu_min=0.01, nu_max=0.2, n_nu=10):
    nus = np.logspace(np.log10(nu_min), np.log10(nu_max), n_nu)
    best_metric = -np.inf
    best_nu = nus[0]
    oc_best = None

    for nu in nus:
      oc = OneClassSVM(kernel="precomputed", nu=nu).fit(K_train)
  
      scores_train = oc.decision_function(K_train)
  
      # Preferisce il modello che assegna score più elevati/stabili
      # ai campioni normali del training set.
      metric = np.mean(scores_train)
  
      if metric > best_metric:
          best_metric = metric
          best_nu = nu
          oc_best = oc

    scores_test = oc_best.decision_function(K_test)
    auc = roc_auc_score((y_true == -1).astype(int), -scores_test)  # invert per allineamento

    print("\n⚛️ One-Class SVM Quantum Kernel (precomputed)")
    print(f"🔹 Selected nu: {best_nu:.5f}")
    print(f"🚀 Test AUC: {auc:.4f}")

    return oc_best, scores_test, best_nu, auc



def main1():
    print("=== ⚙️ START QUANTUM KERNEL PIPELINE ===")

    # --------------------------
    # Caricamento dataset
    # --------------------------
    valve1_df, valve2_df, other_df, anomaly_free_df = load_skab_by_folder("skab_data/")
    dataset = create_unified_dataset(anomaly_free_df, valve1_df, valve2_df, other_df)

    # --------------------------
    # Preparazione finestra e train/test
    # --------------------------
    window_size = 100
    n_train_norm = 12
    n_train_anom = 3
    X_train, y_train, X_train_final, y_train_final, X_test, y_test = prepare_windows_for_qtk(
      dataset,
      n_train_norm=n_train_norm,
      n_train_anom=n_train_anom
    )

    y_true_test = np.array(y_test)
    y_final_train = np.array(y_train_final)

    # --------------------------
    # Quantum Kernel - init
    # --------------------------
    print("\n⚛️ Initializing Quantum Kernel...")
    start_qk = time.time()
    n_qubits = X_train_final[0].shape[1]
    qk = QuantumKernel(n_qubits=n_qubits)
    end_qk = time.time()
    print(f"🕒 Initialization time: {end_qk - start_qk:.2f}s")

    # --------------------------
    # Generazione kernel
    # --------------------------
    print("\n🔹 Generating Quantum Kernel matrices...")
    start_kernel = time.time()
    K_train = qk.generate_K_train(X_train_final)
    K_test = qk.generate_K_test(X_train_final, X_test)
    end_kernel = time.time()
    print(f"🕒 Kernel generation time: {end_kernel - start_kernel:.2f}s")

    # --------------------------
    # One-Class SVM Quantum
    # --------------------------
    print("\n🔮 Running Quantum One-Class SVM...")
    oc_qk, scores_qk, best_nu_qk, _ = run_oneclass_svm_quantum_std(
        qk, K_train, K_test, y_true_test, y_final_train
    )
    y_pred_qk = oc_qk.predict(K_test)

    compute_metrics("Quantum Kernel", y_true_test, y_pred_qk, scores_qk)
    plot_scores(
        scores_normal=scores_qk[y_true_test == 1],
        scores_anom=scores_qk[y_true_test == -1],
        name="QuantumKernel",
        nu=best_nu_qk
    )

    print("\n=== ✅ COMPLETED QUANTUM KERNEL PIPELINE ===")

# Entry point
if __name__ == "__main__":
    main()
    main1()

