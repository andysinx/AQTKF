# main.py
from preprocessing_hydrogen import *
import os
from qtk import *
from other_kernels import *
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
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



# ===================================
# 1️⃣ Metrics
# ===================================
def compute_metrics(kernel_name, y_true, pred, scores):
    print(f"\n--- Metrics for {kernel_name} Kernel ---")
    bal_acc = balanced_accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, pos_label=-1)
    rec = recall_score(y_true, pred, pos_label=-1)
    f1 = f1_score(y_true, pred, pos_label=-1)
    
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
# Plot decision scores
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
# Kernel diagnostics
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
#One-Class SVM Quantum/Classical
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
        oc = OneClassSVM(kernel="precomputed", nu=nu).fit(K_train)
        y_pred_train = oc.predict(K_train)
        metric = f1_score((y_true_train == -1).astype(int), (y_pred_train == -1).astype(int))

        if  metric > best_metric:
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
        optimizer: "DE", "GA", "PSO", "BBBC", "CGA"
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
        # Aggiorna coefficienti Hamiltoniani
        qtk.set_H_coeffs(params)
        K_train = qtk.generate_K_train(X_train)

        # Target kernel (cluster intra-normal / anom)
        K_target = qtk.K_target
        alignment = np.trace(K_train @ K_target) / (
            np.linalg.norm(K_train, 'fro') * np.linalg.norm(K_target, 'fro')
        )
        return -alignment  # Massimizza alignment -> minimizza negativo

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
    history_all_runs = []
    best_solution_overall = None
    best_fun_overall = np.inf

    for run in tqdm(range(1, n_runs+1), desc=f"{optimizer} runs"):
        res = global_search.optimize(problem, pop_size=pop_size, max_gen=max_gen, verbose=True, seed=42*run)
        
        if hasattr(res, 'history'):
            history_all_runs.append(res.history)
        else:
            history_all_runs.append([res.fun] * max_gen)

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

    return best_solution_info, history_all_runs, K_train_final

# ===================================
#  Main pipeline
# ===================================
def main():
    print("=== ⚙️ START QUANTUM TEMPORAL KERNEL PIPELINE (H2) ===")

    dataset = load_h2_dataset("H2-SimNet")


    window_size = 100
    n_train_norm = 12
    n_train_anom = 3
    X_train, y_train, X_train_final, y_train_final, X_test, y_test = prepare_windows_for_qtk(
        dataset,
        window_size=window_size,
        n_train_norm=n_train_norm,
        n_train_anom=n_train_anom
    )

    y_true_test = np.array(y_test)
    y_final_train = np.array(y_train_final)
    
    for x in range(0,1): #1,20
        # ---------- Quantum Temporal Kernel ----------
        print(f'RUN {x}')
        print("\n⚛️ Start Quantum Temporal Kernel Experiments...")
        start_qtk = time.time()
        qtk = QuantumTemporalKernel(n_qubits=4, L=window_size, embedding_type="amplitude", n_norm=n_train_norm, n_anom=n_train_anom)
        end_qtk = time.time()
        print(f"🕒 Initializing time for QTK: {end_qtk - start_qtk:.2f}s")


        # --------------------------
        # EVOVAQ optimization (opzionale)
        # --------------------------
        print("\n🧪 EVOVAQ optimization on QTK (opzionale)...")
        start_evovaq = time.time()

        optimizer = 'PSO'  # 'DE', 'GA', 'PSO', 'BBBC', 'CGA', 'WITHOUT_OPTIMIZATION'
        K_train_final = None

        if optimizer != 'WITHOUT_OPTIMIZATION':
            best_sol, history, K_train_final = optimize_qtk(
                qtk=qtk,
                X_train=X_train,
                optimizer=optimizer,
                n_runs=5,
                pop_size=20,
                max_gen=10,
                param_bounds=(-0.5, 0.5),
                save_dir="./qtk_evovaq_results_hydrogen/"
            )
            print(best_sol)
            print(history)
            print("✅ EVOVAQ finished. Best solution saved.")
        else:
            print("🔸 Skipping EVOVAQ (optimizer='WITHOUT_OPTIMIZATION'). Will generate K_train from current QTK parameters.")

        end_evovaq = time.time()
        print(f"🕒 Total time EVOVAQ block: {end_evovaq - start_evovaq:.2f}s")

        # Kernel test
        K_train_final = qtk.generate_K_train(X_train_final)
        K_test_final = qtk.generate_K_test(X_train_final, X_test)
        K_train_norm = kernel_diagnostics(K_train_final, optimizer=optimizer)

        # One-Class SVM Quantum
        oc_qtk, scores_qtk, best_nu_qtk, _ = run_oneclass_svm_quantum(qtk, K_train_norm, K_test_final, y_true_test, y_final_train)
        y_pred_qtk = oc_qtk.predict(K_test_final)
        compute_metrics("Quantum EVOVAQ", y_true_test, y_pred_qtk, scores_qtk)
        plot_scores(scores_qtk[y_true_test==1], scores_qtk[y_true_test==-1], name="QuantumTemporal_EVOVAQ", nu=best_nu_qtk)

        # One-Class SVM Classici
        X_train_arr = np.array(X_train_final).reshape(len(X_train_final), -1)
        X_test_arr = np.array(X_test).reshape(len(X_test), -1)
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

        for name, kernel in kernel_dict.items():
            oc, scores, best_nu, _ = run_oneclass_svm_classical(X_train_arr, X_test_arr, y_true_test, y_final_train, kernel=kernel)
            y_pred = oc.predict(X_test_arr)
            compute_metrics(name, y_true_test, y_pred, scores)
            plot_scores(scores[y_true_test==1], scores[y_true_test==-1], name=name)


def run_oneclass_svm_quantum_std(qk, K_train, K_test, y_true_test, y_final_train, nu_min=0.01, nu_max=0.2, n_nu=10):
    """
    One-Class SVM con kernel quantistico precomputato.

    Args:
        qk: QuantumKernel object
        K_train: kernel matrix train (n_train x n_train)
        K_test: kernel matrix test (n_test x n_train)
        y_true_test: etichette test (+1 normal, -1 anomalia)
        y_final_train: etichette train (+1 normal)
        nu_min, nu_max, n_nu: range di ricerca per parametro nu

    Returns:
        oc_svm: modello One-Class SVM addestrato
        scores: decision_function per test set
        best_nu: valore di nu usato
        y_pred: predizioni su test set
    """

    best_nu = 0.05
    best_acc = -1
    nu_values = np.linspace(nu_min, nu_max, n_nu)

    # ricerca grid su nu
    for nu in nu_values:
        oc = OneClassSVM(kernel='precomputed', nu=nu)
        oc.fit(K_train)
        y_pred_train = oc.predict(K_train)
        # valutiamo sul train solo in termini di errori su dati normali (+1)
        acc = np.mean(y_pred_train == 1)
        if acc > best_acc:
            best_acc = acc
            best_nu = nu

    # addestramento finale
    oc_svm = OneClassSVM(kernel='precomputed', nu=best_nu)
    oc_svm.fit(K_train)

    # predizioni test
    scores = oc_svm.decision_function(K_test).ravel()
    y_pred = oc_svm.predict(K_test)

    return oc_svm, scores, best_nu, y_pred



def main1():
    print("=== ⚙️ START QUANTUM KERNEL PIPELINE (H2) ===")

    # --------------------------
    # Caricamento dataset
    # --------------------------
    dataset = load_h2_dataset("H2-SimNet")
    print(f"✅ Dataset loaded with {len(dataset)} samples")

    # --------------------------
    # Preparazione finestre e train/test
    # --------------------------
    window_size = 100
    n_train_norm = 12
    n_train_anom = 3
    X_train, y_train, X_train_final, y_train_final, X_test, y_test = prepare_windows_for_qtk(
        dataset,
        window_size=window_size,
        n_train_norm=n_train_norm,
        n_train_anom=n_train_anom
    )

    y_true_test = np.array(y_test)
    y_final_train = np.array(y_train_final)
    print(f"Prepared windows -> X_train_final: {len(X_train_final)}, X_test: {len(X_test)}")

    # --------------------------
    # Quantum Kernel - init
    # --------------------------
    print("\n⚛️ Initializing Quantum Kernel...")
    n_qubits = X_train_final[0].shape[1]
    qk = QuantumKernel(n_qubits=n_qubits)
    print(f"Quantum Kernel with {n_qubits} qubits initialized")

    # --------------------------
    # Generazione kernel
    # --------------------------
    print("\n🔹 Generating Quantum Kernel matrices...")
    K_train = qk.generate_K_train(X_train_final)
    K_test = qk.generate_K_test(X_train_final, X_test)
    print("✅ Kernel matrices generated")

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

    print("\n=== ✅ COMPLETED QUANTUM KERNEL PIPELINE (H2) ===")


if __name__ == "__main__":
    main()
    main1()
