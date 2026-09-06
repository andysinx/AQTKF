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
    print("ROC-AUC      :", roc_auc_score((y_true == -1).astype(int), -scores))


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
        datalist_norm[55:105], window_size
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

