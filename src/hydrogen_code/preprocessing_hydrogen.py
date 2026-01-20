# main.py
from preprocessing_hydrogen import *
from sklearn.metrics import roc_auc_score
import numpy as np
from ChaothicEnhancedGeneticAlgorithm import CGA
from pathlib import Path
import pandas as pd
# ===================================
# 0️⃣ H2 Dataset loader
# ===================================
def apply_moving_average(df, cols, window=5):
    df_ma = df.copy()
    df_ma[cols] = df_ma[cols].rolling(window=window, min_periods=1).mean()
    return df_ma

def load_h2_dataset(base_folder="H2-SimNet"):

    # -----------------------------
    # 0️⃣ Colonne di feature
    # -----------------------------
    feature_cols = [f"PS{i}" for i in range(1,13)] + [f"MFS{i}" for i in range(1,5)]

    # -----------------------------
    # 1️⃣ Normal samples
    # -----------------------------
    normal_paths = list(Path(base_folder, "normality-scenarios/parquet").glob("*.parquet"))
    normal_list = []
    for f in normal_paths:
        df = pd.read_parquet(f)
        df[feature_cols] = df[feature_cols].rolling(window=5, min_periods=1).mean()  # moving average
        df["label"] = 1
        normal_list.append(df)
    df_norm = pd.concat(normal_list, ignore_index=True)
    print(f"✅ Caricate {len(df_norm)} righe normali da {len(normal_paths)} file")

    # -----------------------------
    # 2️⃣ Anomalous samples
    # -----------------------------
    anomalous_base = Path(base_folder, "anomalous-scenarios/parquet")
    topologies = ["anom_gestdown", "anom_gestcent", "anom_gestup"]
    files_ordered = [
        "assestment_transient/leakG4.parquet",
        "assestment_transient/leakG5G6_overlapped.parquet",
        "constant_segment/brokenC1G5_weak.parquet",
        "constant_segment/brokenC1_strong.parquet",
        "constant_segment/brokenC1_weak.parquet",
        "constant_segment/leakG3.parquet",
        "constant_segment/leakG3G4_strong.parquet",
        "constant_segment/leakG3G4_weak.parquet",
        "constant_segment/leakG3G5_overlapped.parquet",
        "constant_segment/leakG3G6.parquet",
        "constant_segment/leakG4_strong.parquet",
        "constant_segment/leakG4_weak.parquet",
        "constant_segment/leakG5_strong.parquet",
        "constant_segment/leakG5_weak.parquet",
        "constant_segment/leakG6.parquet",
        "constant_segment/startdelayC1.parquet",
        "start_transient/leakG3.parquet",
        "start_transient/leakG4brokenC1.parquet"
    ]

    anom_list = []

    for topo in topologies:
        topo_path = anomalous_base / topo
        for f in files_ordered:
            file_path = topo_path / f
            if not file_path.exists():
                print(f"⚠️ File mancante: {file_path}")
                continue

            df = pd.read_parquet(file_path)

            # Se non c'è la colonna label_anomaly, salta
            if "label_anomaly" not in df.columns:
                print(f"⚠️ File {file_path} non ha la colonna 'label_anomaly'")
                continue

            # Normalizza la colonna label_anomaly in 0/1 intero
            df["label_anomaly"] = df["label_anomaly"].astype(int)
            if df["label_anomaly"].sum() == 0:
                print(f"⚠️ File {file_path} non ha righe con label_anomaly==1")
                continue

            # Applica media mobile
            df[feature_cols] = df[feature_cols].rolling(window=5, min_periods=1).mean()

            # Prendi solo da primo punto anomalo in poi
            first_anom_idx = int(df.index[df["label_anomaly"]==1].min())
            df_anom = df.iloc[first_anom_idx:].copy()
            df_anom["label"] = -1
            df_anom = df_anom.drop(columns=["label_anomaly"], errors="ignore")
            anom_list.append(df_anom)

    df_anom_all = pd.concat(anom_list, ignore_index=True) if anom_list else pd.DataFrame()
    print(f"Totale righe anomale aggregate: {len(df_anom_all)}")

    # -----------------------------
    # 3️⃣ Concatenazione finale
    # -----------------------------
    unified_df = pd.concat([df_norm, df_anom_all], ignore_index=True)
    print(f"✅ Unified H2 dataset shape: {unified_df.shape}")
    print(f" - Normal samples: {len(df_norm)}")
    print(f" - Anomalous samples: {(unified_df['label']==-1).sum()}")

    return unified_df


# ===================================
# Create windows for QTK
# ===================================
def create_windows(df, window_size=100, overlap_size=30):
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

    for i in range(0, len(df) - window_size, window_size - overlap_size):
        windows.append(df.iloc[i:i + window_size])

    return windows

def prepare_windows_for_qtk(unified_df, window_size=100, overlap_size=40,
                            n_train_norm=9, n_train_anom=2,
                            n_final_norm=45, n_final_anom=10,
                            n_test_norm=45, n_test_anom=10,
                            seed=42):

    rng = np.random.default_rng(seed)

    # -----------------------------
    # 1️⃣ Separa X e y SUBITO
    # -----------------------------
    feature_cols = [f"PS{i}" for i in range(1,13)] + [f"MFS{i}" for i in range(1,5)]

    X_df = unified_df[feature_cols].reset_index(drop=True)
    y_all = unified_df["label"].values

    # Split normal / anomalous sugli INDICI
    idx_norm = np.where(y_all == 1)[0]
    idx_anom = np.where(y_all == -1)[0]

    X_norm = X_df.iloc[idx_norm].reset_index(drop=True)
    X_anom = X_df.iloc[idx_anom].reset_index(drop=True)

    # -----------------------------
    # 2️⃣ Finestre SOLO su X
    # -----------------------------
    windows_normal = [
        w for w in create_windows(X_norm, window_size, overlap_size)
        if len(w) == window_size
    ]
    windows_anom = [
        w for w in create_windows(X_anom, window_size, overlap_size)
        if len(w) == window_size
    ]
    
    print(f"Available normal windows: {len(windows_normal)}, anomalous windows: {len(windows_anom)}")

    # -----------------------------
    # 3️⃣ Split train / final / test
    # -----------------------------
    X_train = windows_normal[:n_train_norm] + windows_anom[:n_train_anom]
    y_train = np.array([1]*n_train_norm + [-1]*n_train_anom)

    X_train_final = (
        windows_normal[n_train_norm:n_train_norm+n_final_norm] +
        windows_anom[n_train_anom:n_train_anom+n_final_anom]
    )
    y_train_final = np.array([1]*n_final_norm + [-1]*n_final_anom)

    X_test = (
        windows_normal[n_train_norm+n_final_norm:n_train_norm+n_final_norm+n_test_norm] +
        windows_anom[n_train_anom+n_final_anom:n_train_anom+n_final_anom+n_test_anom]
    )
    y_test = np.array([1]*n_test_norm + [-1]*n_test_anom)

    # -----------------------------
    # 4️⃣ Shuffle (DOPO finestre)
    # -----------------------------
    def shuffle(X, y):
        idx = rng.permutation(len(X))
        return [X[i] for i in idx], y[idx]

    X_train, y_train = shuffle(X_train, y_train)
    X_train_final, y_train_final = shuffle(X_train_final, y_train_final)
    X_test, y_test = shuffle(X_test, y_test)

    # -----------------------------
    # 5️⃣ Conversione finale → numpy
    # -----------------------------
    def to_numpy_windows(X):
        return [w.to_numpy(dtype=np.float64) for w in X]

    X_train = to_numpy_windows(X_train)
    X_train_final = to_numpy_windows(X_train_final)
    X_test = to_numpy_windows(X_test)

    # -----------------------------
    # 6️⃣ CHECK DI SANITÀ 🔥
    # -----------------------------
    for name, X in zip(
        ["train", "train_final", "test"],
        [X_train, X_train_final, X_test]
    ):
        assert X[0].shape[1] == 16, f"❌ {name}: found {X[0].shape[1]} features"

    print("✅ Windows prepared correctly (16 features, labels separated)")

    return X_train, y_train, X_train_final, y_train_final, X_test, y_test

