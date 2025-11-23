import os
import pandas as pd

# ==========================
# Funzione per importare SKAB per cartelle
# ==========================
def load_skab_by_folder(path_to_data="skab_data/"):
    """
    Carica dataset SKAB separati per cartelle:
    - valve1
    - valve2
    - other
    - anomaly-free
    """
    valve1_df = []
    valve2_df = []
    other_df = []
    anomaly_free_df = None

    for root, dirs, files in os.walk(path_to_data):
        for file in files:
            if file.endswith(".csv"):
                full_path = os.path.join(root, file)
                df = pd.read_csv(full_path, sep=";", index_col="datetime", parse_dates=True)

                # Controllo in quale cartella siamo
                if "anomaly-free" in file:
                    anomaly_free_df = df
                elif "valve1" in root.lower():
                    valve1_df.append(df)
                elif "valve2" in root.lower():
                    valve2_df.append(df)
                else:
                    other_df.append(df)
    
    return valve1_df, valve2_df, other_df, anomaly_free_df