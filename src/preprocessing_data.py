import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from scipy import signal, stats
from tqdm import tqdm


def find_all_h5s_in_dir(s_dir: os.PathLike):
    """
    list all .h5 files in a directory and subdirectories
    """
    fileslist = []
    for root, _, files in os.walk(s_dir):
        for file in files:
            if file.endswith(".h5"):
                fileslist.append(os.path.join(root, file))  # percorso completo!
    return fileslist


def load_tool_research_data(data_path: os.PathLike, label: Literal["good","bad"], add_additional_label: bool =True, verbose: bool =True):
    """
    load data (good and bad) from the research data storages
    
    Keyword Arguments:
            data_path {str} -- [path to the directory] 
            label {str} -- ["good" or "bad"]
            add_additional_label {bool} -- [if true the labels will be in the form of "Mxx_Aug20xx_OPxx_sampleNr_label" otherwise "label"] (default: True)
            verbose {bool}

        Returns:
            datalist --  [list of the the X samples]
            label --  [list of the the y labels ]
    """
    datalist = []
    data_label = []

    # list all .h5 files
    list_paths = find_all_h5s_in_dir(data_path)
    list_paths.sort()
    if not list_paths and verbose:
        print(f"skipping {data_path} empty directory...")

    # read and append the samples with the corresponding labels
    if verbose:
        print(f"loading files from {data_path}... ")
    for element in tqdm(list_paths, desc="...Processing files..."):
        # check if additional label needed ("Mxx_Aug20xx_Tool,nrX") 
        if add_additional_label:
            add_label = element.split('/')[-1]
            additional_label = add_label[:-3] + "_" + label
        else:
            additional_label = label
        # extract data X and y 
        with h5py.File(element, 'r') as f:
            vibration_data = f['vibration_data'][:]
        datalist.append(vibration_data)
        data_label.append(additional_label)

    return datalist, data_label


def data_printing(vibration_data: np.ndarray, plotting=True):
    """Plots vibration data already loaded in memory, identico allo stile originale.

    Keyword Arguments:
        vibration_data {ndarray} -- array dei dati già caricati, shape (N, 3)
        plotting {bool} -- se True mostra i grafici (default: True)

    Returns:
        ndarray -- dati originali (stesso input)
    """
    # interpolation for x axis plot
    freq = 2000
    samples_s = len(vibration_data[:, 0]) / freq
    samples = np.linspace(0, samples_s, len(vibration_data[:, 0]))

    # plotting identico alla funzione originale
    if plotting:
        plt.figure(figsize=(20, 5))
        plt.plot(samples, vibration_data[:, 0], 'b')
        plt.ylabel('X-axis Vibration Data')
        plt.xlabel('Time [sec]')
        plt.locator_params(axis='y', nbins=10)
        plt.grid()
        plt.savefig('image/img1.png',dpi=300)
        
        plt.figure(figsize=(20, 5))
        plt.plot(samples, vibration_data[:, 1], 'b')
        plt.ylabel('Y-axis Vibration Data')
        plt.xlabel('Time [sec]')
        plt.locator_params(axis='y', nbins=10)
        plt.grid()
        plt.savefig('image/img2.png',dpi=300)
        
        plt.figure(figsize=(20, 5))
        plt.plot(samples, vibration_data[:, 2], 'b')
        plt.ylabel('Z-axis Vibration Data')
        plt.xlabel('Time [sec]')
        plt.locator_params(axis='y', nbins=10)
        plt.grid()
        plt.savefig('image/img3.png',dpi=300)
    
    return vibration_data


def select_random_windows(data_list, window_size=None, random_seed=None, verbose=True):
    """
    Divide ogni segmento in finestre di lunghezza `window_size` e prende
    una finestra casuale per segmento.

    Parameters
    ----------
    data_list : list of np.ndarray
        Lista di array 2D (N_i, C) rappresentanti segmenti del segnale.
    window_size : int
        Lunghezza delle finestre in campioni.
    random_seed : int or None
        Seed per riproducibilità.
    verbose : bool
        Se True, stampa informazioni diagnostiche.

    Returns
    -------
    selected_windows : np.ndarray
        Array di shape (num_segments, window_size, C) con finestre casuali.
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)

    selected_windows = []

    for idx, seg in enumerate(data_list):
        N, C = seg.shape
        if N < window_size:
            if verbose:
                print(f"Segment {idx} troppo corto ({N} < {window_size}), saltato.")
            continue

        # Scegli un punto di partenza casuale valido
        start = np.random.randint(0, N - window_size + 1)
        end = start + window_size
        selected_windows.append(seg[start:end])

    selected_windows = np.array(selected_windows)

    if verbose:
        print(f"Selezionate {len(selected_windows)} finestre casuali (una per segmento).")

    return selected_windows