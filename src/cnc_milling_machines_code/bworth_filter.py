import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from tqdm import tqdm


class ButterworthFilter:
    def __init__(self, fs=2000, cutoff=75, order=4, axes_labels=('X', 'Y', 'Z')):
        """
        Class to apply a Butterworth high-pass filter to multi-axis signals.

        Parameters
        ----------
        fs : float
            Sampling frequency (Hz).
        cutoff : float
            Cutoff frequency of the high-pass filter (Hz).
        order : int
            Filter order.
        axes_labels : tuple of str
            Labels for the signal axes (e.g., ('X', 'Y', 'Z')).
        """
        self.fs = fs
        self.cutoff = cutoff
        self.order = order
        self.axes_labels = axes_labels

    # === HIGH-PASS FILTER ===
    def butter_highpass(self, x):
        """Apply a Butterworth high-pass filter to a 1D signal."""
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = signal.butter(self.order, normal_cutoff, btype='high', analog=False)
        return signal.filtfilt(b, a, x)

    # === METRICS ===
    @staticmethod
    def compute_metrics(x):
        """Compute basic signal metrics."""
        return {
            'max': np.max(x),
            'min': np.min(x),
            'mean': np.mean(x),
            'std': np.std(x),
            'energy': np.sum(x**2),
            'kurtosis': stats.kurtosis(x, fisher=False)
        }

    # === PSD IN dB (NO OFFSET) ===
    @staticmethod
    def psd_plot_db(x, fs=2000, ax=None, label=None):
        """Plot the Power Spectral Density (PSD) in decibels."""
        f, Pxx = signal.welch(x, fs=fs, nperseg=4096)
        Pxx_db = 10 * np.log10(Pxx)
        ax.semilogx(f, Pxx_db, label=label, linewidth=1.5)
        ax.set_ylabel('PSD [dB]')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # === MAIN PROCESS ===
    def process_and_plot_signals(self, datalist, start_idx=31, end_idx=69, plot_first=True, type='good'):
        """
        Apply high-pass filtering, compute metrics, and generate time/PSD plots for multi-axis signals.

        Parameters
        ----------
        datalist : list of np.ndarray
            List of N×3 arrays containing the signals.
        start_idx : int
            Starting index of the signals to process.
        end_idx : int
            Ending index (exclusive) of the signals to process.
        plot_first : bool
            If True, plots are shown only for the first signal.

        Returns
        -------
        filtered_signals : list of np.ndarray
            List of filtered signals.
        relevant_data : list of np.ndarray
            List of original signals processed.
        """
        relevant_data = datalist[start_idx:end_idx]
        filtered_signals = []

        for k, el in enumerate(tqdm(relevant_data, desc=f"..Filtering {type} signals..")):
            # === FILTERING ===
            filtered_data = np.zeros_like(el)
            for i in range(3):
                filtered_data[:, i] = self.butter_highpass(el[:, i])
            filtered_signals.append(filtered_data)

            # === METRICS ===
            #print(f"\n=== Signal {start_idx + k} ===")
            for i, label in enumerate(self.axes_labels):
                orig = self.compute_metrics(el[:, i])
                filt = self.compute_metrics(filtered_data[:, i])
                '''print(f"{label}-axis: max {orig['max']:.2f} → {filt['max']:.2f}, "
                      f"std {orig['std']:.2f} → {filt['std']:.2f}")'''

            # === PLOTS FOR THE FIRST SIGNAL ONLY ===
            if plot_first and k == 0:
                t = np.arange(len(el)) / self.fs

                # Time-domain plot
                fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
                for i, label in enumerate(self.axes_labels):
                    axs[i].plot(t, el[:, i], label='Original', linewidth=1.5)
                    axs[i].plot(t, filtered_data[:, i], label='High-pass Filtered', linewidth=1.5)
                    axs[i].set_ylabel(f'{label}-axis')
                    axs[i].grid()
                axs[0].legend()
                axs[-1].set_xlabel('Time [s]')
                fig.suptitle(f'Time-domain: Original vs High-pass Filtered ({self.cutoff} Hz)')
                plt.savefig('image/orig_vs_highpassfilt.png',dpi=300)

                # PSD plot
                fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
                for i, label in enumerate(self.axes_labels):
                    self.psd_plot_db(el[:, i], fs=self.fs, ax=axs[i], label='Original')
                    self.psd_plot_db(filtered_data[:, i], fs=self.fs, ax=axs[i], label='High-pass Filtered')
                    axs[i].legend()
                    axs[i].set_title(f'{label}-axis PSD')
                axs[-1].set_xlabel('Frequency [Hz]')
                fig.suptitle(f'PSD Comparison: Original vs High-pass Filtered ({self.cutoff} Hz)', fontsize=14)
                plt.savefig('image/psd_comparison.png',dpi=300)

        return filtered_signals, relevant_data