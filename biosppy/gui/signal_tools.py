"""
Signal Processing Tools
=======================

Advanced signal processing tools including:
- Filter design and application
- Signal smoothing
- Resampling and normalization
- Detrending
- Signal synthesis
- Signal comparison
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np


class FilterDesignDialog(tk.Toplevel):
    """Advanced filter design dialog."""

    def __init__(self, parent, main_window):
        """Initialize filter design dialog."""
        super().__init__(parent)
        self.main_window = main_window

        self.title("Advanced Filter Design")
        self.geometry("600x500")

        self._create_widgets()

    def _create_widgets(self):
        """Create filter design widgets."""
        # Filter type
        type_frame = ttk.LabelFrame(self, text="Filter Type", padding=10)
        type_frame.pack(fill=tk.X, padx=10, pady=5)

        self.filter_type = tk.StringVar(value="butter")
        filters = [
            ("Butterworth", "butter"),
            ("Chebyshev I", "cheby1"),
            ("Chebyshev II", "cheby2"),
            ("Elliptic", "ellip"),
            ("Bessel", "bessel"),
        ]

        for text, value in filters:
            ttk.Radiobutton(type_frame, text=text,
                           variable=self.filter_type,
                           value=value).pack(anchor=tk.W)

        # Band type
        band_frame = ttk.LabelFrame(self, text="Band Type", padding=10)
        band_frame.pack(fill=tk.X, padx=10, pady=5)

        self.band_type = tk.StringVar(value="bandpass")
        bands = [
            ("Lowpass", "lowpass"),
            ("Highpass", "highpass"),
            ("Bandpass", "bandpass"),
            ("Bandstop", "bandstop"),
        ]

        for text, value in bands:
            ttk.Radiobutton(band_frame, text=text,
                           variable=self.band_type,
                           value=value).pack(anchor=tk.W)

        # Parameters
        params_frame = ttk.LabelFrame(self, text="Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(params_frame, text="Order:").grid(row=0, column=0, sticky=tk.W)
        self.filter_order = ttk.Spinbox(params_frame, from_=1, to=20, width=20)
        self.filter_order.set(4)
        self.filter_order.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(params_frame, text="Low Cutoff (Hz):").grid(row=1, column=0,
                                                              sticky=tk.W)
        self.low_cutoff = ttk.Entry(params_frame, width=20)
        self.low_cutoff.insert(0, "0.5")
        self.low_cutoff.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(params_frame, text="High Cutoff (Hz):").grid(row=2, column=0,
                                                               sticky=tk.W)
        self.high_cutoff = ttk.Entry(params_frame, width=20)
        self.high_cutoff.insert(0, "40")
        self.high_cutoff.grid(row=2, column=1, padx=5, pady=2)

        # Options
        options_frame = ttk.LabelFrame(self, text="Options", padding=10)
        options_frame.pack(fill=tk.X, padx=10, pady=5)

        self.zero_phase = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Zero-phase filtering (filtfilt)",
                       variable=self.zero_phase).pack(anchor=tk.W)

        self.plot_response = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Plot frequency response",
                       variable=self.plot_response).pack(anchor=tk.W)

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="Apply Filter",
                  command=self._apply_filter).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Design Only",
                  command=self._design_filter).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.RIGHT)

    def _design_filter(self):
        """Design filter and show response."""
        try:
            from biosppy.signals import tools
            import scipy.signal as ss

            # Get parameters
            order = int(self.filter_order.get())
            low_freq = float(self.low_cutoff.get())
            high_freq = float(self.high_cutoff.get())

            # Get sampling rate from selected signal
            selection = self.main_window.signal_listbox.curselection()
            if selection:
                idx = selection[0]
                signal_name = self.main_window.signal_listbox.get(idx)
                signal_data = self.main_window.signal_manager.get_signal(signal_name)
                sampling_rate = signal_data.get('sampling_rate', 1000)
            else:
                sampling_rate = 1000

            # Design filter
            ftype = self.filter_type.get()
            band = self.band_type.get()

            if band in ['lowpass', 'highpass']:
                cutoff = high_freq if band == 'lowpass' else low_freq
                b, a = ss.butter(order, cutoff, btype=band, fs=sampling_rate)
            else:
                b, a = ss.butter(order, [low_freq, high_freq], btype=band,
                               fs=sampling_rate)

            # Plot frequency response if requested
            if self.plot_response.get():
                w, h = ss.freqz(b, a, fs=sampling_rate)

                import matplotlib.pyplot as plt
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

                # Magnitude
                ax1.plot(w, 20 * np.log10(abs(h)))
                ax1.set_xlabel('Frequency (Hz)')
                ax1.set_ylabel('Magnitude (dB)')
                ax1.set_title(f'{ftype.title()} {band.title()} Filter Response')
                ax1.grid(True)

                # Phase
                ax2.plot(w, np.angle(h) * 180 / np.pi)
                ax2.set_xlabel('Frequency (Hz)')
                ax2.set_ylabel('Phase (degrees)')
                ax2.grid(True)

                plt.tight_layout()
                plt.show()

            messagebox.showinfo("Success", "Filter designed successfully!")

        except Exception as e:
            messagebox.showerror("Error", f"Filter design failed:\n{str(e)}")

    def _apply_filter(self):
        """Apply designed filter to signal."""
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "No signal selected")
            return

        try:
            from biosppy.signals import tools

            idx = selection[0]
            signal_name = self.main_window.signal_listbox.get(idx)
            signal_data = self.main_window.signal_manager.get_signal(signal_name)

            if signal_data is None:
                return

            signal = signal_data['signal']
            sampling_rate = signal_data.get('sampling_rate', 1000)

            # Get parameters
            order = int(self.filter_order.get())
            low_freq = float(self.low_cutoff.get())
            high_freq = float(self.high_cutoff.get())
            band = self.band_type.get()

            # Apply filter
            if band in ['lowpass', 'highpass']:
                frequency = [high_freq if band == 'lowpass' else low_freq]
            else:
                frequency = [low_freq, high_freq]

            filtered, _, _ = tools.filter_signal(
                signal=signal,
                ftype=self.filter_type.get(),
                band=band,
                order=order,
                frequency=frequency,
                sampling_rate=sampling_rate
            )

            # Update signal
            self.main_window.signal_manager.update_signal(
                signal_name,
                filtered,
                f"{self.filter_type.get().title()} {band} filter "
                f"({low_freq}-{high_freq} Hz)"
            )

            # Refresh plot
            self.main_window.refresh_plot()
            self.main_window.statusbar.set_message("Filter applied successfully")

            self.destroy()

        except Exception as e:
            messagebox.showerror("Error", f"Filter application failed:\n{str(e)}")


class SignalSynthesisDialog(tk.Toplevel):
    """Dialog for synthesizing biosignals."""

    def __init__(self, parent, main_window):
        """Initialize signal synthesis dialog."""
        super().__init__(parent)
        self.main_window = main_window

        self.title("Signal Synthesis")
        self.geometry("500x400")

        self._create_widgets()

    def _create_widgets(self):
        """Create synthesis widgets."""
        # Signal type
        type_frame = ttk.LabelFrame(self, text="Signal Type", padding=10)
        type_frame.pack(fill=tk.X, padx=10, pady=5)

        self.signal_type = tk.StringVar(value="ecg")
        signal_types = [
            ("ECG", "ecg"),
            ("EMG", "emg"),
            ("Noise", "noise"),
        ]

        for text, value in signal_types:
            ttk.Radiobutton(type_frame, text=text,
                           variable=self.signal_type,
                           value=value).pack(anchor=tk.W)

        # Parameters
        params_frame = ttk.LabelFrame(self, text="Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(params_frame, text="Duration (seconds):").grid(row=0,
                                                                 column=0,
                                                                 sticky=tk.W)
        self.duration = ttk.Entry(params_frame, width=20)
        self.duration.insert(0, "10")
        self.duration.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(params_frame, text="Sampling Rate (Hz):").grid(row=1,
                                                                 column=0,
                                                                 sticky=tk.W)
        self.sampling_rate = ttk.Entry(params_frame, width=20)
        self.sampling_rate.insert(0, "1000")
        self.sampling_rate.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(params_frame, text="Heart Rate (bpm):").grid(row=2,
                                                               column=0,
                                                               sticky=tk.W)
        self.heart_rate = ttk.Entry(params_frame, width=20)
        self.heart_rate.insert(0, "70")
        self.heart_rate.grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(params_frame, text="Signal Name:").grid(row=3, column=0,
                                                          sticky=tk.W)
        self.signal_name = ttk.Entry(params_frame, width=20)
        self.signal_name.insert(0, "synthetic_signal")
        self.signal_name.grid(row=3, column=1, padx=5, pady=2)

        # Options
        options_frame = ttk.LabelFrame(self, text="Options", padding=10)
        options_frame.pack(fill=tk.X, padx=10, pady=5)

        self.add_noise = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Add noise",
                       variable=self.add_noise).pack(anchor=tk.W)

        ttk.Label(options_frame, text="Noise level (SNR dB):").pack(anchor=tk.W)
        self.noise_level = ttk.Spinbox(options_frame, from_=0, to=50, width=20)
        self.noise_level.set(20)
        self.noise_level.pack(anchor=tk.W, padx=20)

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="Synthesize",
                  command=self._synthesize).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.RIGHT)

    def _synthesize(self):
        """Synthesize signal."""
        try:
            signal_type = self.signal_type.get()
            duration = float(self.duration.get())
            sampling_rate = float(self.sampling_rate.get())
            heart_rate = float(self.heart_rate.get())
            name = self.signal_name.get()

            if signal_type == "ecg":
                from biosppy.synthesizers import ecg as ecg_synth
                signal, _ = ecg_synth.ecg(duration=duration,
                                         sampling_rate=sampling_rate,
                                         heart_rate=heart_rate)

            elif signal_type == "emg":
                from biosppy.synthesizers import emg as emg_synth
                burst_duration = 1.0
                signal = emg_synth.synth_gaussian(
                    duration=duration,
                    sampling_rate=sampling_rate,
                    burst_duration=burst_duration
                )

            elif signal_type == "noise":
                n_samples = int(duration * sampling_rate)
                signal = np.random.randn(n_samples)

            # Add noise if requested
            if self.add_noise.get() and signal_type != "noise":
                snr_db = float(self.noise_level.get())
                signal_power = np.mean(signal ** 2)
                noise_power = signal_power / (10 ** (snr_db / 10))
                noise = np.sqrt(noise_power) * np.random.randn(len(signal))
                signal = signal + noise

            # Add to signal manager
            self.main_window.signal_manager.add_signal(
                name,
                signal,
                signal_type=signal_type.upper(),
                sampling_rate=sampling_rate,
                units='mV'
            )

            # Add to listbox
            self.main_window.add_signal_to_list(name)

            # Update status
            self.main_window.statusbar.set_message(
                f"Synthesized {signal_type.upper()} signal: {name}"
            )

            self.destroy()

        except Exception as e:
            messagebox.showerror("Error", f"Signal synthesis failed:\n{str(e)}")


class SignalComparisonDialog(tk.Toplevel):
    """Dialog for comparing multiple signals."""

    def __init__(self, parent, main_window):
        """Initialize signal comparison dialog."""
        super().__init__(parent)
        self.main_window = main_window

        self.title("Signal Comparison")
        self.geometry("500x400")

        self._create_widgets()

    def _create_widgets(self):
        """Create comparison widgets."""
        info_label = ttk.Label(self, text="Compare Multiple Signals",
                              font=("Arial", 12, "bold"))
        info_label.pack(pady=10)

        # Signal selection
        selection_frame = ttk.LabelFrame(self, text="Select Signals", padding=10)
        selection_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Get all signals
        signals = self.main_window.signal_manager.get_all_signals()

        self.signal_vars = {}
        for signal_name in signals:
            var = tk.BooleanVar(value=False)
            ttk.Checkbutton(selection_frame, text=signal_name,
                           variable=var).pack(anchor=tk.W)
            self.signal_vars[signal_name] = var

        # Comparison options
        options_frame = ttk.LabelFrame(self, text="Comparison Options", padding=10)
        options_frame.pack(fill=tk.X, padx=10, pady=5)

        self.comparison_type = tk.StringVar(value="overlay")
        ttk.Radiobutton(options_frame, text="Overlay plots",
                       variable=self.comparison_type,
                       value="overlay").pack(anchor=tk.W)
        ttk.Radiobutton(options_frame, text="Separate subplots",
                       variable=self.comparison_type,
                       value="separate").pack(anchor=tk.W)
        ttk.Radiobutton(options_frame, text="Correlation analysis",
                       variable=self.comparison_type,
                       value="correlation").pack(anchor=tk.W)

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="Compare",
                  command=self._compare).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.RIGHT)

    def _compare(self):
        """Compare selected signals."""
        selected = [name for name, var in self.signal_vars.items() if var.get()]

        if len(selected) < 2:
            messagebox.showerror("Error", "Please select at least 2 signals")
            return

        try:
            import matplotlib.pyplot as plt

            comp_type = self.comparison_type.get()

            if comp_type == "overlay":
                fig, ax = plt.subplots(figsize=(12, 6))

                for signal_name in selected:
                    signal_data = self.main_window.signal_manager.get_signal(signal_name)
                    signal = signal_data['signal']
                    sampling_rate = signal_data.get('sampling_rate', 1000)
                    ts = np.arange(len(signal)) / sampling_rate

                    ax.plot(ts, signal, label=signal_name, alpha=0.7)

                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.set_title('Signal Comparison')
                ax.legend()
                ax.grid(True, alpha=0.3)

            elif comp_type == "separate":
                n_signals = len(selected)
                fig, axes = plt.subplots(n_signals, 1, figsize=(12, 4 * n_signals),
                                        sharex=True)

                if n_signals == 1:
                    axes = [axes]

                for i, signal_name in enumerate(selected):
                    signal_data = self.main_window.signal_manager.get_signal(signal_name)
                    signal = signal_data['signal']
                    sampling_rate = signal_data.get('sampling_rate', 1000)
                    ts = np.arange(len(signal)) / sampling_rate

                    axes[i].plot(ts, signal)
                    axes[i].set_ylabel('Amplitude')
                    axes[i].set_title(signal_name)
                    axes[i].grid(True, alpha=0.3)

                axes[-1].set_xlabel('Time (s)')

            elif comp_type == "correlation":
                from biosppy.signals import tools

                # Compute pairwise correlations
                corr_matrix = np.zeros((len(selected), len(selected)))

                for i, name1 in enumerate(selected):
                    for j, name2 in enumerate(selected):
                        signal1 = self.main_window.signal_manager.get_signal(name1)['signal']
                        signal2 = self.main_window.signal_manager.get_signal(name2)['signal']

                        # Truncate to same length
                        min_len = min(len(signal1), len(signal2))
                        corr = np.corrcoef(signal1[:min_len], signal2[:min_len])[0, 1]
                        corr_matrix[i, j] = corr

                # Plot correlation matrix
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                ax.set_xticks(range(len(selected)))
                ax.set_yticks(range(len(selected)))
                ax.set_xticklabels(selected, rotation=45, ha='right')
                ax.set_yticklabels(selected)
                plt.colorbar(im, ax=ax, label='Correlation')
                ax.set_title('Signal Correlation Matrix')

            plt.tight_layout()
            plt.show()

            self.destroy()

        except Exception as e:
            messagebox.showerror("Error", f"Comparison failed:\n{str(e)}")
