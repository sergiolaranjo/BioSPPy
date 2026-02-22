"""
Dialogs
=======

Various dialog windows for configuration and user input.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os
import json


class ImportDialog(tk.Toplevel):
    """Dialog for importing signals."""

    def __init__(self, parent, main_window, file_format='auto'):
        super().__init__(parent)
        self.main_window = main_window
        self.file_format = file_format
        self.title("Import Signal")
        self.geometry("500x400")
        self._create_widgets()
        self._select_file()

    def _create_widgets(self):
        file_frame = ttk.LabelFrame(self, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(file_frame, text="File:").grid(row=0, column=0, sticky=tk.W)
        self.file_entry = ttk.Entry(file_frame, width=40)
        self.file_entry.grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse...",
                  command=self._select_file).grid(row=0, column=2)

        props_frame = ttk.LabelFrame(self, text="Signal Properties", padding=10)
        props_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        ttk.Label(props_frame, text="Signal Name:").grid(row=0, column=0,
                                                         sticky=tk.W, pady=2)
        self.name_entry = ttk.Entry(props_frame, width=30)
        self.name_entry.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(props_frame, text="Signal Type:").grid(row=1, column=0,
                                                         sticky=tk.W, pady=2)
        self.type_combo = ttk.Combobox(props_frame, width=28,
                                       values=['ECG', 'EDA', 'EMG', 'EEG',
                                             'PPG', 'Respiration', 'BVP',
                                             'ABP', 'PCG', 'Accelerometer', 'Other'])
        self.type_combo.current(0)
        self.type_combo.grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(props_frame, text="Sampling Rate (Hz):").grid(row=2, column=0,
                                                                sticky=tk.W, pady=2)
        self.sampling_rate_entry = ttk.Entry(props_frame, width=30)
        self.sampling_rate_entry.insert(0, "1000")
        self.sampling_rate_entry.grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(props_frame, text="Units:").grid(row=3, column=0,
                                                   sticky=tk.W, pady=2)
        self.units_entry = ttk.Entry(props_frame, width=30)
        self.units_entry.insert(0, "mV")
        self.units_entry.grid(row=3, column=1, padx=5, pady=2)

        if self.file_format in ['txt', 'csv', 'auto']:
            ttk.Label(props_frame, text="Column/Channel:").grid(row=4, column=0,
                                                                sticky=tk.W, pady=2)
            self.column_entry = ttk.Entry(props_frame, width=30)
            self.column_entry.insert(0, "0")
            self.column_entry.grid(row=4, column=1, padx=5, pady=2)

        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_frame, text="Import",
                  command=self._import).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.RIGHT)

    def _select_file(self):
        filetypes = [
            ("All supported", "*.txt *.edf *.h5 *.csv"),
            ("Text files", "*.txt"),
            ("EDF files", "*.edf"),
            ("HDF5 files", "*.h5"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="Select Signal File", filetypes=filetypes)
        if filename:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)
            base_name = os.path.splitext(os.path.basename(filename))[0]
            self.name_entry.delete(0, tk.END)
            self.name_entry.insert(0, base_name)

    def _import(self):
        filename = self.file_entry.get()
        if not filename:
            messagebox.showerror("Error", "Please select a file")
            return
        signal_name = self.name_entry.get()
        if not signal_name:
            messagebox.showerror("Error", "Please enter a signal name")
            return
        try:
            sampling_rate = float(self.sampling_rate_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid sampling rate")
            return

        signal_type = self.type_combo.get().lower()
        units = self.units_entry.get()

        try:
            from biosppy import storage
            ext = os.path.splitext(filename)[1].lower()

            if ext in ['.txt', '.csv']:
                signal, mdata = storage.load_txt(filename)
                if hasattr(self, 'column_entry'):
                    try:
                        col = int(self.column_entry.get())
                        if signal.ndim > 1:
                            signal = signal[:, col]
                    except Exception:
                        pass
            elif ext == '.edf':
                signals, mdata = storage.load_edf(filename)
                signal = signals[0] if isinstance(signals, list) else signals
            elif ext == '.h5':
                with storage.HDF(filename, mode='r') as f:
                    signal, mdata = f.get_signal(name=signal_name)
            else:
                messagebox.showerror("Error", f"Unsupported file format: {ext}")
                return

            self.main_window.signal_manager.add_signal(
                signal_name, signal,
                signal_type=signal_type, sampling_rate=sampling_rate,
                units=units, filename=filename)
            self.main_window.add_signal_to_list(signal_name)
            self.main_window.statusbar.set_message(
                f"Imported: {signal_name} ({len(signal)} samples)")
            self.destroy()
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import signal:\n{str(e)}")


class ProcessDialog(tk.Toplevel):
    """Dialog for signal processing - supports all BioSPPy signal types."""

    # Map of all supported signal types and their modules
    SIGNAL_MODULES = {
        'ecg': ('biosppy.signals.ecg', 'ecg'),
        'eda': ('biosppy.signals.eda', 'eda'),
        'emg': ('biosppy.signals.emg', 'emg'),
        'eeg': ('biosppy.signals.eeg', 'eeg'),
        'ppg': ('biosppy.signals.ppg', 'ppg'),
        'resp': ('biosppy.signals.resp', 'resp'),
        'bvp': ('biosppy.signals.bvp', 'bvp'),
        'abp': ('biosppy.signals.abp', 'abp'),
        'acc': ('biosppy.signals.acc', 'acc'),
        'pcg': ('biosppy.signals.pcg', 'pcg'),
    }

    def __init__(self, parent, main_window, signal_type):
        super().__init__(parent)
        self.main_window = main_window
        self.signal_type = signal_type
        self.title(f"Process Signal - {signal_type.upper()}")
        self.geometry("500x400")
        self._create_widgets()

    def _create_widgets(self):
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "No signal selected")
            self.destroy()
            return

        idx = selection[0]
        self.signal_name = self.main_window.signal_listbox.get(idx)

        params_frame = ttk.LabelFrame(self, text="Processing Parameters", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        row = 0
        self.show_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Show plot",
                       variable=self.show_var).grid(row=row, column=0,
                                                   columnspan=2, sticky=tk.W, pady=2)
        row += 1
        self.interactive_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="Interactive mode",
                       variable=self.interactive_var).grid(row=row, column=0,
                                                          columnspan=2, sticky=tk.W, pady=2)
        row += 1

        # ECG-specific: R-peak detector selection
        if self.signal_type == 'ecg':
            ttk.Label(params_frame, text="R-peak detector:").grid(row=row, column=0,
                                                                  sticky=tk.W, pady=2)
            self.detector_combo = ttk.Combobox(params_frame, width=20,
                                              values=['hamilton', 'christov',
                                                     'engzee', 'gamboa'])
            self.detector_combo.current(0)
            self.detector_combo.grid(row=row, column=1, padx=5, pady=2)
            row += 1

        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_frame, text="Process",
                  command=self._process).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.RIGHT)

    def _process(self):
        signal_data = self.main_window.signal_manager.get_signal(self.signal_name)
        if signal_data is None:
            return

        if self.signal_type not in self.SIGNAL_MODULES:
            messagebox.showerror("Error", f"Unknown signal type: {self.signal_type}")
            return

        try:
            import importlib
            mod_path, func_name = self.SIGNAL_MODULES[self.signal_type]
            sig_module = importlib.import_module(mod_path)

            kwargs = {}
            if self.signal_type == 'ecg' and hasattr(self, 'detector_combo'):
                kwargs['segmenter'] = self.detector_combo.get()

            self.main_window.statusbar.show_progress()
            self.main_window.statusbar.set_message(f"Processing {self.signal_type.upper()}...")

            result = getattr(sig_module, func_name)(
                signal=signal_data['signal'],
                sampling_rate=signal_data['sampling_rate'],
                show=False,
                **kwargs
            )

            # Build results dict from all available keys
            results_dict = {}
            for key in result.keys():
                results_dict[key] = result[key]

            self.main_window.signal_manager.add_processing_results(
                self.signal_name, results_dict)
            signal_data['results'] = results_dict
            self.main_window.plot_signal(self.signal_name)
            self.main_window.statusbar.hide_progress()
            self.main_window.statusbar.set_message(
                f"Processed {self.signal_name} as {self.signal_type.upper()}")
            self.destroy()

        except Exception as e:
            self.main_window.statusbar.hide_progress()
            messagebox.showerror("Processing Error",
                               f"Failed to process signal:\n{str(e)}")


class FilterDialog(tk.Toplevel):
    """Dialog for applying filters using biosppy.signals.tools."""

    def __init__(self, parent, main_window, filter_type):
        super().__init__(parent)
        self.main_window = main_window
        self.filter_type = filter_type
        self.title(f"Apply {filter_type.title()} Filter")
        self.geometry("400x350")
        self._create_widgets()

    def _create_widgets(self):
        params_frame = ttk.LabelFrame(self, text="Filter Parameters", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        row = 0
        if self.filter_type in ['bandpass', 'lowpass', 'highpass']:
            if self.filter_type in ['bandpass', 'highpass']:
                ttk.Label(params_frame, text="Low Cutoff (Hz):").grid(
                    row=row, column=0, sticky=tk.W, pady=2)
                self.low_cutoff_entry = ttk.Entry(params_frame, width=20)
                self.low_cutoff_entry.insert(0, "0.67")
                self.low_cutoff_entry.grid(row=row, column=1, padx=5, pady=2)
                row += 1

            if self.filter_type in ['bandpass', 'lowpass']:
                ttk.Label(params_frame, text="High Cutoff (Hz):").grid(
                    row=row, column=0, sticky=tk.W, pady=2)
                self.high_cutoff_entry = ttk.Entry(params_frame, width=20)
                self.high_cutoff_entry.insert(0, "45")
                self.high_cutoff_entry.grid(row=row, column=1, padx=5, pady=2)
                row += 1

            ttk.Label(params_frame, text="Order:").grid(
                row=row, column=0, sticky=tk.W, pady=2)
            self.order_entry = ttk.Entry(params_frame, width=20)
            self.order_entry.insert(0, "4")
            self.order_entry.grid(row=row, column=1, padx=5, pady=2)
            row += 1

        elif self.filter_type == 'notch':
            ttk.Label(params_frame, text="Frequency (Hz):").grid(
                row=row, column=0, sticky=tk.W, pady=2)
            self.freq_entry = ttk.Entry(params_frame, width=20)
            self.freq_entry.insert(0, "50")
            self.freq_entry.grid(row=row, column=1, padx=5, pady=2)
            row += 1

        elif self.filter_type == 'smooth':
            ttk.Label(params_frame, text="Window Size:").grid(
                row=row, column=0, sticky=tk.W, pady=2)
            self.window_entry = ttk.Entry(params_frame, width=20)
            self.window_entry.insert(0, "5")
            self.window_entry.grid(row=row, column=1, padx=5, pady=2)
            row += 1

        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_frame, text="Apply",
                  command=self._apply).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.RIGHT)

    def _apply(self):
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "No signal selected")
            return

        idx = selection[0]
        signal_name = self.main_window.signal_listbox.get(idx)
        signal_data = self.main_window.signal_manager.get_signal(signal_name)
        if signal_data is None:
            return

        try:
            from biosppy.signals import tools
            signal = signal_data['signal']
            sampling_rate = signal_data.get('sampling_rate', 1000)

            if self.filter_type == 'smooth':
                window_size = int(self.window_entry.get())
                filtered = np.convolve(signal,
                                       np.ones(window_size) / window_size,
                                       mode='same')
                desc = f"Smoothed (window={window_size})"
            elif self.filter_type == 'notch':
                freq = float(self.freq_entry.get())
                filtered, _, _ = tools.filter_signal(
                    signal=signal, ftype='butter', band='bandstop',
                    order=4, frequency=[freq - 1, freq + 1],
                    sampling_rate=sampling_rate)
                desc = f"Notch filter ({freq} Hz)"
            else:
                order = int(self.order_entry.get())
                if self.filter_type == 'bandpass':
                    low = float(self.low_cutoff_entry.get())
                    high = float(self.high_cutoff_entry.get())
                    filtered, _, _ = tools.filter_signal(
                        signal=signal, ftype='butter', band='bandpass',
                        order=order, frequency=[low, high],
                        sampling_rate=sampling_rate)
                    desc = f"Bandpass {low}-{high} Hz (order {order})"
                elif self.filter_type == 'lowpass':
                    high = float(self.high_cutoff_entry.get())
                    filtered, _, _ = tools.filter_signal(
                        signal=signal, ftype='butter', band='lowpass',
                        order=order, frequency=[high],
                        sampling_rate=sampling_rate)
                    desc = f"Lowpass {high} Hz (order {order})"
                elif self.filter_type == 'highpass':
                    low = float(self.low_cutoff_entry.get())
                    filtered, _, _ = tools.filter_signal(
                        signal=signal, ftype='butter', band='highpass',
                        order=order, frequency=[low],
                        sampling_rate=sampling_rate)
                    desc = f"Highpass {low} Hz (order {order})"

            self.main_window.signal_manager.update_signal(
                signal_name, filtered, desc)
            self.main_window.refresh_plot()
            self.main_window.statusbar.set_message(f"Applied: {desc}")
            self.destroy()

        except Exception as e:
            messagebox.showerror("Error", f"Filter failed:\n{str(e)}")


class AnalyzeDialog(tk.Toplevel):
    """Dialog for signal analysis - computes basic statistics and spectral info."""

    def __init__(self, parent, main_window, selection_only=False):
        super().__init__(parent)
        self.main_window = main_window
        self.selection_only = selection_only
        self.title("Analyze Signal" + (" (Selection)" if selection_only else ""))
        self.geometry("500x500")
        self._create_widgets()

    def _create_widgets(self):
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "No signal selected")
            self.destroy()
            return

        idx = selection[0]
        signal_name = self.main_window.signal_listbox.get(idx)
        signal_data = self.main_window.signal_manager.get_signal(signal_name)
        if signal_data is None:
            self.destroy()
            return

        signal = signal_data['signal']
        sr = signal_data.get('sampling_rate', 1000)

        if self.selection_only and 'selection' in signal_data:
            signal = signal_data['selection']['segment']
            title_suffix = " (selection)"
        else:
            title_suffix = ""

        # Compute statistics
        stats = {
            'Samples': len(signal),
            'Duration (s)': f"{len(signal) / sr:.3f}",
            'Sampling Rate (Hz)': sr,
            'Mean': f"{np.mean(signal):.6f}",
            'Std Dev': f"{np.std(signal):.6f}",
            'Variance': f"{np.var(signal):.6f}",
            'Min': f"{np.min(signal):.6f}",
            'Max': f"{np.max(signal):.6f}",
            'Range': f"{np.ptp(signal):.6f}",
            'RMS': f"{np.sqrt(np.mean(signal**2)):.6f}",
            'Median': f"{np.median(signal):.6f}",
            'Skewness': f"{_skewness(signal):.6f}",
            'Kurtosis': f"{_kurtosis(signal):.6f}",
            'Zero Crossings': int(np.sum(np.diff(np.sign(signal - np.mean(signal))) != 0)),
        }

        ttk.Label(self, text=f"Signal Analysis: {signal_name}{title_suffix}",
                 font=("Arial", 11, "bold")).pack(pady=5)

        # Results in a treeview
        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        tree = ttk.Treeview(tree_frame, columns=('Value',), show='tree headings')
        tree.heading('#0', text='Metric')
        tree.heading('Value', text='Value')
        tree.column('#0', width=200)
        tree.column('Value', width=250)

        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        for name, value in stats.items():
            tree.insert('', tk.END, text=name, values=(value,))

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_frame, text="Close",
                  command=self.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Copy to Clipboard",
                  command=lambda: self._copy_stats(stats)).pack(side=tk.RIGHT, padx=5)

    def _copy_stats(self, stats):
        text = "\n".join(f"{k}: {v}" for k, v in stats.items())
        self.clipboard_clear()
        self.clipboard_append(text)
        self.main_window.statusbar.set_message("Statistics copied to clipboard")


class FeatureExtractionDialog(tk.Toplevel):
    """Dialog for feature extraction using biosppy.features."""

    def __init__(self, parent, main_window, feature_type):
        super().__init__(parent)
        self.main_window = main_window
        self.feature_type = feature_type
        self.title(f"Extract {feature_type.title()} Features")
        self.geometry("550x450")
        self._create_widgets()

    def _create_widgets(self):
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "No signal selected")
            self.destroy()
            return

        idx = selection[0]
        self.signal_name = self.main_window.signal_listbox.get(idx)

        ttk.Label(self, text=f"{self.feature_type.title()} Feature Extraction",
                 font=("Arial", 11, "bold")).pack(pady=5)

        ttk.Button(self, text="Extract Features",
                  command=self._extract).pack(pady=10)

        # Results area
        self.results_text = tk.Text(self, height=20, width=60, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=5)

    def _extract(self):
        signal_data = self.main_window.signal_manager.get_signal(self.signal_name)
        if signal_data is None:
            return

        try:
            signal = signal_data['signal']
            sr = signal_data.get('sampling_rate', 1000)
            results = {}

            if self.feature_type == 'time':
                from biosppy.features import time as time_feats
                results = time_feats.time(signal, sr)
            elif self.feature_type == 'frequency':
                from biosppy.features import frequency as freq_feats
                results = freq_feats.frequency(signal, sr)
            elif self.feature_type == 'time_freq':
                from biosppy.features import time_freq as tf_feats
                results = tf_feats.time_freq(signal, sr)
            elif self.feature_type in ('nonlinear', 'phase_space'):
                from biosppy.features import phase_space as ps_feats
                results = ps_feats.phase_space(signal, sr)
            elif self.feature_type == 'cepstral':
                from biosppy.features import cepstral as cep_feats
                results = cep_feats.cepstral(signal, sr)

            # Display results
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            if hasattr(results, 'keys'):
                for key in results.keys():
                    val = results[key]
                    if isinstance(val, np.ndarray):
                        val = val.tolist()
                    self.results_text.insert(tk.END, f"{key}: {val}\n")
            else:
                self.results_text.insert(tk.END, str(results))
            self.results_text.config(state=tk.DISABLED)

            self.main_window.statusbar.set_message(
                f"Extracted {self.feature_type} features from {self.signal_name}")

        except Exception as e:
            messagebox.showerror("Error", f"Feature extraction failed:\n{str(e)}")


class HRVDialog(tk.Toplevel):
    """Dialog for HRV analysis using biosppy.signals.hrv."""

    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.title("HRV Analysis")
        self.geometry("600x500")
        self._create_widgets()

    def _create_widgets(self):
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showerror("Error",
                "No signal selected. Please select an ECG signal first.")
            self.destroy()
            return

        idx = selection[0]
        self.signal_name = self.main_window.signal_listbox.get(idx)

        ttk.Label(self, text="HRV Analysis",
                 font=("Arial", 12, "bold")).pack(pady=5)
        ttk.Label(self, text=f"Signal: {self.signal_name}").pack(pady=2)

        info = ttk.Label(self, text="Requires ECG signal with R-peaks detected.\n"
                        "Run 'Process as ECG' first if needed.",
                        foreground="gray")
        info.pack(pady=5)

        ttk.Button(self, text="Run HRV Analysis",
                  command=self._analyze).pack(pady=10)

        self.results_text = tk.Text(self, height=20, width=70, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=5)

    def _analyze(self):
        signal_data = self.main_window.signal_manager.get_signal(self.signal_name)
        if signal_data is None:
            return

        try:
            from biosppy.signals import hrv as hrv_module

            results = signal_data.get('results', {})
            rpeaks = results.get('rpeaks')

            if rpeaks is None:
                # Try to detect R-peaks first
                from biosppy.signals import ecg
                ecg_result = ecg.ecg(
                    signal=signal_data['signal'],
                    sampling_rate=signal_data['sampling_rate'],
                    show=False)
                rpeaks = ecg_result['rpeaks']

            sr = signal_data['sampling_rate']
            rri = np.diff(rpeaks) / sr * 1000  # in ms

            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)

            # Time-domain
            self.results_text.insert(tk.END, "=== Time-Domain HRV ===\n")
            self.results_text.insert(tk.END, f"Mean RR (ms): {np.mean(rri):.2f}\n")
            self.results_text.insert(tk.END, f"SDNN (ms): {np.std(rri, ddof=1):.2f}\n")
            rmssd = np.sqrt(np.mean(np.diff(rri)**2))
            self.results_text.insert(tk.END, f"RMSSD (ms): {rmssd:.2f}\n")
            pnn50 = np.sum(np.abs(np.diff(rri)) > 50) / len(np.diff(rri)) * 100
            self.results_text.insert(tk.END, f"pNN50 (%): {pnn50:.2f}\n")
            pnn20 = np.sum(np.abs(np.diff(rri)) > 20) / len(np.diff(rri)) * 100
            self.results_text.insert(tk.END, f"pNN20 (%): {pnn20:.2f}\n")
            self.results_text.insert(tk.END, f"Mean HR (bpm): {60000/np.mean(rri):.2f}\n\n")

            # Frequency-domain (basic)
            self.results_text.insert(tk.END, "=== Frequency-Domain HRV ===\n")
            try:
                from scipy import interpolate, signal as scipy_signal
                # Interpolate RR intervals for spectral analysis
                rr_times = np.cumsum(rri) / 1000  # in seconds
                rr_times = np.insert(rr_times, 0, 0)
                fs_resamp = 4.0  # 4 Hz resampling
                t_interp = np.arange(rr_times[0], rr_times[-1], 1.0/fs_resamp)
                f_interp = interpolate.interp1d(rr_times, np.insert(rri, 0, rri[0]),
                                                kind='cubic', fill_value='extrapolate')
                rri_interp = f_interp(t_interp)
                rri_interp = rri_interp - np.mean(rri_interp)

                freqs, psd = scipy_signal.welch(rri_interp, fs=fs_resamp,
                                                nperseg=min(256, len(rri_interp)))

                vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
                lf_mask = (freqs >= 0.04) & (freqs < 0.15)
                hf_mask = (freqs >= 0.15) & (freqs < 0.4)

                vlf_power = np.trapz(psd[vlf_mask], freqs[vlf_mask])
                lf_power = np.trapz(psd[lf_mask], freqs[lf_mask])
                hf_power = np.trapz(psd[hf_mask], freqs[hf_mask])
                total_power = vlf_power + lf_power + hf_power

                self.results_text.insert(tk.END, f"VLF Power (ms2): {vlf_power:.2f}\n")
                self.results_text.insert(tk.END, f"LF Power (ms2): {lf_power:.2f}\n")
                self.results_text.insert(tk.END, f"HF Power (ms2): {hf_power:.2f}\n")
                self.results_text.insert(tk.END, f"Total Power (ms2): {total_power:.2f}\n")
                if hf_power > 0:
                    self.results_text.insert(tk.END, f"LF/HF Ratio: {lf_power/hf_power:.3f}\n")
                if total_power > 0:
                    self.results_text.insert(tk.END,
                        f"LF (n.u.): {lf_power/(lf_power+hf_power)*100:.1f}%\n")
                    self.results_text.insert(tk.END,
                        f"HF (n.u.): {hf_power/(lf_power+hf_power)*100:.1f}%\n")
            except Exception as e:
                self.results_text.insert(tk.END, f"Spectral analysis error: {e}\n")

            # Non-linear
            self.results_text.insert(tk.END, "\n=== Non-linear HRV ===\n")
            sd1 = np.sqrt(0.5 * np.var(np.diff(rri)))
            sd2 = np.sqrt(2 * np.var(rri) - 0.5 * np.var(np.diff(rri)))
            self.results_text.insert(tk.END, f"SD1 (ms): {sd1:.2f}\n")
            self.results_text.insert(tk.END, f"SD2 (ms): {sd2:.2f}\n")
            if sd2 > 0:
                self.results_text.insert(tk.END, f"SD1/SD2: {sd1/sd2:.3f}\n")
            self.results_text.insert(tk.END,
                f"Ellipse Area (ms2): {np.pi * sd1 * sd2:.2f}\n")

            self.results_text.config(state=tk.DISABLED)
            self.main_window.statusbar.set_message("HRV analysis complete")

        except Exception as e:
            messagebox.showerror("Error", f"HRV analysis failed:\n{str(e)}")


class StatisticsDialog(tk.Toplevel):
    """Dialog for signal statistics."""

    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.title("Signal Statistics")
        self.geometry("500x400")
        self._create_widgets()

    def _create_widgets(self):
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "No signal selected")
            self.destroy()
            return

        idx = selection[0]
        signal_name = self.main_window.signal_listbox.get(idx)
        signal_data = self.main_window.signal_manager.get_signal(signal_name)
        if signal_data is None:
            self.destroy()
            return

        signal = signal_data['signal']
        sr = signal_data.get('sampling_rate', 1000)

        ttk.Label(self, text=f"Statistics: {signal_name}",
                 font=("Arial", 11, "bold")).pack(pady=5)

        stats_text = tk.Text(self, height=20, width=55, state=tk.DISABLED)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        stats_text.config(state=tk.NORMAL)
        stats_text.insert(tk.END, f"Signal: {signal_name}\n")
        stats_text.insert(tk.END, f"Type: {signal_data.get('type', 'Unknown')}\n")
        stats_text.insert(tk.END, f"Samples: {len(signal)}\n")
        stats_text.insert(tk.END, f"Duration: {len(signal)/sr:.3f} s\n")
        stats_text.insert(tk.END, f"Sampling Rate: {sr} Hz\n\n")
        stats_text.insert(tk.END, f"Mean: {np.mean(signal):.6f}\n")
        stats_text.insert(tk.END, f"Median: {np.median(signal):.6f}\n")
        stats_text.insert(tk.END, f"Std Dev: {np.std(signal):.6f}\n")
        stats_text.insert(tk.END, f"Variance: {np.var(signal):.6f}\n")
        stats_text.insert(tk.END, f"Min: {np.min(signal):.6f}\n")
        stats_text.insert(tk.END, f"Max: {np.max(signal):.6f}\n")
        stats_text.insert(tk.END, f"Range: {np.ptp(signal):.6f}\n")
        stats_text.insert(tk.END, f"RMS: {np.sqrt(np.mean(signal**2)):.6f}\n")
        stats_text.insert(tk.END, f"Skewness: {_skewness(signal):.6f}\n")
        stats_text.insert(tk.END, f"Kurtosis: {_kurtosis(signal):.6f}\n")
        zc = int(np.sum(np.diff(np.sign(signal - np.mean(signal))) != 0))
        stats_text.insert(tk.END, f"Zero Crossings: {zc}\n")

        # Percentiles
        stats_text.insert(tk.END, f"\nPercentiles:\n")
        for p in [5, 25, 50, 75, 95]:
            stats_text.insert(tk.END, f"  P{p}: {np.percentile(signal, p):.6f}\n")

        stats_text.config(state=tk.DISABLED)

        ttk.Button(self, text="Close", command=self.destroy).pack(pady=5)


class ResampleDialog(tk.Toplevel):
    """Dialog for resampling signals."""

    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.title("Resample Signal")
        self.geometry("400x250")
        self._create_widgets()

    def _create_widgets(self):
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "No signal selected")
            self.destroy()
            return

        idx = selection[0]
        self.signal_name = self.main_window.signal_listbox.get(idx)
        signal_data = self.main_window.signal_manager.get_signal(self.signal_name)
        current_sr = signal_data.get('sampling_rate', 1000) if signal_data else 1000

        ttk.Label(self, text="Resample Signal",
                 font=("Arial", 11, "bold")).pack(pady=5)
        ttk.Label(self, text=f"Current: {current_sr} Hz").pack(pady=2)

        params_frame = ttk.LabelFrame(self, text="Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(params_frame, text="New Sampling Rate (Hz):").grid(
            row=0, column=0, sticky=tk.W, pady=2)
        self.new_sr_entry = ttk.Entry(params_frame, width=20)
        self.new_sr_entry.insert(0, str(int(current_sr)))
        self.new_sr_entry.grid(row=0, column=1, padx=5, pady=2)

        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_frame, text="Resample",
                  command=self._resample).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.RIGHT)

    def _resample(self):
        signal_data = self.main_window.signal_manager.get_signal(self.signal_name)
        if signal_data is None:
            return

        try:
            from scipy import signal as scipy_signal
            new_sr = float(self.new_sr_entry.get())
            old_sr = signal_data['sampling_rate']
            sig = signal_data['signal']

            num_samples = int(len(sig) * new_sr / old_sr)
            resampled = scipy_signal.resample(sig, num_samples)

            signal_data['sampling_rate'] = new_sr
            self.main_window.signal_manager.update_signal(
                self.signal_name, resampled, f"Resampled {old_sr}->{new_sr} Hz")
            self.main_window.refresh_plot()
            self.main_window.statusbar.set_message(
                f"Resampled {self.signal_name}: {old_sr} -> {new_sr} Hz")
            self.destroy()

        except Exception as e:
            messagebox.showerror("Error", f"Resample failed:\n{str(e)}")


class PreferencesDialog(tk.Toplevel):
    """Dialog for application preferences."""

    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.title("Preferences")
        self.geometry("500x400")
        self._create_widgets()

    def _create_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Display settings
        display_frame = ttk.Frame(notebook, padding=10)
        notebook.add(display_frame, text="Display")

        ttk.Label(display_frame, text="Default Sampling Rate (Hz):").grid(
            row=0, column=0, sticky=tk.W, pady=5)
        self.default_sr = ttk.Entry(display_frame, width=20)
        self.default_sr.insert(0, "1000")
        self.default_sr.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(display_frame, text="Plot Line Width:").grid(
            row=1, column=0, sticky=tk.W, pady=5)
        self.line_width = ttk.Spinbox(display_frame, from_=0.1, to=5.0,
                                       increment=0.1, width=18)
        self.line_width.set("0.5")
        self.line_width.grid(row=1, column=1, padx=5, pady=5)

        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Show grid by default",
                       variable=self.show_grid_var).grid(
            row=2, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Processing settings
        proc_frame = ttk.Frame(notebook, padding=10)
        notebook.add(proc_frame, text="Processing")

        ttk.Label(proc_frame, text="Default filter order:").grid(
            row=0, column=0, sticky=tk.W, pady=5)
        self.default_order = ttk.Spinbox(proc_frame, from_=1, to=20, width=18)
        self.default_order.set("4")
        self.default_order.grid(row=0, column=1, padx=5, pady=5)

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_frame, text="OK",
                  command=self.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.RIGHT)


class BatchProcessingDialog(tk.Toplevel):
    """Dialog for batch processing multiple files."""

    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.title("Batch Processing")
        self.geometry("600x500")
        self.files = []
        self._create_widgets()

    def _create_widgets(self):
        ttk.Label(self, text="Batch Processing",
                 font=("Arial", 12, "bold")).pack(pady=5)

        # File list
        file_frame = ttk.LabelFrame(self, text="Input Files", padding=10)
        file_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Add Files...",
                  command=self._add_files).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Clear",
                  command=self._clear_files).pack(side=tk.LEFT, padx=2)

        self.file_listbox = tk.Listbox(file_frame, height=8)
        self.file_listbox.pack(fill=tk.BOTH, expand=True, pady=5)

        # Processing options
        opts_frame = ttk.LabelFrame(self, text="Processing", padding=10)
        opts_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(opts_frame, text="Signal Type:").grid(row=0, column=0, sticky=tk.W)
        self.type_combo = ttk.Combobox(opts_frame, width=20,
                                       values=['ECG', 'EDA', 'EMG', 'EEG',
                                              'PPG', 'Respiration', 'PCG'])
        self.type_combo.current(0)
        self.type_combo.grid(row=0, column=1, padx=5)

        ttk.Label(opts_frame, text="Sampling Rate (Hz):").grid(
            row=1, column=0, sticky=tk.W)
        self.sr_entry = ttk.Entry(opts_frame, width=22)
        self.sr_entry.insert(0, "1000")
        self.sr_entry.grid(row=1, column=1, padx=5)

        # Output
        out_frame = ttk.LabelFrame(self, text="Output", padding=10)
        out_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(out_frame, text="Output Directory:").grid(row=0, column=0, sticky=tk.W)
        self.out_entry = ttk.Entry(out_frame, width=35)
        self.out_entry.grid(row=0, column=1, padx=5)
        ttk.Button(out_frame, text="Browse...",
                  command=self._select_output).grid(row=0, column=2)

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_frame, text="Run Batch",
                  command=self._run_batch).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.RIGHT)

    def _add_files(self):
        files = filedialog.askopenfilenames(
            title="Select Files",
            filetypes=[("All supported", "*.txt *.edf *.csv *.h5"),
                      ("All files", "*.*")])
        for f in files:
            self.files.append(f)
            self.file_listbox.insert(tk.END, os.path.basename(f))

    def _clear_files(self):
        self.files.clear()
        self.file_listbox.delete(0, tk.END)

    def _select_output(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.out_entry.delete(0, tk.END)
            self.out_entry.insert(0, directory)

    def _run_batch(self):
        if not self.files:
            messagebox.showerror("Error", "No files selected")
            return

        sig_type = self.type_combo.get().lower()
        try:
            sr = float(self.sr_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid sampling rate")
            return

        output_dir = self.out_entry.get()
        processed = 0
        errors = []

        self.main_window.statusbar.show_progress()

        for filepath in self.files:
            try:
                from biosppy import storage
                signal, _ = storage.load_txt(filepath)
                if signal.ndim > 1:
                    signal = signal[:, 0]

                import importlib
                mod = importlib.import_module(f'biosppy.signals.{sig_type}')
                result = getattr(mod, sig_type)(
                    signal=signal, sampling_rate=sr, show=False)

                if output_dir:
                    base = os.path.splitext(os.path.basename(filepath))[0]
                    out_path = os.path.join(output_dir, f"{base}_results.json")
                    out_data = {}
                    for key in result.keys():
                        val = result[key]
                        if isinstance(val, np.ndarray):
                            out_data[key] = val.tolist()
                        elif isinstance(val, (int, float, str)):
                            out_data[key] = val
                    with open(out_path, 'w') as f:
                        json.dump(out_data, f, indent=2)

                processed += 1
            except Exception as e:
                errors.append(f"{os.path.basename(filepath)}: {e}")

        self.main_window.statusbar.hide_progress()

        msg = f"Processed {processed}/{len(self.files)} files."
        if errors:
            msg += f"\n\nErrors ({len(errors)}):\n" + "\n".join(errors[:10])
        messagebox.showinfo("Batch Complete", msg)
        self.destroy()


class ExportDialog(tk.Toplevel):
    """Dialog for exporting data."""

    def __init__(self, parent, main_window, export_type, file_format):
        super().__init__(parent)
        self.main_window = main_window
        self.export_type = export_type
        self.file_format = file_format
        self.title(f"Export {export_type.title()}")
        self.geometry("400x250")
        self._create_widgets()

    def _create_widgets(self):
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "No signal selected")
            self.destroy()
            return

        idx = selection[0]
        self.signal_name = self.main_window.signal_listbox.get(idx)

        ttk.Label(self, text=f"Export {self.export_type.title()}",
                 font=("Arial", 11, "bold")).pack(pady=5)
        ttk.Label(self, text=f"Signal: {self.signal_name}").pack(pady=2)

        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(button_frame, text="Export",
                  command=self._export).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.RIGHT)

    def _export(self):
        signal_data = self.main_window.signal_manager.get_signal(self.signal_name)
        if signal_data is None:
            return

        if self.export_type == 'signal':
            ext_map = {'txt': '.txt', 'csv': '.csv', 'json': '.json'}
            ext = ext_map.get(self.file_format, '.txt')
            filename = filedialog.asksaveasfilename(
                title="Export Signal",
                defaultextension=ext,
                filetypes=[(f"{self.file_format.upper()} files", f"*{ext}"),
                          ("All files", "*.*")])
            if filename:
                try:
                    signal = signal_data['signal']
                    if ext in ['.txt', '.csv']:
                        np.savetxt(filename, signal, delimiter=',')
                    else:
                        data = {'signal': signal.tolist(),
                                'sampling_rate': signal_data.get('sampling_rate', 1000),
                                'type': signal_data.get('type', 'unknown')}
                        with open(filename, 'w') as f:
                            json.dump(data, f, indent=2)
                    self.main_window.statusbar.set_message(f"Exported: {filename}")
                    self.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Export failed:\n{str(e)}")

        elif self.export_type == 'results':
            filename = filedialog.asksaveasfilename(
                title="Export Results",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
            if filename:
                try:
                    results = signal_data.get('results', {})
                    out = {}
                    for key, val in results.items():
                        if isinstance(val, np.ndarray):
                            out[key] = val.tolist()
                        elif isinstance(val, (int, float, str, list)):
                            out[key] = val
                    with open(filename, 'w') as f:
                        json.dump(out, f, indent=2)
                    self.main_window.statusbar.set_message(f"Results exported: {filename}")
                    self.destroy()
                except Exception as e:
                    messagebox.showerror("Error", f"Export failed:\n{str(e)}")


# -- Helper functions --

def _skewness(x):
    """Compute skewness."""
    n = len(x)
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return np.mean(((x - m) / s) ** 3)


def _kurtosis(x):
    """Compute excess kurtosis."""
    n = len(x)
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return np.mean(((x - m) / s) ** 4) - 3
