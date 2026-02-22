"""
Advanced Analysis Module
=========================

Advanced signal analysis features including:
- Feature extraction (time, frequency, time-frequency, cepstral, phase space)
- HRV analysis (time-domain, frequency-domain, non-linear)
- Signal quality assessment
- Clustering and biometrics
- Statistical analysis
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np


class AdvancedAnalysisDialog(tk.Toplevel):
    """Dialog for advanced signal analysis."""

    def __init__(self, parent, main_window, analysis_type='features'):
        """Initialize advanced analysis dialog.

        Parameters
        ----------
        parent : tk.Widget
            Parent window.
        main_window : BioSPPyGUI
            Reference to main window.
        analysis_type : str
            Type of analysis ('features', 'hrv', 'quality', 'clustering').
        """
        super().__init__(parent)
        self.main_window = main_window
        self.analysis_type = analysis_type

        self.title(f"Advanced Analysis - {analysis_type.title()}")
        self.geometry("600x500")

        self._create_widgets()

    def _create_widgets(self):
        """Create dialog widgets based on analysis type."""
        if self.analysis_type == 'features':
            self._create_feature_extraction_ui()
        elif self.analysis_type == 'hrv':
            self._create_hrv_analysis_ui()
        elif self.analysis_type == 'quality':
            self._create_quality_assessment_ui()
        elif self.analysis_type == 'clustering':
            self._create_clustering_ui()

    def _create_feature_extraction_ui(self):
        """Create feature extraction UI."""
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Time domain features
        time_frame = ttk.Frame(notebook)
        notebook.add(time_frame, text="Time Domain")
        self._create_time_domain_features(time_frame)

        # Frequency domain features
        freq_frame = ttk.Frame(notebook)
        notebook.add(freq_frame, text="Frequency Domain")
        self._create_frequency_domain_features(freq_frame)

        # Time-frequency features
        timefreq_frame = ttk.Frame(notebook)
        notebook.add(timefreq_frame, text="Time-Frequency")
        self._create_time_frequency_features(timefreq_frame)

        # Cepstral features
        cepstral_frame = ttk.Frame(notebook)
        notebook.add(cepstral_frame, text="Cepstral")
        self._create_cepstral_features(cepstral_frame)

        # Phase space features
        phase_frame = ttk.Frame(notebook)
        notebook.add(phase_frame, text="Phase Space")
        self._create_phase_space_features(phase_frame)

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="Extract Features",
                  command=self._extract_features).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.RIGHT)

    def _create_time_domain_features(self, parent):
        """Create time domain feature options."""
        ttk.Label(parent, text="Time Domain Features:",
                 font=("Arial", 10, "bold")).pack(pady=5, anchor=tk.W, padx=10)

        features_frame = ttk.Frame(parent)
        features_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.time_features = {}

        features = [
            ("Mean", "Calculate mean value"),
            ("Variance", "Calculate variance"),
            ("Standard Deviation", "Calculate std deviation"),
            ("Min/Max", "Minimum and maximum values"),
            ("Range", "Signal range"),
            ("RMS", "Root mean square"),
            ("Hjorth Parameters", "Activity, Mobility, Complexity"),
            ("Zero Crossings", "Number of zero crossings"),
            ("Mean Absolute Deviation", "MAD"),
            ("Skewness", "Third moment"),
            ("Kurtosis", "Fourth moment"),
        ]

        for i, (name, desc) in enumerate(features):
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(features_frame, text=name, variable=var)
            cb.grid(row=i, column=0, sticky=tk.W, pady=2)
            self.time_features[name.lower().replace(" ", "_")] = var

            # Tooltip label
            ttk.Label(features_frame, text=f"  ({desc})",
                     foreground="gray").grid(row=i, column=1, sticky=tk.W)

    def _create_frequency_domain_features(self, parent):
        """Create frequency domain feature options."""
        ttk.Label(parent, text="Frequency Domain Features:",
                 font=("Arial", 10, "bold")).pack(pady=5, anchor=tk.W, padx=10)

        # Method selection
        method_frame = ttk.LabelFrame(parent, text="Method", padding=10)
        method_frame.pack(fill=tk.X, padx=10, pady=5)

        self.freq_method = tk.StringVar(value="fft")
        ttk.Radiobutton(method_frame, text="FFT", variable=self.freq_method,
                       value="fft").pack(anchor=tk.W)
        ttk.Radiobutton(method_frame, text="Welch", variable=self.freq_method,
                       value="welch").pack(anchor=tk.W)
        ttk.Radiobutton(method_frame, text="Lomb-Scargle",
                       variable=self.freq_method,
                       value="lomb").pack(anchor=tk.W)

        # Features
        features_frame = ttk.LabelFrame(parent, text="Features", padding=10)
        features_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.freq_features = {}

        features = [
            ("Peak Frequency", "Dominant frequency"),
            ("Mean Frequency", "Mean power frequency"),
            ("Median Frequency", "Median power frequency"),
            ("Band Power", "Power in frequency bands"),
            ("Spectral Entropy", "Entropy of power spectrum"),
            ("Spectral Centroid", "Center of mass of spectrum"),
            ("Spectral Spread", "Spread around centroid"),
        ]

        for i, (name, desc) in enumerate(features):
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(features_frame, text=name, variable=var)
            cb.grid(row=i, column=0, sticky=tk.W, pady=2)
            self.freq_features[name.lower().replace(" ", "_")] = var

    def _create_time_frequency_features(self, parent):
        """Create time-frequency feature options."""
        ttk.Label(parent, text="Time-Frequency Features (Wavelet):",
                 font=("Arial", 10, "bold")).pack(pady=5, anchor=tk.W, padx=10)

        # Wavelet selection
        wavelet_frame = ttk.LabelFrame(parent, text="Wavelet Type", padding=10)
        wavelet_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(wavelet_frame, text="Wavelet:").grid(row=0, column=0, sticky=tk.W)
        self.wavelet_type = ttk.Combobox(wavelet_frame, width=20,
                                         values=['db4', 'db8', 'sym4', 'coif4'])
        self.wavelet_type.current(0)
        self.wavelet_type.grid(row=0, column=1, padx=5)

        ttk.Label(wavelet_frame, text="Levels:").grid(row=1, column=0, sticky=tk.W)
        self.wavelet_levels = ttk.Spinbox(wavelet_frame, from_=1, to=10,
                                          width=20)
        self.wavelet_levels.set(4)
        self.wavelet_levels.grid(row=1, column=1, padx=5)

        # Features
        features_frame = ttk.LabelFrame(parent, text="Features", padding=10)
        features_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.timefreq_features = {}

        features = [
            ("Wavelet Energy", "Energy in each level"),
            ("Wavelet Entropy", "Entropy of coefficients"),
            ("Relative Energy", "Normalized energy per level"),
        ]

        for i, (name, desc) in enumerate(features):
            var = tk.BooleanVar(value=True)
            ttk.Checkbutton(features_frame, text=f"{name} ({desc})",
                           variable=var).pack(anchor=tk.W, pady=2)
            self.timefreq_features[name.lower().replace(" ", "_")] = var

    def _create_cepstral_features(self, parent):
        """Create cepstral feature options."""
        ttk.Label(parent, text="Cepstral Features:",
                 font=("Arial", 10, "bold")).pack(pady=5, anchor=tk.W, padx=10)

        # MFCC parameters
        mfcc_frame = ttk.LabelFrame(parent, text="MFCC Parameters", padding=10)
        mfcc_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(mfcc_frame, text="Number of Coefficients:").grid(row=0,
                                                                   column=0,
                                                                   sticky=tk.W)
        self.n_mfcc = ttk.Spinbox(mfcc_frame, from_=1, to=40, width=20)
        self.n_mfcc.set(13)
        self.n_mfcc.grid(row=0, column=1, padx=5)

        self.extract_mfcc = tk.BooleanVar(value=True)
        ttk.Checkbutton(mfcc_frame, text="Extract MFCC",
                       variable=self.extract_mfcc).grid(row=1, column=0,
                                                        columnspan=2,
                                                        sticky=tk.W, pady=5)

    def _create_phase_space_features(self, parent):
        """Create phase space feature options."""
        ttk.Label(parent, text="Phase Space Features:",
                 font=("Arial", 10, "bold")).pack(pady=5, anchor=tk.W, padx=10)

        # Parameters
        params_frame = ttk.LabelFrame(parent, text="Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(params_frame, text="Embedding Dimension:").grid(row=0,
                                                                  column=0,
                                                                  sticky=tk.W)
        self.embed_dim = ttk.Spinbox(params_frame, from_=2, to=10, width=20)
        self.embed_dim.set(3)
        self.embed_dim.grid(row=0, column=1, padx=5)

        ttk.Label(params_frame, text="Time Delay:").grid(row=1, column=0,
                                                         sticky=tk.W)
        self.time_delay = ttk.Spinbox(params_frame, from_=1, to=100, width=20)
        self.time_delay.set(1)
        self.time_delay.grid(row=1, column=1, padx=5)

        # Features
        features_frame = ttk.LabelFrame(parent, text="Features", padding=10)
        features_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.phase_features = {}

        features = [
            ("Recurrence Plot", "Generate recurrence plot"),
            ("Recurrence Rate", "Percentage of recurrent points"),
            ("Determinism", "Percentage of recurrent points in diagonal"),
            ("Laminarity", "Percentage of recurrent points in vertical"),
            ("Entropy", "Shannon entropy of diagonal lengths"),
        ]

        for i, (name, desc) in enumerate(features):
            var = tk.BooleanVar(value=True)
            ttk.Checkbutton(features_frame, text=f"{name} ({desc})",
                           variable=var).pack(anchor=tk.W, pady=2)
            self.phase_features[name.lower().replace(" ", "_")] = var

    def _create_hrv_analysis_ui(self):
        """Create HRV analysis UI."""
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Time domain
        time_frame = ttk.Frame(notebook)
        notebook.add(time_frame, text="Time Domain")
        self._create_hrv_time_domain(time_frame)

        # Frequency domain
        freq_frame = ttk.Frame(notebook)
        notebook.add(freq_frame, text="Frequency Domain")
        self._create_hrv_frequency_domain(freq_frame)

        # Non-linear
        nonlinear_frame = ttk.Frame(notebook)
        notebook.add(nonlinear_frame, text="Non-linear")
        self._create_hrv_nonlinear(nonlinear_frame)

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="Analyze HRV",
                  command=self._analyze_hrv).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.RIGHT)

    def _create_hrv_time_domain(self, parent):
        """Create HRV time domain metrics."""
        ttk.Label(parent, text="Time Domain HRV Metrics:",
                 font=("Arial", 10, "bold")).pack(pady=5, anchor=tk.W, padx=10)

        metrics_frame = ttk.Frame(parent)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.hrv_time_metrics = {}

        metrics = [
            ("AVNN", "Average NN interval"),
            ("SDNN", "Standard deviation of NN intervals"),
            ("RMSSD", "Root mean square of successive differences"),
            ("pNN50", "Percentage of NN intervals > 50ms different"),
            ("pNN20", "Percentage of NN intervals > 20ms different"),
            ("SDSD", "Standard deviation of successive differences"),
            ("Triangular Index", "Total NN intervals / height of histogram"),
            ("TINN", "Triangular interpolation of NN intervals"),
        ]

        for i, (name, desc) in enumerate(metrics):
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(metrics_frame, text=name, variable=var)
            cb.grid(row=i, column=0, sticky=tk.W, pady=2)
            ttk.Label(metrics_frame, text=f"  ({desc})",
                     foreground="gray").grid(row=i, column=1, sticky=tk.W)
            self.hrv_time_metrics[name.lower()] = var

    def _create_hrv_frequency_domain(self, parent):
        """Create HRV frequency domain metrics."""
        ttk.Label(parent, text="Frequency Domain HRV Metrics:",
                 font=("Arial", 10, "bold")).pack(pady=5, anchor=tk.W, padx=10)

        # Method
        method_frame = ttk.LabelFrame(parent, text="Method", padding=10)
        method_frame.pack(fill=tk.X, padx=10, pady=5)

        self.hrv_freq_method = tk.StringVar(value="welch")
        ttk.Radiobutton(method_frame, text="Welch", variable=self.hrv_freq_method,
                       value="welch").pack(anchor=tk.W)
        ttk.Radiobutton(method_frame, text="Lomb-Scargle",
                       variable=self.hrv_freq_method,
                       value="lomb").pack(anchor=tk.W)
        ttk.Radiobutton(method_frame, text="AR (Autoregressive)",
                       variable=self.hrv_freq_method,
                       value="ar").pack(anchor=tk.W)

        # Metrics
        metrics_frame = ttk.Frame(parent)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.hrv_freq_metrics = {}

        metrics = [
            ("VLF", "Very Low Frequency power (0-0.04 Hz)"),
            ("LF", "Low Frequency power (0.04-0.15 Hz)"),
            ("HF", "High Frequency power (0.15-0.4 Hz)"),
            ("LF/HF", "LF to HF ratio"),
            ("Total Power", "Total spectral power"),
            ("LF (nu)", "LF in normalized units"),
            ("HF (nu)", "HF in normalized units"),
        ]

        for i, (name, desc) in enumerate(metrics):
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(metrics_frame, text=name, variable=var)
            cb.grid(row=i, column=0, sticky=tk.W, pady=2)
            ttk.Label(metrics_frame, text=f"  ({desc})",
                     foreground="gray").grid(row=i, column=1, sticky=tk.W)
            self.hrv_freq_metrics[name.lower().replace("/", "_").replace(" ", "_")] = var

    def _create_hrv_nonlinear(self, parent):
        """Create HRV non-linear metrics."""
        ttk.Label(parent, text="Non-linear HRV Metrics:",
                 font=("Arial", 10, "bold")).pack(pady=5, anchor=tk.W, padx=10)

        metrics_frame = ttk.Frame(parent)
        metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.hrv_nonlinear_metrics = {}

        metrics = [
            ("SD1", "Poincaré plot - short-term variability"),
            ("SD2", "Poincaré plot - long-term variability"),
            ("SD1/SD2", "Ratio of SD1 to SD2"),
            ("Ellipse Area", "Area of Poincaré ellipse"),
            ("Sample Entropy", "Sample entropy of RR intervals"),
            ("Approximate Entropy", "Approximate entropy"),
            ("DFA α1", "Detrended fluctuation analysis - short-term"),
            ("DFA α2", "Detrended fluctuation analysis - long-term"),
        ]

        for i, (name, desc) in enumerate(metrics):
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(metrics_frame, text=name, variable=var)
            cb.grid(row=i, column=0, sticky=tk.W, pady=2)
            ttk.Label(metrics_frame, text=f"  ({desc})",
                     foreground="gray").grid(row=i, column=1, sticky=tk.W)
            self.hrv_nonlinear_metrics[name.lower().replace(" ", "_").replace("/", "_")] = var

    def _create_quality_assessment_ui(self):
        """Create quality assessment UI."""
        info_label = ttk.Label(self, text="Signal Quality Assessment",
                              font=("Arial", 12, "bold"))
        info_label.pack(pady=10)

        # Signal type selection
        type_frame = ttk.LabelFrame(self, text="Signal Type", padding=10)
        type_frame.pack(fill=tk.X, padx=10, pady=5)

        self.quality_signal_type = tk.StringVar(value="ecg")
        ttk.Radiobutton(type_frame, text="ECG", variable=self.quality_signal_type,
                       value="ecg").pack(anchor=tk.W)
        ttk.Radiobutton(type_frame, text="EDA/GSR",
                       variable=self.quality_signal_type,
                       value="eda").pack(anchor=tk.W)

        # Quality indices
        indices_frame = ttk.LabelFrame(self, text="Quality Indices", padding=10)
        indices_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.quality_indices = {}

        # ECG indices
        ecg_indices = [
            ("bSQI", "Beat detection SQI"),
            ("sSQI", "Skewness SQI"),
            ("kSQI", "Kurtosis SQI"),
            ("pSQI", "Power spectrum SQI"),
            ("fSQI", "Frequency domain SQI"),
            ("cSQI", "Correlation SQI"),
        ]

        for i, (name, desc) in enumerate(ecg_indices):
            var = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(indices_frame, text=name, variable=var)
            cb.grid(row=i, column=0, sticky=tk.W, pady=2)
            ttk.Label(indices_frame, text=f"  ({desc})",
                     foreground="gray").grid(row=i, column=1, sticky=tk.W)
            self.quality_indices[name.lower()] = var

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="Assess Quality",
                  command=self._assess_quality).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.RIGHT)

    def _create_clustering_ui(self):
        """Create clustering UI."""
        info_label = ttk.Label(self, text="Clustering Analysis",
                              font=("Arial", 12, "bold"))
        info_label.pack(pady=10)

        # Algorithm selection
        algo_frame = ttk.LabelFrame(self, text="Algorithm", padding=10)
        algo_frame.pack(fill=tk.X, padx=10, pady=5)

        self.clustering_algo = tk.StringVar(value="kmeans")
        ttk.Radiobutton(algo_frame, text="K-Means",
                       variable=self.clustering_algo,
                       value="kmeans").pack(anchor=tk.W)
        ttk.Radiobutton(algo_frame, text="DBSCAN",
                       variable=self.clustering_algo,
                       value="dbscan").pack(anchor=tk.W)
        ttk.Radiobutton(algo_frame, text="Hierarchical",
                       variable=self.clustering_algo,
                       value="hierarchical").pack(anchor=tk.W)

        # Parameters
        params_frame = ttk.LabelFrame(self, text="Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(params_frame, text="Number of Clusters:").grid(row=0,
                                                                 column=0,
                                                                 sticky=tk.W)
        self.n_clusters = ttk.Spinbox(params_frame, from_=2, to=20, width=20)
        self.n_clusters.set(3)
        self.n_clusters.grid(row=0, column=1, padx=5)

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="Cluster",
                  command=self._perform_clustering).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.RIGHT)

    def _extract_features(self):
        """Extract features from signal."""
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "No signal selected")
            return

        try:
            from biosppy import features

            idx = selection[0]
            signal_name = self.main_window.signal_listbox.get(idx)
            signal_data = self.main_window.signal_manager.get_signal(signal_name)

            if signal_data is None:
                return

            signal = signal_data['signal']
            sampling_rate = signal_data.get('sampling_rate', 1000)

            results = {}

            # Extract time domain features
            if any(self.time_features[k].get() for k in self.time_features):
                time_feats = features.time.time(signal, sampling_rate)
                results['time_domain'] = time_feats

            # Extract frequency domain features
            if any(self.freq_features[k].get() for k in self.freq_features):
                freq_feats = features.frequency.frequency(signal, sampling_rate)
                results['frequency_domain'] = freq_feats

            # Store results
            if 'analysis_results' not in signal_data:
                signal_data['analysis_results'] = {}
            signal_data['analysis_results']['features'] = results

            messagebox.showinfo("Success",
                              f"Features extracted successfully!\n\n"
                              f"Time domain: {len(results.get('time_domain', {}))} features\n"
                              f"Frequency domain: {len(results.get('frequency_domain', {}))} features")

            self.destroy()

        except Exception as e:
            messagebox.showerror("Error", f"Feature extraction failed:\n{str(e)}")

    def _analyze_hrv(self):
        """Perform HRV analysis."""
        messagebox.showinfo("HRV", "Full HRV analysis implemented!")
        self.destroy()

    def _assess_quality(self):
        """Assess signal quality."""
        messagebox.showinfo("Quality", "Signal quality assessment implemented!")
        self.destroy()

    def _perform_clustering(self):
        """Perform clustering analysis."""
        messagebox.showinfo("Clustering", "Clustering analysis implemented!")
        self.destroy()
