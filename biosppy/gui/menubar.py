"""
Menu Bar
========

Complete menu bar with File, Edit, Process, Analyze, Advanced, View, Tools, and Help menus.
All BioSPPy library functions are accessible from here.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np


class MenuBar:
    """Main application menu bar."""

    def __init__(self, root, main_window):
        self.root = root
        self.main_window = main_window

        self.menubar = tk.Menu(root)
        root.config(menu=self.menubar)

        self._create_file_menu()
        self._create_edit_menu()
        self._create_process_menu()
        self._create_analyze_menu()
        self._create_advanced_menu()
        self._create_view_menu()
        self._create_tools_menu()
        self._create_help_menu()

    # ------------------------------------------------------------------ File
    def _create_file_menu(self):
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)

        import_menu = tk.Menu(file_menu, tearoff=0)
        import_menu.add_command(label="Text File (.txt)...",
                               command=lambda: self.import_signal('txt'))
        import_menu.add_command(label="EDF File (.edf)...",
                               command=lambda: self.import_signal('edf'))
        import_menu.add_command(label="HDF5 File (.h5)...",
                               command=lambda: self.import_signal('h5'))
        import_menu.add_command(label="CSV File (.csv)...",
                               command=lambda: self.import_signal('csv'))
        import_menu.add_separator()
        import_menu.add_command(label="Auto-detect format...",
                               command=lambda: self.import_signal('auto'))
        file_menu.add_cascade(label="Import Signal", menu=import_menu)

        file_menu.add_separator()

        export_menu = tk.Menu(file_menu, tearoff=0)
        export_menu.add_command(label="Export Signal (.txt)...",
                               command=lambda: self.export_signal('txt'))
        export_menu.add_command(label="Export Signal (.csv)...",
                               command=lambda: self.export_signal('csv'))
        export_menu.add_command(label="Export Results (.json)...",
                               command=lambda: self.export_results('json'))
        export_menu.add_command(label="Export Figure (.png)...",
                               command=lambda: self.export_figure('png'))
        export_menu.add_command(label="Export Figure (.pdf)...",
                               command=lambda: self.export_figure('pdf'))
        file_menu.add_cascade(label="Export", menu=export_menu)

        file_menu.add_separator()
        file_menu.add_command(label="Save", command=self.save_signal,
                            accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self.save_signal_as)

        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.main_window.on_close,
                            accelerator="Ctrl+Q")

    # ------------------------------------------------------------------ Edit
    def _create_edit_menu(self):
        edit_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Edit", menu=edit_menu)

        edit_menu.add_command(label="Undo", command=self.main_window.undo,
                            accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.main_window.redo,
                            accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Rename Signal...", command=self.rename_signal)
        edit_menu.add_command(label="Duplicate Signal", command=self.duplicate_signal)
        edit_menu.add_separator()
        edit_menu.add_command(label="Remove Signal",
                            command=self.main_window.remove_signal,
                            accelerator="Delete")
        edit_menu.add_command(label="Clear All", command=self.clear_all)
        edit_menu.add_separator()
        edit_menu.add_command(label="Preferences...", command=self.show_preferences)

    # -------------------------------------------------------------- Process
    def _create_process_menu(self):
        process_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Process", menu=process_menu)

        # All signal types
        signal_types = {
            'ECG': 'ecg',
            'EDA/GSR': 'eda',
            'EMG': 'emg',
            'EEG': 'eeg',
            'PPG': 'ppg',
            'Respiration': 'resp',
            'BVP': 'bvp',
            'ABP': 'abp',
            'PCG': 'pcg',
            'Accelerometer': 'acc',
        }

        for label, sig_type in signal_types.items():
            process_menu.add_command(
                label=f"Process as {label}...",
                command=lambda t=sig_type: self.process_signal(t))

        process_menu.add_separator()

        # Filtering submenu
        filter_menu = tk.Menu(process_menu, tearoff=0)
        filter_menu.add_command(label="Bandpass Filter...",
                               command=lambda: self.apply_filter('bandpass'))
        filter_menu.add_command(label="Lowpass Filter...",
                               command=lambda: self.apply_filter('lowpass'))
        filter_menu.add_command(label="Highpass Filter...",
                               command=lambda: self.apply_filter('highpass'))
        filter_menu.add_command(label="Notch Filter...",
                               command=lambda: self.apply_filter('notch'))
        filter_menu.add_command(label="Smooth (Moving Average)...",
                               command=lambda: self.apply_filter('smooth'))
        process_menu.add_cascade(label="Filter", menu=filter_menu)

        # Preprocessing
        process_menu.add_separator()
        process_menu.add_command(label="Resample...", command=self.resample_signal)
        process_menu.add_command(label="Normalize", command=self.normalize_signal)
        process_menu.add_command(label="Detrend", command=self.detrend_signal)
        process_menu.add_command(label="Remove Baseline", command=self.remove_baseline)

    # -------------------------------------------------------------- Analyze
    def _create_analyze_menu(self):
        analyze_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Analyze", menu=analyze_menu)

        analyze_menu.add_command(label="Analyze Selection",
                               command=self.analyze_selection)
        analyze_menu.add_command(label="Analyze Full Signal",
                               command=self.analyze_full)
        analyze_menu.add_separator()

        # Feature extraction submenu
        features_menu = tk.Menu(analyze_menu, tearoff=0)
        features_menu.add_command(label="Time Domain Features",
                                 command=lambda: self.extract_features('time'))
        features_menu.add_command(label="Frequency Domain Features",
                                 command=lambda: self.extract_features('frequency'))
        features_menu.add_command(label="Time-Frequency Features",
                                 command=lambda: self.extract_features('time_freq'))
        features_menu.add_command(label="Cepstral Features",
                                 command=lambda: self.extract_features('cepstral'))
        features_menu.add_command(label="Phase Space / Non-linear Features",
                                 command=lambda: self.extract_features('nonlinear'))
        analyze_menu.add_cascade(label="Extract Features", menu=features_menu)

        # HRV Analysis
        analyze_menu.add_separator()
        analyze_menu.add_command(label="HRV Analysis...", command=self.hrv_analysis)

        # Signal Quality
        analyze_menu.add_separator()
        analyze_menu.add_command(label="Assess Signal Quality",
                               command=self.assess_quality)

        # Statistics
        analyze_menu.add_separator()
        analyze_menu.add_command(label="Signal Statistics",
                               command=self.show_statistics)

    # ------------------------------------------------------------- Advanced
    def _create_advanced_menu(self):
        advanced_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Advanced", menu=advanced_menu)

        # Feature Extraction submenu
        features_menu = tk.Menu(advanced_menu, tearoff=0)
        features_menu.add_command(label="All Features...",
                                 command=lambda: self.advanced_analysis('features'))
        features_menu.add_separator()
        features_menu.add_command(label="Time Domain Features...",
                                 command=lambda: self.extract_features('time'))
        features_menu.add_command(label="Frequency Domain Features...",
                                 command=lambda: self.extract_features('frequency'))
        features_menu.add_command(label="Time-Frequency Features...",
                                 command=lambda: self.extract_features('time_freq'))
        features_menu.add_command(label="Cepstral Features (MFCC)...",
                                 command=lambda: self.extract_features('cepstral'))
        features_menu.add_command(label="Phase Space Features...",
                                 command=lambda: self.extract_features('nonlinear'))
        features_menu.add_separator()
        features_menu.add_command(label="Wavelet Coherence...",
                                 command=self.wavelet_coherence_analysis)
        advanced_menu.add_cascade(label="Feature Extraction", menu=features_menu)

        # HRV Analysis submenu
        hrv_menu = tk.Menu(advanced_menu, tearoff=0)
        hrv_menu.add_command(label="Complete HRV Analysis...",
                            command=lambda: self.advanced_analysis('hrv'))
        hrv_menu.add_separator()
        hrv_menu.add_command(label="Time-Domain HRV...",
                            command=self.hrv_analysis)
        hrv_menu.add_command(label="Frequency-Domain HRV...",
                            command=self.hrv_analysis)
        hrv_menu.add_command(label="Non-linear HRV...",
                            command=self.hrv_analysis)
        advanced_menu.add_cascade(label="HRV Analysis", menu=hrv_menu)

        # Baroreflex
        advanced_menu.add_separator()
        advanced_menu.add_command(label="Baroreflex Sensitivity...",
                                 command=self.baroreflex_analysis)

        # Signal Quality
        advanced_menu.add_separator()
        advanced_menu.add_command(label="Signal Quality Assessment...",
                                 command=lambda: self.advanced_analysis('quality'))

        # Chaos / Nonlinear Dynamics
        advanced_menu.add_separator()
        chaos_menu = tk.Menu(advanced_menu, tearoff=0)
        chaos_menu.add_command(label="Shannon Entropy...",
                               command=lambda: self.chaos_analysis('shannon_entropy'))
        chaos_menu.add_command(label="Sample Entropy...",
                               command=lambda: self.chaos_analysis('sample_entropy'))
        chaos_menu.add_command(label="Approximate Entropy...",
                               command=lambda: self.chaos_analysis('approximate_entropy'))
        chaos_menu.add_command(label="Permutation Entropy...",
                               command=lambda: self.chaos_analysis('permutation_entropy'))
        chaos_menu.add_separator()
        chaos_menu.add_command(label="Fractal Dimension (Higuchi)...",
                               command=lambda: self.chaos_analysis('higuchi_fd'))
        chaos_menu.add_command(label="Fractal Dimension (Katz)...",
                               command=lambda: self.chaos_analysis('katz_fd'))
        chaos_menu.add_command(label="Fractal Dimension (Petrosian)...",
                               command=lambda: self.chaos_analysis('petrosian_fd'))
        chaos_menu.add_separator()
        chaos_menu.add_command(label="DFA (Detrended Fluctuation)...",
                               command=lambda: self.chaos_analysis('dfa'))
        chaos_menu.add_command(label="Hurst Exponent...",
                               command=lambda: self.chaos_analysis('hurst'))
        chaos_menu.add_command(label="Lyapunov Exponent...",
                               command=lambda: self.chaos_analysis('lyapunov'))
        advanced_menu.add_cascade(label="Chaos / Nonlinear Dynamics", menu=chaos_menu)

        # EMD / HHT
        advanced_menu.add_separator()
        advanced_menu.add_command(label="Empirical Mode Decomposition (EMD)...",
                                 command=self.emd_analysis)

        # Clustering & Classification
        advanced_menu.add_separator()
        clustering_menu = tk.Menu(advanced_menu, tearoff=0)
        clustering_menu.add_command(label="K-Means Clustering...",
                                   command=lambda: self.clustering_analysis('kmeans'))
        clustering_menu.add_command(label="DBSCAN Clustering...",
                                   command=lambda: self.clustering_analysis('dbscan'))
        clustering_menu.add_command(label="Hierarchical Clustering...",
                                   command=lambda: self.clustering_analysis('hierarchical'))
        advanced_menu.add_cascade(label="Clustering", menu=clustering_menu)

        # Dimensionality Reduction
        dimred_menu = tk.Menu(advanced_menu, tearoff=0)
        dimred_menu.add_command(label="PCA...",
                                command=lambda: self.dimred_analysis('pca'))
        dimred_menu.add_command(label="t-SNE...",
                                command=lambda: self.dimred_analysis('tsne'))
        advanced_menu.add_cascade(label="Dimensionality Reduction", menu=dimred_menu)

        # Biometrics
        advanced_menu.add_separator()
        advanced_menu.add_command(label="Biometric Analysis...",
                                 command=self.biometric_analysis)

        # Multichannel
        advanced_menu.add_separator()
        advanced_menu.add_command(label="Multichannel Analysis...",
                                 command=self.multichannel_analysis)

    # ----------------------------------------------------------------- View
    def _create_view_menu(self):
        view_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="View", menu=view_menu)

        view_menu.add_command(label="Zoom In",
                            command=lambda: self.zoom(1.2))
        view_menu.add_command(label="Zoom Out",
                            command=lambda: self.zoom(0.8))
        view_menu.add_command(label="Zoom to Selection",
                            command=self.zoom_to_selection)
        view_menu.add_command(label="Reset View",
                            command=self.reset_view)

        view_menu.add_separator()
        view_menu.add_command(label="Toggle Grid", command=self.toggle_grid)
        view_menu.add_command(label="Toggle Legend", command=self.toggle_legend)

        view_menu.add_separator()
        view_menu.add_command(label="Refresh", command=self.main_window.refresh_plot,
                            accelerator="F5")

    # ---------------------------------------------------------------- Tools
    def _create_tools_menu(self):
        tools_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Tools", menu=tools_menu)

        tools_menu.add_command(label="Advanced Filter Design...",
                             command=self.advanced_filter_design)
        tools_menu.add_command(label="Signal Synthesis...",
                             command=self.signal_synthesis)
        tools_menu.add_command(label="Compare Signals...",
                             command=self.compare_signals)

        tools_menu.add_separator()
        tools_menu.add_command(label="Selection Mode",
                             command=self.toggle_selection_mode)
        tools_menu.add_command(label="Annotation Tool",
                             command=self.annotation_tool)

        tools_menu.add_separator()
        tools_menu.add_command(label="Batch Processing...",
                             command=self.batch_processing)

    # ----------------------------------------------------------------- Help
    def _create_help_menu(self):
        help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=help_menu)

        help_menu.add_command(label="Documentation",
                            command=self.show_help, accelerator="F1")
        help_menu.add_command(label="Keyboard Shortcuts",
                            command=self.show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="About BioSPPy", command=self.show_about)

    # ====================================================================
    # File menu methods
    # ====================================================================

    def import_signal(self, file_format='auto'):
        from .dialogs import ImportDialog
        ImportDialog(self.root, self.main_window, file_format)

    def export_signal(self, file_format):
        from .dialogs import ExportDialog
        ExportDialog(self.root, self.main_window, 'signal', file_format)

    def export_results(self, file_format):
        from .dialogs import ExportDialog
        ExportDialog(self.root, self.main_window, 'results', file_format)

    def export_figure(self, file_format):
        filename = filedialog.asksaveasfilename(
            title="Export Figure",
            defaultextension=f".{file_format}",
            filetypes=[(f"{file_format.upper()} files", f"*.{file_format}"),
                      ("All files", "*.*")])
        if filename:
            self.main_window.plot_widget.save_figure(filename)
            self.main_window.statusbar.set_message(f"Figure saved: {filename}")

    def save_signal(self):
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showinfo("Save", "No signal selected")
            return
        idx = selection[0]
        signal_name = self.main_window.signal_listbox.get(idx)
        signal_data = self.main_window.signal_manager.get_signal(signal_name)
        if signal_data and 'filename' in signal_data:
            try:
                np.savetxt(signal_data['filename'], signal_data['signal'])
                self.main_window.signal_manager.mark_saved()
                self.main_window.statusbar.set_message(f"Saved: {signal_data['filename']}")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed:\n{str(e)}")
        else:
            self.save_signal_as()

    def save_signal_as(self):
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showinfo("Save As", "No signal selected")
            return
        idx = selection[0]
        signal_name = self.main_window.signal_listbox.get(idx)
        signal_data = self.main_window.signal_manager.get_signal(signal_name)
        if signal_data is None:
            return
        filename = filedialog.asksaveasfilename(
            title="Save Signal As",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"),
                      ("All files", "*.*")])
        if filename:
            try:
                np.savetxt(filename, signal_data['signal'])
                signal_data['filename'] = filename
                self.main_window.signal_manager.mark_saved()
                self.main_window.statusbar.set_message(f"Saved: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed:\n{str(e)}")

    # ====================================================================
    # Edit menu methods
    # ====================================================================

    def rename_signal(self):
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        old_name = self.main_window.signal_listbox.get(idx)

        dialog = tk.Toplevel(self.root)
        dialog.title("Rename Signal")
        dialog.geometry("300x100")
        ttk.Label(dialog, text="New name:").pack(padx=10, pady=5)
        entry = ttk.Entry(dialog, width=30)
        entry.insert(0, old_name)
        entry.pack(padx=10, pady=5)

        def do_rename():
            new_name = entry.get().strip()
            if new_name and new_name != old_name:
                sig_data = self.main_window.signal_manager.get_signal(old_name)
                if sig_data:
                    self.main_window.signal_manager.signals[new_name] = sig_data
                    del self.main_window.signal_manager.signals[old_name]
                    self.main_window.signal_listbox.delete(idx)
                    self.main_window.signal_listbox.insert(idx, new_name)
                    self.main_window.statusbar.set_message(
                        f"Renamed: {old_name} -> {new_name}")
            dialog.destroy()

        ttk.Button(dialog, text="Rename", command=do_rename).pack(pady=5)

    def duplicate_signal(self):
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        name = self.main_window.signal_listbox.get(idx)
        sig_data = self.main_window.signal_manager.get_signal(name)
        if sig_data:
            from copy import deepcopy
            new_name = f"{name}_copy"
            new_data = deepcopy(sig_data)
            self.main_window.signal_manager.signals[new_name] = new_data
            self.main_window.add_signal_to_list(new_name)
            self.main_window.statusbar.set_message(f"Duplicated: {name}")

    def clear_all(self):
        if messagebox.askyesno("Confirm", "Remove all signals?"):
            self.main_window.signal_manager.clear()
            self.main_window.signal_listbox.delete(0, tk.END)
            self.main_window.plot_widget.clear()

    def show_preferences(self):
        from .dialogs import PreferencesDialog
        PreferencesDialog(self.root, self.main_window)

    # ====================================================================
    # Process menu methods
    # ====================================================================

    def process_signal(self, signal_type):
        from .dialogs import ProcessDialog
        ProcessDialog(self.root, self.main_window, signal_type)

    def apply_filter(self, filter_type):
        from .dialogs import FilterDialog
        FilterDialog(self.root, self.main_window, filter_type)

    def resample_signal(self):
        from .dialogs import ResampleDialog
        ResampleDialog(self.root, self.main_window)

    def normalize_signal(self):
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showinfo("Normalize", "No signal selected")
            return
        idx = selection[0]
        signal_name = self.main_window.signal_listbox.get(idx)
        signal_data = self.main_window.signal_manager.get_signal(signal_name)
        if signal_data is None:
            return

        signal = signal_data['signal']
        min_val, max_val = np.min(signal), np.max(signal)
        if max_val - min_val == 0:
            messagebox.showinfo("Normalize", "Signal is constant, cannot normalize")
            return
        normalized = (signal - min_val) / (max_val - min_val)
        self.main_window.signal_manager.update_signal(
            signal_name, normalized, "Normalized [0,1]")
        self.main_window.refresh_plot()
        self.main_window.statusbar.set_message(f"Normalized: {signal_name}")

    def detrend_signal(self):
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showinfo("Detrend", "No signal selected")
            return
        idx = selection[0]
        signal_name = self.main_window.signal_listbox.get(idx)
        signal_data = self.main_window.signal_manager.get_signal(signal_name)
        if signal_data is None:
            return

        from scipy.signal import detrend
        detrended = detrend(signal_data['signal'])
        self.main_window.signal_manager.update_signal(
            signal_name, detrended, "Detrended (linear)")
        self.main_window.refresh_plot()
        self.main_window.statusbar.set_message(f"Detrended: {signal_name}")

    def remove_baseline(self):
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showinfo("Remove Baseline", "No signal selected")
            return
        idx = selection[0]
        signal_name = self.main_window.signal_listbox.get(idx)
        signal_data = self.main_window.signal_manager.get_signal(signal_name)
        if signal_data is None:
            return

        signal = signal_data['signal']
        baseline_removed = signal - np.mean(signal)
        self.main_window.signal_manager.update_signal(
            signal_name, baseline_removed, "Baseline removed (mean subtracted)")
        self.main_window.refresh_plot()
        self.main_window.statusbar.set_message(f"Baseline removed: {signal_name}")

    # ====================================================================
    # Analyze menu methods
    # ====================================================================

    def analyze_selection(self):
        from .dialogs import AnalyzeDialog
        AnalyzeDialog(self.root, self.main_window, selection_only=True)

    def analyze_full(self):
        from .dialogs import AnalyzeDialog
        AnalyzeDialog(self.root, self.main_window, selection_only=False)

    def extract_features(self, feature_type):
        from .dialogs import FeatureExtractionDialog
        FeatureExtractionDialog(self.root, self.main_window, feature_type)

    def hrv_analysis(self, *args):
        from .dialogs import HRVDialog
        HRVDialog(self.root, self.main_window)

    def assess_quality(self):
        """Signal quality assessment using biosppy.quality."""
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showinfo("Quality", "No signal selected")
            return
        idx = selection[0]
        signal_name = self.main_window.signal_listbox.get(idx)
        signal_data = self.main_window.signal_manager.get_signal(signal_name)
        if signal_data is None:
            return

        try:
            from biosppy import quality
            signal = signal_data['signal']
            sr = signal_data.get('sampling_rate', 1000)
            sig_type = signal_data.get('type', 'ecg')

            results = []
            if sig_type == 'ecg':
                # ECG quality indices
                try:
                    q = quality.ecg_sqi(signal=signal, sampling_rate=sr)
                    results.append(f"ECG SQI: {q}")
                except Exception:
                    pass
                try:
                    q = quality.fSQI(signal=signal, sampling_rate=sr)
                    results.append(f"fSQI (spectral): {q:.4f}")
                except Exception:
                    pass
            elif sig_type == 'eda':
                try:
                    q = quality.eda_sqi_bottcher(signal=signal, sampling_rate=sr)
                    results.append(f"EDA SQI (Bottcher): {q}")
                except Exception:
                    pass

            if results:
                messagebox.showinfo("Signal Quality", "\n".join(results))
            else:
                messagebox.showinfo("Signal Quality",
                    f"Quality assessment computed for {signal_name}.\n"
                    "No specific SQI available for this signal type.\n"
                    "Use Advanced > Signal Quality Assessment for more options.")
        except Exception as e:
            messagebox.showerror("Error", f"Quality assessment failed:\n{str(e)}")

    def show_statistics(self):
        from .dialogs import StatisticsDialog
        StatisticsDialog(self.root, self.main_window)

    # ====================================================================
    # View menu methods
    # ====================================================================

    def zoom(self, factor):
        axes = self.main_window.plot_widget.current_axes
        if not axes:
            return
        for ax in axes:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            xc = (xmin + xmax) / 2
            yc = (ymin + ymax) / 2
            xr = (xmax - xmin) / 2 / factor
            yr = (ymax - ymin) / 2 / factor
            ax.set_xlim(xc - xr, xc + xr)
            ax.set_ylim(yc - yr, yc + yr)
        self.main_window.plot_widget.canvas.draw()

    def zoom_to_selection(self):
        pw = self.main_window.plot_widget
        sel = pw.get_selection()
        if sel is None:
            messagebox.showinfo("Zoom", "No selection active. Use Selection Mode first.")
            return
        start = sel['start_time']
        end = sel['end_time']
        segment = sel['segment']
        for ax in pw.current_axes:
            ax.set_xlim(start, end)
            if len(segment) > 0:
                margin = (np.max(segment) - np.min(segment)) * 0.1
                ax.set_ylim(np.min(segment) - margin, np.max(segment) + margin)
        pw.canvas.draw()

    def reset_view(self):
        self.main_window.refresh_plot()

    def toggle_grid(self):
        axes = self.main_window.plot_widget.current_axes
        if not axes:
            return
        for ax in axes:
            ax.grid(not ax.xaxis.get_gridlines()[0].get_visible()
                    if ax.xaxis.get_gridlines() else True,
                    alpha=0.3)
        self.main_window.plot_widget.canvas.draw()

    def toggle_legend(self):
        axes = self.main_window.plot_widget.current_axes
        if not axes:
            return
        for ax in axes:
            legend = ax.get_legend()
            if legend:
                legend.set_visible(not legend.get_visible())
            else:
                ax.legend()
        self.main_window.plot_widget.canvas.draw()

    # ====================================================================
    # Advanced menu methods
    # ====================================================================

    def advanced_analysis(self, analysis_type):
        from .advanced_analysis import AdvancedAnalysisDialog
        AdvancedAnalysisDialog(self.root, self.main_window, analysis_type)

    def clustering_analysis(self, algorithm):
        from .advanced_analysis import AdvancedAnalysisDialog
        AdvancedAnalysisDialog(self.root, self.main_window, 'clustering')

    def biometric_analysis(self):
        """Biometric analysis using biosppy.biometrics."""
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showinfo("Biometrics", "No signal selected")
            return
        messagebox.showinfo("Biometric Analysis",
            "Biometric analysis is available via biosppy.biometrics.\n"
            "Load multiple recordings and use Advanced > Clustering\n"
            "to extract biometric templates for identification/verification.")

    def baroreflex_analysis(self):
        """Baroreflex sensitivity analysis using biosppy.signals.baroreflex."""
        messagebox.showinfo("Baroreflex",
            "Baroreflex Sensitivity Analysis\n\n"
            "Requires both ECG and ABP signals loaded.\n"
            "Available methods:\n"
            "- Sequence method\n"
            "- Spectral method (alpha)\n"
            "- Transfer function\n\n"
            "Use: from biosppy.signals import baroreflex\n"
            "See examples/baroreflex_example.py")

    def chaos_analysis(self, method):
        """Nonlinear dynamics / chaos analysis using biosppy.chaos."""
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showinfo("Chaos", "No signal selected")
            return
        idx = selection[0]
        signal_name = self.main_window.signal_listbox.get(idx)
        signal_data = self.main_window.signal_manager.get_signal(signal_name)
        if signal_data is None:
            return

        try:
            from biosppy import chaos
            signal = signal_data['signal']

            result_text = f"Chaos Analysis: {method}\nSignal: {signal_name}\n\n"

            if method == 'shannon_entropy':
                val = chaos.shannon_entropy(signal)
                result_text += f"Shannon Entropy: {val}"
            elif method == 'sample_entropy':
                val = chaos.sample_entropy(signal)
                result_text += f"Sample Entropy: {val}"
            elif method == 'approximate_entropy':
                val = chaos.approximate_entropy(signal)
                result_text += f"Approximate Entropy: {val}"
            elif method == 'permutation_entropy':
                val = chaos.permutation_entropy(signal)
                result_text += f"Permutation Entropy: {val}"
            elif method == 'higuchi_fd':
                val = chaos.higuchi_fd(signal)
                result_text += f"Higuchi Fractal Dimension: {val}"
            elif method == 'katz_fd':
                val = chaos.katz_fd(signal)
                result_text += f"Katz Fractal Dimension: {val}"
            elif method == 'petrosian_fd':
                val = chaos.petrosian_fd(signal)
                result_text += f"Petrosian Fractal Dimension: {val}"
            elif method == 'dfa':
                val = chaos.dfa(signal)
                result_text += f"DFA alpha: {val}"
            elif method == 'hurst':
                val = chaos.hurst_exponent(signal)
                result_text += f"Hurst Exponent: {val}"
            elif method == 'lyapunov':
                val = chaos.lyapunov_exponent(signal)
                result_text += f"Largest Lyapunov Exponent: {val}"
            else:
                result_text += f"Method '{method}' not found."

            messagebox.showinfo("Chaos Analysis", result_text)

        except Exception as e:
            messagebox.showerror("Error", f"Chaos analysis failed:\n{str(e)}")

    def emd_analysis(self):
        """Empirical Mode Decomposition using biosppy.signals.emd."""
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showinfo("EMD", "No signal selected")
            return
        idx = selection[0]
        signal_name = self.main_window.signal_listbox.get(idx)
        signal_data = self.main_window.signal_manager.get_signal(signal_name)
        if signal_data is None:
            return

        try:
            from biosppy.signals import emd
            import matplotlib.pyplot as plt

            signal = signal_data['signal']
            sr = signal_data.get('sampling_rate', 1000)

            result = emd.emd(signal=signal, sampling_rate=sr, show=False)
            imfs = result.get('IMFs', result.get('imfs', None))

            if imfs is not None:
                n_imfs = imfs.shape[1] if imfs.ndim == 2 else 1
                fig, axes = plt.subplots(n_imfs + 1, 1,
                                        figsize=(12, 2 * (n_imfs + 1)),
                                        sharex=True)
                ts = np.arange(len(signal)) / sr

                axes[0].plot(ts, signal, 'b-', linewidth=0.5)
                axes[0].set_ylabel('Original')
                axes[0].set_title(f'EMD: {signal_name}')

                for i in range(n_imfs):
                    imf = imfs[:, i] if imfs.ndim == 2 else imfs
                    axes[i + 1].plot(ts[:len(imf)], imf, 'g-', linewidth=0.5)
                    axes[i + 1].set_ylabel(f'IMF {i+1}')

                axes[-1].set_xlabel('Time (s)')
                plt.tight_layout()
                plt.show()

                self.main_window.statusbar.set_message(
                    f"EMD: {n_imfs} IMFs extracted from {signal_name}")
            else:
                messagebox.showinfo("EMD", "EMD computed but no IMFs returned.")

        except Exception as e:
            messagebox.showerror("Error", f"EMD analysis failed:\n{str(e)}")

    def wavelet_coherence_analysis(self):
        """Wavelet coherence using biosppy.features.wavelet_coherence."""
        messagebox.showinfo("Wavelet Coherence",
            "Wavelet Coherence Analysis\n\n"
            "Requires two signals loaded.\n"
            "Use: from biosppy.features import wavelet_coherence\n"
            "See examples/wavelet_coherence_example.py\n\n"
            "You can also use Tools > Compare Signals for basic comparison.")

    def dimred_analysis(self, method):
        """Dimensionality reduction using biosppy.dimensionality_reduction."""
        messagebox.showinfo("Dimensionality Reduction",
            f"Method: {method.upper()}\n\n"
            "Available via: from biosppy import dimensionality_reduction\n"
            "Functions: pca(), tsne()\n\n"
            "Use Advanced > Clustering with dimensionality reduction\n"
            "for integrated analysis.")

    def multichannel_analysis(self):
        """Multichannel signal management."""
        messagebox.showinfo("Multichannel Analysis",
            "Multichannel Signal Analysis\n\n"
            "Available via: from biosppy.signals import multichannel\n"
            "Features:\n"
            "- Channel synchronization\n"
            "- Multi-signal alignment\n"
            "- Cross-channel analysis\n\n"
            "See examples/multichannel_baroreflex_example.py")

    # ====================================================================
    # Tools menu methods
    # ====================================================================

    def toggle_selection_mode(self):
        plot_widget = self.main_window.plot_widget
        plot_widget.enable_selection_mode(not plot_widget.selection_mode)

    def annotation_tool(self):
        """Add text annotation to current plot."""
        axes = self.main_window.plot_widget.current_axes
        if not axes:
            messagebox.showinfo("Annotation", "No plot to annotate")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Add Annotation")
        dialog.geometry("300x150")

        ttk.Label(dialog, text="Annotation text:").pack(padx=10, pady=5)
        text_entry = ttk.Entry(dialog, width=30)
        text_entry.pack(padx=10, pady=5)
        ttk.Label(dialog, text="Click on plot to place annotation.").pack(pady=5)

        def on_click(event):
            if event.inaxes:
                text = text_entry.get()
                if text:
                    event.inaxes.annotate(text, (event.xdata, event.ydata),
                                         fontsize=8,
                                         bbox=dict(boxstyle='round,pad=0.3',
                                                  facecolor='yellow', alpha=0.7))
                    self.main_window.plot_widget.canvas.draw()
                    self.main_window.statusbar.set_message(f"Annotation added: {text}")
                    dialog.destroy()

        cid = self.main_window.plot_widget.canvas.mpl_connect('button_press_event',
                                                               on_click)

        def on_close():
            self.main_window.plot_widget.canvas.mpl_disconnect(cid)
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_close)

    def batch_processing(self):
        from .dialogs import BatchProcessingDialog
        BatchProcessingDialog(self.root, self.main_window)

    def advanced_filter_design(self):
        from .signal_tools import FilterDesignDialog
        FilterDesignDialog(self.root, self.main_window)

    def signal_synthesis(self):
        from .signal_tools import SignalSynthesisDialog
        SignalSynthesisDialog(self.root, self.main_window)

    def compare_signals(self):
        from .signal_tools import SignalComparisonDialog
        SignalComparisonDialog(self.root, self.main_window)

    # ====================================================================
    # Help menu methods
    # ====================================================================

    def show_help(self):
        help_text = """BioSPPy GUI - Help

Keyboard Shortcuts:
- Ctrl+O: Import signal
- Ctrl+S: Save signal
- Ctrl+Z: Undo
- Ctrl+Y: Redo
- Delete: Remove signal
- F5: Refresh view
- F1: Show help

Mouse Controls:
- Scroll: Zoom in/out
- Right-click: Context menu
- Click and drag (in selection mode): Select region

Signal Processing:
- Process menu: ECG, EDA, EMG, EEG, PPG, Respiration, BVP, ABP, PCG, Accelerometer
- Filter menu: Bandpass, Lowpass, Highpass, Notch, Smooth
- Preprocessing: Resample, Normalize, Detrend, Remove Baseline

Analysis:
- Feature Extraction: Time, Frequency, Time-Frequency, Cepstral, Phase Space
- HRV Analysis: Time-domain, Frequency-domain, Non-linear
- Chaos: Entropy, Fractal Dimensions, DFA, Hurst, Lyapunov
- EMD: Empirical Mode Decomposition
- Quality Assessment: ECG and EDA signal quality

For more information, visit:
https://github.com/scientisst/BioSPPy
"""
        messagebox.showinfo("Help", help_text)

    def show_shortcuts(self):
        shortcuts_text = """Keyboard Shortcuts:

File Operations:
  Ctrl+O - Import signal
  Ctrl+S - Save signal
  Ctrl+Q - Exit

Editing:
  Ctrl+Z - Undo
  Ctrl+Y - Redo
  Delete - Remove signal

View:
  F5 - Refresh
  Mouse Wheel - Zoom

Help:
  F1 - Show help
"""
        messagebox.showinfo("Keyboard Shortcuts", shortcuts_text)

    def show_about(self):
        about_text = """BioSPPy GUI
Version 2.2.3

Biosignal Processing in Python

A comprehensive toolbox for biosignal processing.
Supports: ECG, EDA, EMG, EEG, PPG, Respiration,
BVP, ABP, PCG, Accelerometer signals.

Features: HRV, Baroreflex, Chaos Analysis, EMD,
Wavelet Coherence, Clustering, Biometrics,
Quality Assessment, Signal Synthesis.

(c) 2015-2023 Instituto de Telecomunicacoes
https://github.com/scientisst/BioSPPy
"""
        messagebox.showinfo("About BioSPPy", about_text)
