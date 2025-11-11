"""
Menu Bar
========

Complete menu bar with File, Edit, Process, Analyze, View, and Help menus.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np


class MenuBar:
    """Main application menu bar."""

    def __init__(self, root, main_window):
        """Initialize menu bar.

        Parameters
        ----------
        root : tk.Tk
            Root window.
        main_window : BioSPPyGUI
            Reference to main window.
        """
        self.root = root
        self.main_window = main_window

        # Create menu bar
        self.menubar = tk.Menu(root)
        root.config(menu=self.menubar)

        # File menu
        self._create_file_menu()

        # Edit menu
        self._create_edit_menu()

        # Process menu
        self._create_process_menu()

        # Analyze menu
        self._create_analyze_menu()

        # View menu
        self._create_view_menu()

        # Tools menu
        self._create_tools_menu()

        # Help menu
        self._create_help_menu()

    def _create_file_menu(self):
        """Create File menu."""
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)

        # Import submenu
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

        # Export submenu
        export_menu = tk.Menu(file_menu, tearoff=0)
        export_menu.add_command(label="Export Signal (.txt)...",
                               command=lambda: self.export_signal('txt'))
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

    def _create_edit_menu(self):
        """Create Edit menu."""
        edit_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Edit", menu=edit_menu)

        edit_menu.add_command(label="Undo", command=self.main_window.undo,
                            accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.main_window.redo,
                            accelerator="Ctrl+Y")

        edit_menu.add_separator()
        edit_menu.add_command(label="Remove Signal",
                            command=self.main_window.remove_signal,
                            accelerator="Delete")
        edit_menu.add_command(label="Clear All", command=self.clear_all)

        edit_menu.add_separator()
        edit_menu.add_command(label="Preferences...", command=self.show_preferences)

    def _create_process_menu(self):
        """Create Process menu."""
        process_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Process", menu=process_menu)

        # Signal type specific processing
        signal_types = {
            'ECG': 'ecg',
            'EDA/GSR': 'eda',
            'EMG': 'emg',
            'EEG': 'eeg',
            'PPG': 'ppg',
            'Respiration': 'resp',
            'BVP': 'bvp',
            'ABP': 'abp',
            'PPG': 'ppg',
            'Accelerometer': 'acc'
        }

        for label, sig_type in signal_types.items():
            process_menu.add_command(
                label=f"Process as {label}...",
                command=lambda t=sig_type: self.process_signal(t)
            )

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

    def _create_analyze_menu(self):
        """Create Analyze menu."""
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
        features_menu.add_command(label="Non-linear Features",
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

    def _create_view_menu(self):
        """Create View menu."""
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
        view_menu.add_command(label="Show Grid", command=self.toggle_grid)
        view_menu.add_command(label="Show Legend", command=self.toggle_legend)

        view_menu.add_separator()
        view_menu.add_command(label="Refresh", command=self.main_window.refresh_plot,
                            accelerator="F5")

    def _create_tools_menu(self):
        """Create Tools menu."""
        tools_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Tools", menu=tools_menu)

        tools_menu.add_command(label="Selection Mode",
                             command=self.toggle_selection_mode)
        tools_menu.add_command(label="Annotation Tool",
                             command=self.annotation_tool)

        tools_menu.add_separator()
        tools_menu.add_command(label="Batch Processing...",
                             command=self.batch_processing)
        tools_menu.add_command(label="Compare Signals...",
                             command=self.compare_signals)

    def _create_help_menu(self):
        """Create Help menu."""
        help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=help_menu)

        help_menu.add_command(label="Documentation",
                            command=self.show_help, accelerator="F1")
        help_menu.add_command(label="Keyboard Shortcuts",
                            command=self.show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="About BioSPPy", command=self.show_about)

    # File menu methods
    def import_signal(self, file_format='auto'):
        """Import a signal from file."""
        from .dialogs import ImportDialog
        dialog = ImportDialog(self.root, self.main_window, file_format)

    def export_signal(self, file_format):
        """Export current signal."""
        from .dialogs import ExportDialog
        dialog = ExportDialog(self.root, self.main_window, 'signal', file_format)

    def export_results(self, file_format):
        """Export processing results."""
        from .dialogs import ExportDialog
        dialog = ExportDialog(self.root, self.main_window, 'results', file_format)

    def export_figure(self, file_format):
        """Export current figure."""
        filename = filedialog.asksaveasfilename(
            title="Export Figure",
            defaultextension=f".{file_format}",
            filetypes=[(f"{file_format.upper()} files", f"*.{file_format}"),
                      ("All files", "*.*")]
        )

        if filename:
            self.main_window.plot_widget.save_figure(filename)
            self.main_window.statusbar.set_message(f"Figure saved: {filename}")

    def save_signal(self):
        """Save current signal."""
        messagebox.showinfo("Save", "Save functionality to be implemented")

    def save_signal_as(self):
        """Save signal with new name."""
        messagebox.showinfo("Save As", "Save As functionality to be implemented")

    # Edit menu methods
    def clear_all(self):
        """Clear all signals."""
        if messagebox.askyesno("Confirm", "Remove all signals?"):
            self.main_window.signal_manager.clear()
            self.main_window.signal_listbox.delete(0, tk.END)
            self.main_window.plot_widget.clear()

    def show_preferences(self):
        """Show preferences dialog."""
        from .dialogs import PreferencesDialog
        dialog = PreferencesDialog(self.root, self.main_window)

    # Process menu methods
    def process_signal(self, signal_type):
        """Process signal with type-specific algorithm."""
        from .dialogs import ProcessDialog
        dialog = ProcessDialog(self.root, self.main_window, signal_type)

    def apply_filter(self, filter_type):
        """Apply filter to signal."""
        from .dialogs import FilterDialog
        dialog = FilterDialog(self.root, self.main_window, filter_type)

    def resample_signal(self):
        """Resample signal."""
        from .dialogs import ResampleDialog
        dialog = ResampleDialog(self.root, self.main_window)

    def normalize_signal(self):
        """Normalize signal."""
        messagebox.showinfo("Normalize", "Normalize functionality to be implemented")

    def detrend_signal(self):
        """Detrend signal."""
        messagebox.showinfo("Detrend", "Detrend functionality to be implemented")

    def remove_baseline(self):
        """Remove baseline."""
        messagebox.showinfo("Remove Baseline",
                          "Remove baseline functionality to be implemented")

    # Analyze menu methods
    def analyze_selection(self):
        """Analyze selected segment."""
        from .dialogs import AnalyzeDialog
        dialog = AnalyzeDialog(self.root, self.main_window, selection_only=True)

    def analyze_full(self):
        """Analyze full signal."""
        from .dialogs import AnalyzeDialog
        dialog = AnalyzeDialog(self.root, self.main_window, selection_only=False)

    def extract_features(self, feature_type):
        """Extract features."""
        from .dialogs import FeatureExtractionDialog
        dialog = FeatureExtractionDialog(self.root, self.main_window, feature_type)

    def hrv_analysis(self):
        """Perform HRV analysis."""
        from .dialogs import HRVDialog
        dialog = HRVDialog(self.root, self.main_window)

    def assess_quality(self):
        """Assess signal quality."""
        messagebox.showinfo("Quality", "Quality assessment to be implemented")

    def show_statistics(self):
        """Show signal statistics."""
        from .dialogs import StatisticsDialog
        dialog = StatisticsDialog(self.root, self.main_window)

    # View menu methods
    def zoom(self, factor):
        """Zoom in/out."""
        # Implement zoom
        pass

    def zoom_to_selection(self):
        """Zoom to selected region."""
        pass

    def reset_view(self):
        """Reset view to default."""
        self.main_window.refresh_plot()

    def toggle_grid(self):
        """Toggle grid display."""
        pass

    def toggle_legend(self):
        """Toggle legend display."""
        pass

    # Tools menu methods
    def toggle_selection_mode(self):
        """Toggle selection mode."""
        plot_widget = self.main_window.plot_widget
        plot_widget.enable_selection_mode(not plot_widget.selection_mode)

    def annotation_tool(self):
        """Open annotation tool."""
        messagebox.showinfo("Annotation", "Annotation tool to be implemented")

    def batch_processing(self):
        """Open batch processing dialog."""
        from .dialogs import BatchProcessingDialog
        dialog = BatchProcessingDialog(self.root, self.main_window)

    def compare_signals(self):
        """Compare multiple signals."""
        messagebox.showinfo("Compare", "Signal comparison to be implemented")

    # Help menu methods
    def show_help(self):
        """Show help documentation."""
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

For more information, visit:
https://github.com/scientisst/BioSPPy
"""
        messagebox.showinfo("Help", help_text)

    def show_shortcuts(self):
        """Show keyboard shortcuts."""
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
        """Show about dialog."""
        about_text = """BioSPPy GUI
Version 2.0

Biosignal Processing in Python

A comprehensive toolbox for biosignal processing.

© 2024 BioSPPy Team
https://github.com/scientisst/BioSPPy
"""
        messagebox.showinfo("About BioSPPy", about_text)
