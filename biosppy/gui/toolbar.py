"""
Toolbar
=======

Toolbar with quick access buttons for common operations.
"""

import tkinter as tk
from tkinter import ttk


class Toolbar(ttk.Frame):
    """Toolbar with buttons for common operations."""

    def __init__(self, parent, main_window):
        """Initialize toolbar.

        Parameters
        ----------
        parent : tk.Widget
            Parent widget.
        main_window : BioSPPyGUI
            Reference to main window.
        """
        super().__init__(parent, relief=tk.RAISED, borderwidth=1)
        self.main_window = main_window

        # Button specifications: (text, command, tooltip)
        buttons = [
            ("📂 Open", self.open_file, "Import signal file (Ctrl+O)"),
            ("💾 Save", self.save_file, "Save current signal (Ctrl+S)"),
            ("|", None, None),  # Separator
            ("↶ Undo", self.main_window.undo, "Undo last operation (Ctrl+Z)"),
            ("↷ Redo", self.main_window.redo, "Redo last operation (Ctrl+Y)"),
            ("|", None, None),
            ("🔍 Zoom In", lambda: self.zoom(1.2), "Zoom in"),
            ("🔍 Zoom Out", lambda: self.zoom(0.8), "Zoom out"),
            ("🔄 Reset View", self.main_window.refresh_plot, "Reset view (F5)"),
            ("|", None, None),
            ("✂️ Select", self.toggle_selection, "Toggle selection mode"),
            ("📊 Analyze", self.analyze, "Analyze signal"),
            ("|", None, None),
            ("⚡ ECG", lambda: self.quick_process('ecg'), "Process as ECG"),
            ("💧 EDA", lambda: self.quick_process('eda'), "Process as EDA"),
            ("💪 EMG", lambda: self.quick_process('emg'), "Process as EMG"),
            ("🧠 EEG", lambda: self.quick_process('eeg'), "Process as EEG"),
            ("❤️ PPG", lambda: self.quick_process('ppg'), "Process as PPG"),
            ("|", None, None),
            ("🔧 Filter", self.filter_signal, "Apply filter"),
            ("📈 Features", self.extract_features, "Extract features"),
            ("|", None, None),
            ("❓ Help", self.show_help, "Show help (F1)"),
        ]

        # Create buttons
        for btn_spec in buttons:
            if btn_spec[0] == "|":
                # Separator
                ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT,
                                                             fill=tk.Y,
                                                             padx=5, pady=5)
            else:
                text, command, tooltip = btn_spec
                btn = ttk.Button(self, text=text, command=command, width=len(text)+2)
                btn.pack(side=tk.LEFT, padx=2, pady=2)

                # Add tooltip
                if tooltip:
                    self._create_tooltip(btn, tooltip)

    def _create_tooltip(self, widget, text):
        """Create tooltip for widget.

        Parameters
        ----------
        widget : tk.Widget
            Widget to add tooltip to.
        text : str
            Tooltip text.
        """
        def enter(event):
            self.main_window.statusbar.set_message(text)

        def leave(event):
            self.main_window.statusbar.set_message("Ready")

        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)

    # Button command methods
    def open_file(self):
        """Open file dialog."""
        self.main_window.menubar.import_signal('auto')

    def save_file(self):
        """Save current signal."""
        self.main_window.menubar.save_signal()

    def zoom(self, factor):
        """Zoom in/out."""
        # Implement zoom in plot widget
        pass

    def toggle_selection(self):
        """Toggle selection mode."""
        plot_widget = self.main_window.plot_widget
        plot_widget.enable_selection_mode(not plot_widget.selection_mode)

    def analyze(self):
        """Analyze current signal."""
        self.main_window.menubar.analyze_full()

    def quick_process(self, signal_type):
        """Quick process with default parameters."""
        self.main_window.menubar.process_signal(signal_type)

    def filter_signal(self):
        """Open filter dialog."""
        self.main_window.menubar.apply_filter('bandpass')

    def extract_features(self):
        """Open feature extraction dialog."""
        self.main_window.menubar.extract_features('time')

    def show_help(self):
        """Show help."""
        self.main_window.menubar.show_help()
