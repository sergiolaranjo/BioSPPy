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
        super().__init__(parent, relief=tk.RAISED, borderwidth=1)
        self.main_window = main_window

        buttons = [
            ("Open", self.open_file, "Import signal file (Ctrl+O)"),
            ("Save", self.save_file, "Save current signal (Ctrl+S)"),
            ("|", None, None),
            ("Undo", self.main_window.undo, "Undo last operation (Ctrl+Z)"),
            ("Redo", self.main_window.redo, "Redo last operation (Ctrl+Y)"),
            ("|", None, None),
            ("Zoom+", lambda: self.zoom(1.2), "Zoom in"),
            ("Zoom-", lambda: self.zoom(0.8), "Zoom out"),
            ("Reset", self.main_window.refresh_plot, "Reset view (F5)"),
            ("|", None, None),
            ("Select", self.toggle_selection, "Toggle selection mode"),
            ("Analyze", self.analyze, "Analyze signal"),
            ("|", None, None),
            ("ECG", lambda: self.quick_process('ecg'), "Process as ECG"),
            ("EDA", lambda: self.quick_process('eda'), "Process as EDA"),
            ("EMG", lambda: self.quick_process('emg'), "Process as EMG"),
            ("EEG", lambda: self.quick_process('eeg'), "Process as EEG"),
            ("PPG", lambda: self.quick_process('ppg'), "Process as PPG"),
            ("Resp", lambda: self.quick_process('resp'), "Process as Respiration"),
            ("PCG", lambda: self.quick_process('pcg'), "Process as PCG"),
            ("|", None, None),
            ("Filter", self.filter_signal, "Apply filter"),
            ("Features", self.extract_features, "Extract features"),
            ("HRV", self.hrv_analysis, "HRV analysis"),
            ("|", None, None),
            ("Help", self.show_help, "Show help (F1)"),
        ]

        for btn_spec in buttons:
            if btn_spec[0] == "|":
                ttk.Separator(self, orient=tk.VERTICAL).pack(
                    side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
            else:
                text, command, tooltip = btn_spec
                btn = ttk.Button(self, text=text, command=command,
                                width=max(len(text) + 1, 5))
                btn.pack(side=tk.LEFT, padx=2, pady=2)
                if tooltip:
                    self._create_tooltip(btn, tooltip)

    def _create_tooltip(self, widget, text):
        def enter(event):
            self.main_window.statusbar.set_message(text)
        def leave(event):
            self.main_window.statusbar.set_message("Ready")
        widget.bind('<Enter>', enter)
        widget.bind('<Leave>', leave)

    def open_file(self):
        self.main_window.menubar.import_signal('auto')

    def save_file(self):
        self.main_window.menubar.save_signal()

    def zoom(self, factor):
        self.main_window.menubar.zoom(factor)

    def toggle_selection(self):
        plot_widget = self.main_window.plot_widget
        plot_widget.enable_selection_mode(not plot_widget.selection_mode)

    def analyze(self):
        self.main_window.menubar.analyze_full()

    def quick_process(self, signal_type):
        self.main_window.menubar.process_signal(signal_type)

    def filter_signal(self):
        self.main_window.menubar.apply_filter('bandpass')

    def extract_features(self):
        self.main_window.menubar.extract_features('time')

    def hrv_analysis(self):
        self.main_window.menubar.hrv_analysis()

    def show_help(self):
        self.main_window.menubar.show_help()
