"""
Context Menus
=============

Right-click context menus for plots and signal list.
"""

import tkinter as tk
from tkinter import messagebox


class PlotContextMenu(tk.Menu):
    """Context menu for plot area."""

    def __init__(self, main_window, event):
        super().__init__(main_window.root, tearoff=0)
        self.main_window = main_window
        self.event = event

        self.add_command(label="Zoom In", command=self.zoom_in)
        self.add_command(label="Zoom Out", command=self.zoom_out)
        self.add_command(label="Reset View", command=self.reset_view)

        self.add_separator()
        self.add_command(label="Select Region", command=self.enable_selection)
        self.add_command(label="Clear Selection", command=self.clear_selection)

        self.add_separator()
        self.add_command(label="Analyze Selection", command=self.analyze_selection)
        self.add_command(label="Extract Features", command=self.extract_features)

        self.add_separator()
        self.add_command(label="Add Annotation", command=self.add_annotation)
        self.add_command(label="Add Marker", command=self.add_marker)

        self.add_separator()
        self.add_command(label="Toggle Grid", command=self.toggle_grid)
        self.add_command(label="Toggle Legend", command=self.toggle_legend)

        self.add_separator()
        self.add_command(label="Export Plot...", command=self.export_plot)
        self.add_command(label="Copy to Clipboard", command=self.copy_to_clipboard)

    def zoom_in(self):
        self.main_window.menubar.zoom(1.3)

    def zoom_out(self):
        self.main_window.menubar.zoom(0.7)

    def reset_view(self):
        self.main_window.refresh_plot()

    def enable_selection(self):
        self.main_window.plot_widget.enable_selection_mode(True)

    def clear_selection(self):
        if self.main_window.plot_widget.current_signal:
            self.main_window.plot_widget.current_signal.pop('selection', None)
        self.main_window.refresh_plot()

    def analyze_selection(self):
        self.main_window.menubar.analyze_selection()

    def extract_features(self):
        self.main_window.menubar.extract_features('time')

    def add_annotation(self):
        self.main_window.menubar.annotation_tool()

    def add_marker(self):
        """Add a vertical marker at the click position."""
        if self.event and self.event.inaxes and self.event.xdata:
            for ax in self.main_window.plot_widget.current_axes:
                ax.axvline(x=self.event.xdata, color='red', linestyle='--',
                          alpha=0.7, linewidth=1)
            self.main_window.plot_widget.canvas.draw()
            self.main_window.statusbar.set_message(
                f"Marker added at t={self.event.xdata:.3f}s")

    def toggle_grid(self):
        self.main_window.menubar.toggle_grid()

    def toggle_legend(self):
        self.main_window.menubar.toggle_legend()

    def export_plot(self):
        self.main_window.menubar.export_figure('png')

    def copy_to_clipboard(self):
        """Copy plot image to clipboard (save to temp file as fallback)."""
        try:
            import io
            buf = io.BytesIO()
            self.main_window.plot_widget.figure.savefig(buf, format='png',
                                                        dpi=150,
                                                        bbox_inches='tight')
            buf.seek(0)
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            tmp.write(buf.read())
            tmp.close()
            self.main_window.statusbar.set_message(
                f"Plot saved to: {tmp.name}")
        except Exception as e:
            messagebox.showerror("Error", f"Copy failed:\n{str(e)}")


class SignalContextMenu(tk.Menu):
    """Context menu for signal list."""

    def __init__(self, main_window, event):
        super().__init__(main_window.root, tearoff=0)
        self.main_window = main_window
        self.event = event

        listbox = main_window.signal_listbox
        index = listbox.nearest(event.y)
        if index >= 0:
            listbox.selection_clear(0, tk.END)
            listbox.selection_set(index)
            self.signal_name = listbox.get(index)
        else:
            self.signal_name = None

        self.add_command(label="Open", command=self.open_signal)
        self.add_command(label="Rename...", command=self.rename_signal)
        self.add_command(label="Duplicate", command=self.duplicate_signal)

        self.add_separator()

        process_menu = tk.Menu(self, tearoff=0)
        for label, sig_type in [('ECG', 'ecg'), ('EDA', 'eda'), ('EMG', 'emg'),
                                ('EEG', 'eeg'), ('PPG', 'ppg'), ('Respiration', 'resp'),
                                ('BVP', 'bvp'), ('ABP', 'abp'), ('PCG', 'pcg'),
                                ('Accelerometer', 'acc')]:
            process_menu.add_command(
                label=f"Process as {label}",
                command=lambda t=sig_type: self.process_as(t))
        self.add_cascade(label="Process As", menu=process_menu)

        self.add_separator()
        self.add_command(label="Export Signal...", command=self.export_signal)
        self.add_command(label="Export Results...", command=self.export_results)

        self.add_separator()
        self.add_command(label="Statistics", command=self.show_statistics)
        self.add_command(label="Properties", command=self.show_properties)

        self.add_separator()
        self.add_command(label="Remove", command=self.remove_signal)

    def open_signal(self):
        if self.signal_name:
            self.main_window.plot_signal(self.signal_name)

    def rename_signal(self):
        if self.signal_name:
            self.main_window.menubar.rename_signal()

    def duplicate_signal(self):
        if self.signal_name:
            self.main_window.menubar.duplicate_signal()

    def process_as(self, signal_type):
        if self.signal_name:
            self.main_window.menubar.process_signal(signal_type)

    def export_signal(self):
        if self.signal_name:
            self.main_window.menubar.export_signal('txt')

    def export_results(self):
        if self.signal_name:
            self.main_window.menubar.export_results('json')

    def show_statistics(self):
        if self.signal_name:
            self.main_window.menubar.show_statistics()

    def show_properties(self):
        if self.signal_name:
            self.main_window.update_signal_properties(self.signal_name)

    def remove_signal(self):
        self.main_window.remove_signal()
