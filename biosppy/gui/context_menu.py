"""
Context Menus
=============

Right-click context menus for plots and signal list.
"""

import tkinter as tk


class PlotContextMenu(tk.Menu):
    """Context menu for plot area."""

    def __init__(self, main_window, event):
        """Initialize plot context menu.

        Parameters
        ----------
        main_window : BioSPPyGUI
            Reference to main window.
        event : matplotlib event
            Mouse event.
        """
        super().__init__(main_window.root, tearoff=0)
        self.main_window = main_window
        self.event = event

        # Add menu items
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

        self.add_command(label="Export Plot...", command=self.export_plot)
        self.add_command(label="Copy to Clipboard", command=self.copy_to_clipboard)

    def zoom_in(self):
        """Zoom in at cursor position."""
        # Implement zoom at cursor
        pass

    def zoom_out(self):
        """Zoom out at cursor position."""
        # Implement zoom out
        pass

    def reset_view(self):
        """Reset view."""
        self.main_window.refresh_plot()

    def enable_selection(self):
        """Enable selection mode."""
        self.main_window.plot_widget.enable_selection_mode(True)

    def clear_selection(self):
        """Clear current selection."""
        self.main_window.plot_widget.current_signal.pop('selection', None)
        self.main_window.refresh_plot()

    def analyze_selection(self):
        """Analyze selected region."""
        self.main_window.menubar.analyze_selection()

    def extract_features(self):
        """Extract features from selection."""
        self.main_window.menubar.extract_features('time')

    def add_annotation(self):
        """Add text annotation at cursor."""
        # Implement annotation
        pass

    def add_marker(self):
        """Add marker at cursor."""
        # Implement marker
        pass

    def export_plot(self):
        """Export plot to file."""
        self.main_window.menubar.export_figure('png')

    def copy_to_clipboard(self):
        """Copy plot to clipboard."""
        # Implement clipboard copy
        pass


class SignalContextMenu(tk.Menu):
    """Context menu for signal list."""

    def __init__(self, main_window, event):
        """Initialize signal list context menu.

        Parameters
        ----------
        main_window : BioSPPyGUI
            Reference to main window.
        event : tk event
            Mouse event.
        """
        super().__init__(main_window.root, tearoff=0)
        self.main_window = main_window
        self.event = event

        # Get selected signal
        listbox = main_window.signal_listbox
        index = listbox.nearest(event.y)
        if index >= 0:
            listbox.selection_clear(0, tk.END)
            listbox.selection_set(index)
            self.signal_name = listbox.get(index)
        else:
            self.signal_name = None

        # Add menu items
        self.add_command(label="Open", command=self.open_signal)
        self.add_command(label="Rename", command=self.rename_signal)
        self.add_command(label="Duplicate", command=self.duplicate_signal)

        self.add_separator()

        # Process submenu
        process_menu = tk.Menu(self, tearoff=0)
        process_menu.add_command(label="Process as ECG",
                               command=lambda: self.process_as('ecg'))
        process_menu.add_command(label="Process as EDA",
                               command=lambda: self.process_as('eda'))
        process_menu.add_command(label="Process as EMG",
                               command=lambda: self.process_as('emg'))
        process_menu.add_command(label="Process as EEG",
                               command=lambda: self.process_as('eeg'))
        process_menu.add_command(label="Process as PPG",
                               command=lambda: self.process_as('ppg'))
        self.add_cascade(label="Process As", menu=process_menu)

        self.add_separator()

        self.add_command(label="Export Signal...", command=self.export_signal)
        self.add_command(label="Export Results...", command=self.export_results)

        self.add_separator()

        self.add_command(label="Properties", command=self.show_properties)

        self.add_separator()

        self.add_command(label="Remove", command=self.remove_signal)

    def open_signal(self):
        """Open/display selected signal."""
        if self.signal_name:
            self.main_window.plot_signal(self.signal_name)

    def rename_signal(self):
        """Rename signal."""
        # Implement rename dialog
        pass

    def duplicate_signal(self):
        """Duplicate signal."""
        # Implement duplication
        pass

    def process_as(self, signal_type):
        """Process signal as specified type."""
        if self.signal_name:
            self.main_window.menubar.process_signal(signal_type)

    def export_signal(self):
        """Export signal data."""
        if self.signal_name:
            self.main_window.menubar.export_signal('txt')

    def export_results(self):
        """Export processing results."""
        if self.signal_name:
            self.main_window.menubar.export_results('json')

    def show_properties(self):
        """Show signal properties dialog."""
        # Implement properties dialog
        pass

    def remove_signal(self):
        """Remove signal."""
        self.main_window.remove_signal()
