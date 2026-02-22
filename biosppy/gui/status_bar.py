"""
Status Bar
==========

Status bar for displaying messages and progress.
"""

import tkinter as tk
from tkinter import ttk


class StatusBar(ttk.Frame):
    """Status bar with message display and progress indicator."""

    def __init__(self, parent):
        """Initialize status bar.

        Parameters
        ----------
        parent : tk.Widget
            Parent widget.
        """
        super().__init__(parent, relief=tk.SUNKEN, borderwidth=1)

        # Message label
        self.message_label = ttk.Label(self, text="Ready", anchor=tk.W)
        self.message_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Progress bar (hidden by default)
        self.progress = ttk.Progressbar(self, mode='indeterminate', length=100)

        # Coordinates label (for mouse position)
        self.coords_label = ttk.Label(self, text="", anchor=tk.E, width=30)
        self.coords_label.pack(side=tk.RIGHT, padx=5)

    def set_message(self, message):
        """Set status message.

        Parameters
        ----------
        message : str
            Status message to display.
        """
        self.message_label.config(text=message)

    def set_coordinates(self, x, y):
        """Set coordinate display.

        Parameters
        ----------
        x : float
            X coordinate.
        y : float
            Y coordinate.
        """
        self.coords_label.config(text=f"X: {x:.3f}, Y: {y:.3f}")

    def clear_coordinates(self):
        """Clear coordinate display."""
        self.coords_label.config(text="")

    def show_progress(self):
        """Show progress indicator."""
        self.progress.pack(side=tk.RIGHT, padx=5)
        self.progress.start(10)

    def hide_progress(self):
        """Hide progress indicator."""
        self.progress.stop()
        self.progress.pack_forget()
