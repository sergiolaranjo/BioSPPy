"""
BioSPPy GUI Module
==================

Professional GUI interface for BioSPPy signal processing.

This module provides a complete interactive application for:
- Importing and managing biosignals
- Preprocessing and filtering
- Signal analysis and feature extraction
- Interactive visualization with zoom, pan, and segment selection
- Modular architecture for easy extension

Usage:
    from biosppy.gui import run_gui
    run_gui()
"""

__all__ = ['run_gui']

def run_gui():
    """Launch the BioSPPy GUI application."""
    from .main_window import BioSPPyGUI
    import tkinter as tk

    root = tk.Tk()
    app = BioSPPyGUI(root)
    root.mainloop()
