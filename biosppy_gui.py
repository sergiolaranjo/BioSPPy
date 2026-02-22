#!/usr/bin/env python
"""
BioSPPy GUI Launcher
====================

Launch the BioSPPy graphical user interface.

Usage:
    python biosppy_gui.py

Or make it executable and run directly:
    chmod +x biosppy_gui.py
    ./biosppy_gui.py
"""

if __name__ == '__main__':
    from biosppy.gui import run_gui
    run_gui()
