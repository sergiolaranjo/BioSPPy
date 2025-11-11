"""
Signal Manager
==============

Manages loaded signals with undo/redo support.
"""

import numpy as np
from copy import deepcopy


class SignalManager:
    """Manages biosignals with undo/redo functionality."""

    def __init__(self):
        """Initialize signal manager."""
        self.signals = {}  # signal_name -> signal_data
        self.history = []  # For undo/redo
        self.history_index = -1
        self.unsaved_changes = False

    def add_signal(self, name, signal, signal_type='unknown', sampling_rate=1000,
                  units='mV', **kwargs):
        """Add a new signal.

        Parameters
        ----------
        name : str
            Signal name/identifier.
        signal : array
            Signal data.
        signal_type : str
            Type of signal (ECG, EDA, EMG, etc.).
        sampling_rate : float
            Sampling rate in Hz.
        units : str
            Signal units.
        **kwargs : dict
            Additional metadata.
        """
        signal_data = {
            'signal': np.array(signal),
            'type': signal_type,
            'sampling_rate': sampling_rate,
            'units': units,
            'processed': False,
            'processing_history': [],
            **kwargs
        }

        self.signals[name] = signal_data
        self._save_state()
        self.unsaved_changes = True

    def get_signal(self, name):
        """Get signal data by name.

        Parameters
        ----------
        name : str
            Signal name.

        Returns
        -------
        dict or None
            Signal data dictionary or None if not found.
        """
        return self.signals.get(name)

    def get_all_signals(self):
        """Get all signal names.

        Returns
        -------
        list
            List of signal names.
        """
        return list(self.signals.keys())

    def remove_signal(self, name):
        """Remove a signal.

        Parameters
        ----------
        name : str
            Signal name to remove.
        """
        if name in self.signals:
            del self.signals[name]
            self._save_state()
            self.unsaved_changes = True

    def update_signal(self, name, signal, processing_step=None):
        """Update signal data (for processing operations).

        Parameters
        ----------
        name : str
            Signal name.
        signal : array
            New signal data.
        processing_step : str, optional
            Description of processing applied.
        """
        if name in self.signals:
            self.signals[name]['signal'] = np.array(signal)
            self.signals[name]['processed'] = True

            if processing_step:
                self.signals[name]['processing_history'].append(processing_step)

            self._save_state()
            self.unsaved_changes = True

    def add_processing_results(self, name, results):
        """Add processing results to signal data.

        Parameters
        ----------
        name : str
            Signal name.
        results : dict
            Processing results (e.g., peaks, heart_rate, templates).
        """
        if name in self.signals:
            self.signals[name]['results'] = results
            self.unsaved_changes = True

    def _save_state(self):
        """Save current state for undo/redo."""
        # Remove any states after current index
        self.history = self.history[:self.history_index + 1]

        # Save current state
        self.history.append(deepcopy(self.signals))
        self.history_index += 1

        # Limit history size
        if len(self.history) > 50:
            self.history.pop(0)
            self.history_index -= 1

    def undo(self):
        """Undo last operation.

        Returns
        -------
        bool
            True if undo was successful.
        """
        if self.history_index > 0:
            self.history_index -= 1
            self.signals = deepcopy(self.history[self.history_index])
            return True
        return False

    def redo(self):
        """Redo last undone operation.

        Returns
        -------
        bool
            True if redo was successful.
        """
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.signals = deepcopy(self.history[self.history_index])
            return True
        return False

    def has_unsaved_changes(self):
        """Check if there are unsaved changes.

        Returns
        -------
        bool
            True if there are unsaved changes.
        """
        return self.unsaved_changes

    def mark_saved(self):
        """Mark all changes as saved."""
        self.unsaved_changes = False

    def clear(self):
        """Clear all signals."""
        self.signals.clear()
        self.history.clear()
        self.history_index = -1
        self.unsaved_changes = False
