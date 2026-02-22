"""
Dialogs
=======

Various dialog windows for configuration and user input.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import os


class ImportDialog(tk.Toplevel):
    """Dialog for importing signals."""

    def __init__(self, parent, main_window, file_format='auto'):
        """Initialize import dialog.

        Parameters
        ----------
        parent : tk.Widget
            Parent window.
        main_window : BioSPPyGUI
            Reference to main window.
        file_format : str
            File format to import.
        """
        super().__init__(parent)
        self.main_window = main_window
        self.file_format = file_format

        self.title("Import Signal")
        self.geometry("500x400")

        self._create_widgets()
        self._select_file()

    def _create_widgets(self):
        """Create dialog widgets."""
        # File selection
        file_frame = ttk.LabelFrame(self, text="File Selection", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(file_frame, text="File:").grid(row=0, column=0, sticky=tk.W)
        self.file_entry = ttk.Entry(file_frame, width=40)
        self.file_entry.grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse...",
                  command=self._select_file).grid(row=0, column=2)

        # Signal properties
        props_frame = ttk.LabelFrame(self, text="Signal Properties", padding=10)
        props_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Signal name
        ttk.Label(props_frame, text="Signal Name:").grid(row=0, column=0,
                                                         sticky=tk.W, pady=2)
        self.name_entry = ttk.Entry(props_frame, width=30)
        self.name_entry.grid(row=0, column=1, padx=5, pady=2)

        # Signal type
        ttk.Label(props_frame, text="Signal Type:").grid(row=1, column=0,
                                                         sticky=tk.W, pady=2)
        self.type_combo = ttk.Combobox(props_frame, width=28,
                                       values=['ECG', 'EDA', 'EMG', 'EEG',
                                             'PPG', 'Respiration', 'BVP',
                                             'ABP', 'Accelerometer', 'Other'])
        self.type_combo.current(0)
        self.type_combo.grid(row=1, column=1, padx=5, pady=2)

        # Sampling rate
        ttk.Label(props_frame, text="Sampling Rate (Hz):").grid(row=2, column=0,
                                                                sticky=tk.W, pady=2)
        self.sampling_rate_entry = ttk.Entry(props_frame, width=30)
        self.sampling_rate_entry.insert(0, "1000")
        self.sampling_rate_entry.grid(row=2, column=1, padx=5, pady=2)

        # Units
        ttk.Label(props_frame, text="Units:").grid(row=3, column=0,
                                                   sticky=tk.W, pady=2)
        self.units_entry = ttk.Entry(props_frame, width=30)
        self.units_entry.insert(0, "mV")
        self.units_entry.grid(row=3, column=1, padx=5, pady=2)

        # Format-specific options
        if self.file_format in ['txt', 'csv', 'auto']:
            ttk.Label(props_frame, text="Column/Channel:").grid(row=4, column=0,
                                                                sticky=tk.W, pady=2)
            self.column_entry = ttk.Entry(props_frame, width=30)
            self.column_entry.insert(0, "0")
            self.column_entry.grid(row=4, column=1, padx=5, pady=2)

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="Import",
                  command=self._import).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.RIGHT)

    def _select_file(self):
        """Open file selection dialog."""
        filetypes = [
            ("All supported", "*.txt *.edf *.h5 *.csv"),
            ("Text files", "*.txt"),
            ("EDF files", "*.edf"),
            ("HDF5 files", "*.h5"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Select Signal File",
            filetypes=filetypes
        )

        if filename:
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, filename)

            # Auto-fill signal name from filename
            base_name = os.path.splitext(os.path.basename(filename))[0]
            self.name_entry.delete(0, tk.END)
            self.name_entry.insert(0, base_name)

    def _import(self):
        """Import the signal."""
        filename = self.file_entry.get()
        if not filename:
            messagebox.showerror("Error", "Please select a file")
            return

        signal_name = self.name_entry.get()
        if not signal_name:
            messagebox.showerror("Error", "Please enter a signal name")
            return

        try:
            sampling_rate = float(self.sampling_rate_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid sampling rate")
            return

        signal_type = self.type_combo.get().lower()
        units = self.units_entry.get()

        # Import signal based on file format
        try:
            from biosppy import storage

            ext = os.path.splitext(filename)[1].lower()

            if ext in ['.txt', '.csv']:
                signal, mdata = storage.load_txt(filename)
                # Handle multi-column files
                if hasattr(self, 'column_entry'):
                    try:
                        col = int(self.column_entry.get())
                        if signal.ndim > 1:
                            signal = signal[:, col]
                    except:
                        pass

            elif ext == '.edf':
                signals, mdata = storage.load_edf(filename)
                # For simplicity, take first channel
                # In a full implementation, let user select channel
                signal = signals[0] if isinstance(signals, list) else signals

            elif ext == '.h5':
                with storage.HDF(filename, mode='r') as f:
                    # List available signals
                    # For now, take first one
                    # In full implementation, let user select
                    signal, mdata = f.get_signal(name=signal_name)

            else:
                messagebox.showerror("Error", f"Unsupported file format: {ext}")
                return

            # Add to signal manager
            self.main_window.signal_manager.add_signal(
                signal_name,
                signal,
                signal_type=signal_type,
                sampling_rate=sampling_rate,
                units=units,
                filename=filename
            )

            # Add to listbox
            self.main_window.add_signal_to_list(signal_name)

            # Update status
            self.main_window.statusbar.set_message(
                f"Imported: {signal_name} ({len(signal)} samples)"
            )

            self.destroy()

        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import signal:\n{str(e)}")


class ProcessDialog(tk.Toplevel):
    """Dialog for signal processing."""

    def __init__(self, parent, main_window, signal_type):
        """Initialize process dialog.

        Parameters
        ----------
        parent : tk.Widget
            Parent window.
        main_window : BioSPPyGUI
            Reference to main window.
        signal_type : str
            Type of signal processing (ecg, eda, emg, etc.).
        """
        super().__init__(parent)
        self.main_window = main_window
        self.signal_type = signal_type

        self.title(f"Process Signal - {signal_type.upper()}")
        self.geometry("500x400")

        self._create_widgets()

    def _create_widgets(self):
        """Create dialog widgets."""
        # Get selected signal
        selection = self.main_window.signal_listbox.curselection()
        if not selection:
            messagebox.showerror("Error", "No signal selected")
            self.destroy()
            return

        idx = selection[0]
        self.signal_name = self.main_window.signal_listbox.get(idx)

        # Parameters frame
        params_frame = ttk.LabelFrame(self, text="Processing Parameters", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Common parameters
        row = 0

        # Show plot
        self.show_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Show plot",
                       variable=self.show_var).grid(row=row, column=0,
                                                   columnspan=2, sticky=tk.W, pady=2)
        row += 1

        # Interactive
        self.interactive_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(params_frame, text="Interactive mode",
                       variable=self.interactive_var).grid(row=row, column=0,
                                                          columnspan=2, sticky=tk.W, pady=2)
        row += 1

        # Signal-specific parameters
        if self.signal_type == 'ecg':
            ttk.Label(params_frame, text="R-peak detector:").grid(row=row, column=0,
                                                                  sticky=tk.W, pady=2)
            self.detector_combo = ttk.Combobox(params_frame, width=20,
                                              values=['hamilton', 'christov',
                                                     'engzee', 'gamboa'])
            self.detector_combo.current(0)
            self.detector_combo.grid(row=row, column=1, padx=5, pady=2)
            row += 1

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="Process",
                  command=self._process).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.RIGHT)

    def _process(self):
        """Process the signal."""
        signal_data = self.main_window.signal_manager.get_signal(self.signal_name)
        if signal_data is None:
            return

        try:
            # Import appropriate processing module
            if self.signal_type == 'ecg':
                from biosppy.signals import ecg as sig_module
                kwargs = {}
                if hasattr(self, 'detector_combo'):
                    kwargs['segmenter'] = self.detector_combo.get()

            elif self.signal_type == 'eda':
                from biosppy.signals import eda as sig_module
                kwargs = {}

            elif self.signal_type == 'emg':
                from biosppy.signals import emg as sig_module
                kwargs = {}

            elif self.signal_type == 'eeg':
                from biosppy.signals import eeg as sig_module
                kwargs = {}

            elif self.signal_type == 'ppg':
                from biosppy.signals import ppg as sig_module
                kwargs = {}

            elif self.signal_type == 'resp':
                from biosppy.signals import resp as sig_module
                kwargs = {}

            else:
                messagebox.showerror("Error", f"Unknown signal type: {self.signal_type}")
                return

            # Show progress
            self.main_window.statusbar.show_progress()
            self.main_window.statusbar.set_message(f"Processing {self.signal_type.upper()}...")

            # Process signal
            result = getattr(sig_module, self.signal_type)(
                signal=signal_data['signal'],
                sampling_rate=signal_data['sampling_rate'],
                show=False,  # We'll plot in our GUI
                **kwargs
            )

            # Store results
            results_dict = {
                'ts': result['ts'],
            }

            # Add signal-specific results
            if self.signal_type == 'ecg':
                results_dict.update({
                    'filtered': result['filtered'],
                    'rpeaks': result['rpeaks'],
                    'templates_ts': result['templates_ts'],
                    'templates': result['templates'],
                    'heart_rate_ts': result['heart_rate_ts'],
                    'heart_rate': result['heart_rate'],
                })
            elif self.signal_type in ['eda', 'emg', 'ppg']:
                results_dict.update({
                    'filtered': result.get('filtered'),
                    'peaks': result.get('peaks', result.get('onsets')),
                })

            self.main_window.signal_manager.add_processing_results(
                self.signal_name,
                results_dict
            )

            # Update signal data
            signal_data['results'] = results_dict

            # Refresh plot
            self.main_window.plot_signal(self.signal_name)

            # Hide progress
            self.main_window.statusbar.hide_progress()
            self.main_window.statusbar.set_message(
                f"Processed {self.signal_name} as {self.signal_type.upper()}"
            )

            self.destroy()

        except Exception as e:
            self.main_window.statusbar.hide_progress()
            messagebox.showerror("Processing Error",
                               f"Failed to process signal:\n{str(e)}")


class FilterDialog(tk.Toplevel):
    """Dialog for applying filters."""

    def __init__(self, parent, main_window, filter_type):
        """Initialize filter dialog.

        Parameters
        ----------
        parent : tk.Widget
            Parent window.
        main_window : BioSPPyGUI
            Reference to main window.
        filter_type : str
            Type of filter (bandpass, lowpass, highpass, notch, smooth).
        """
        super().__init__(parent)
        self.main_window = main_window
        self.filter_type = filter_type

        self.title(f"Apply {filter_type.title()} Filter")
        self.geometry("400x300")

        self._create_widgets()

    def _create_widgets(self):
        """Create dialog widgets."""
        # Parameters frame
        params_frame = ttk.LabelFrame(self, text="Filter Parameters", padding=10)
        params_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        row = 0

        # Filter-specific parameters
        if self.filter_type in ['bandpass', 'lowpass', 'highpass']:
            if self.filter_type in ['bandpass', 'highpass']:
                ttk.Label(params_frame, text="Low Cutoff (Hz):").grid(row=row,
                                                                      column=0,
                                                                      sticky=tk.W,
                                                                      pady=2)
                self.low_cutoff_entry = ttk.Entry(params_frame, width=20)
                self.low_cutoff_entry.insert(0, "0.67")
                self.low_cutoff_entry.grid(row=row, column=1, padx=5, pady=2)
                row += 1

            if self.filter_type in ['bandpass', 'lowpass']:
                ttk.Label(params_frame, text="High Cutoff (Hz):").grid(row=row,
                                                                       column=0,
                                                                       sticky=tk.W,
                                                                       pady=2)
                self.high_cutoff_entry = ttk.Entry(params_frame, width=20)
                self.high_cutoff_entry.insert(0, "45")
                self.high_cutoff_entry.grid(row=row, column=1, padx=5, pady=2)
                row += 1

            ttk.Label(params_frame, text="Order:").grid(row=row, column=0,
                                                       sticky=tk.W, pady=2)
            self.order_entry = ttk.Entry(params_frame, width=20)
            self.order_entry.insert(0, "4")
            self.order_entry.grid(row=row, column=1, padx=5, pady=2)
            row += 1

        elif self.filter_type == 'notch':
            ttk.Label(params_frame, text="Frequency (Hz):").grid(row=row,
                                                                 column=0,
                                                                 sticky=tk.W,
                                                                 pady=2)
            self.freq_entry = ttk.Entry(params_frame, width=20)
            self.freq_entry.insert(0, "50")  # or 60 for US
            self.freq_entry.grid(row=row, column=1, padx=5, pady=2)
            row += 1

        elif self.filter_type == 'smooth':
            ttk.Label(params_frame, text="Window Size:").grid(row=row,
                                                             column=0,
                                                             sticky=tk.W,
                                                             pady=2)
            self.window_entry = ttk.Entry(params_frame, width=20)
            self.window_entry.insert(0, "5")
            self.window_entry.grid(row=row, column=1, padx=5, pady=2)
            row += 1

        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="Apply",
                  command=self._apply).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel",
                  command=self.destroy).pack(side=tk.RIGHT)

    def _apply(self):
        """Apply filter."""
        messagebox.showinfo("Filter", "Filter functionality to be fully implemented")
        self.destroy()


class AnalyzeDialog(tk.Toplevel):
    """Dialog for signal analysis."""

    def __init__(self, parent, main_window, selection_only=False):
        """Initialize analyze dialog."""
        super().__init__(parent)
        self.main_window = main_window
        self.selection_only = selection_only

        self.title("Analyze Signal")
        self.geometry("400x300")

        self._create_widgets()

    def _create_widgets(self):
        """Create dialog widgets."""
        messagebox.showinfo("Analyze", "Analysis dialog to be implemented")
        self.destroy()


class FeatureExtractionDialog(tk.Toplevel):
    """Dialog for feature extraction."""

    def __init__(self, parent, main_window, feature_type):
        """Initialize feature extraction dialog."""
        super().__init__(parent)
        self.main_window = main_window
        self.feature_type = feature_type

        self.title(f"Extract {feature_type.title()} Features")
        self.geometry("400x300")

        self._create_widgets()

    def _create_widgets(self):
        """Create dialog widgets."""
        messagebox.showinfo("Features", "Feature extraction dialog to be implemented")
        self.destroy()


class HRVDialog(tk.Toplevel):
    """Dialog for HRV analysis."""

    def __init__(self, parent, main_window):
        """Initialize HRV dialog."""
        super().__init__(parent)
        self.main_window = main_window

        self.title("HRV Analysis")
        self.geometry("500x400")

        self._create_widgets()

    def _create_widgets(self):
        """Create dialog widgets."""
        messagebox.showinfo("HRV", "HRV analysis dialog to be implemented")
        self.destroy()


class StatisticsDialog(tk.Toplevel):
    """Dialog for signal statistics."""

    def __init__(self, parent, main_window):
        """Initialize statistics dialog."""
        super().__init__(parent)
        self.main_window = main_window

        self.title("Signal Statistics")
        self.geometry("400x300")

        self._create_widgets()

    def _create_widgets(self):
        """Create dialog widgets."""
        messagebox.showinfo("Statistics", "Statistics dialog to be implemented")
        self.destroy()


class ResampleDialog(tk.Toplevel):
    """Dialog for resampling."""

    def __init__(self, parent, main_window):
        """Initialize resample dialog."""
        super().__init__(parent)
        self.main_window = main_window

        self.title("Resample Signal")
        self.geometry("400x200")

        self._create_widgets()

    def _create_widgets(self):
        """Create dialog widgets."""
        messagebox.showinfo("Resample", "Resample dialog to be implemented")
        self.destroy()


class PreferencesDialog(tk.Toplevel):
    """Dialog for application preferences."""

    def __init__(self, parent, main_window):
        """Initialize preferences dialog."""
        super().__init__(parent)
        self.main_window = main_window

        self.title("Preferences")
        self.geometry("500x400")

        self._create_widgets()

    def _create_widgets(self):
        """Create dialog widgets."""
        messagebox.showinfo("Preferences", "Preferences dialog to be implemented")
        self.destroy()


class BatchProcessingDialog(tk.Toplevel):
    """Dialog for batch processing."""

    def __init__(self, parent, main_window):
        """Initialize batch processing dialog."""
        super().__init__(parent)
        self.main_window = main_window

        self.title("Batch Processing")
        self.geometry("600x500")

        self._create_widgets()

    def _create_widgets(self):
        """Create dialog widgets."""
        messagebox.showinfo("Batch", "Batch processing dialog to be implemented")
        self.destroy()


class ExportDialog(tk.Toplevel):
    """Dialog for exporting data."""

    def __init__(self, parent, main_window, export_type, file_format):
        """Initialize export dialog."""
        super().__init__(parent)
        self.main_window = main_window
        self.export_type = export_type
        self.file_format = file_format

        self.title(f"Export {export_type.title()}")
        self.geometry("400x200")

        self._create_widgets()

    def _create_widgets(self):
        """Create dialog widgets."""
        messagebox.showinfo("Export", f"Export {self.export_type} dialog to be implemented")
        self.destroy()
