"""
Interactive Plot Widget
=======================

Matplotlib-based interactive plotting with zoom, pan, and selection.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class PlotWidget:
    """Interactive plot widget with zoom, pan, and segment selection.

    Features:
    - Mouse wheel zoom
    - Pan and zoom with matplotlib toolbar
    - Segment selection with mouse drag
    - Multiple subplots support
    - Annotation tools
    """

    def __init__(self, parent, main_window):
        """Initialize plot widget.

        Parameters
        ----------
        parent : tk.Widget
            Parent widget.
        main_window : BioSPPyGUI
            Reference to main window.
        """
        self.parent = parent
        self.main_window = main_window

        # Create frame
        self.frame = ttk.Frame(parent)

        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add navigation toolbar
        toolbar_frame = ttk.Frame(self.frame)
        toolbar_frame.pack(fill=tk.X)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # Selection variables
        self.selection_active = False
        self.selection_start = None
        self.selection_end = None
        self.selection_rect = None

        # Current signal data
        self.current_signal = None
        self.current_axes = []

        # Connect mouse events for selection
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

        # Selection mode control
        self.selection_mode = False

    def enable_selection_mode(self, enable=True):
        """Enable/disable selection mode.

        Parameters
        ----------
        enable : bool
            True to enable selection mode.
        """
        self.selection_mode = enable
        if enable:
            self.main_window.statusbar.set_message("Selection mode: Click and drag to select region")
        else:
            self.main_window.statusbar.set_message("Ready")

    def plot_signal(self, signal_data, show_results=True):
        """Plot signal data.

        Parameters
        ----------
        signal_data : dict
            Signal data dictionary with 'signal', 'sampling_rate', etc.
        show_results : bool
            Whether to show processing results if available.
        """
        self.current_signal = signal_data
        self.figure.clear()

        signal = signal_data['signal']
        sampling_rate = signal_data.get('sampling_rate', 1000)
        signal_type = signal_data.get('type', 'Signal')
        units = signal_data.get('units', 'mV')

        # Create time axis
        ts = np.arange(len(signal)) / sampling_rate

        # Check if we have processing results
        results = signal_data.get('results', {})

        if results and show_results:
            # Create subplots based on results
            n_plots = self._count_result_plots(results)
            self.current_axes = []

            # Main signal plot
            ax1 = self.figure.add_subplot(n_plots, 1, 1)
            ax1.plot(ts, signal, 'b-', linewidth=0.5, label='Raw Signal')

            # Plot filtered signal if available
            if 'filtered' in results:
                ax1.plot(ts, results['filtered'], 'g-', linewidth=0.5,
                        label='Filtered', alpha=0.7)

            # Plot detected features (e.g., R-peaks for ECG)
            if 'peaks' in results or 'rpeaks' in results:
                peaks = results.get('peaks', results.get('rpeaks', []))
                ax1.plot(ts[peaks], signal[peaks], 'ro', markersize=4,
                        label='Detected Peaks')

            ax1.set_ylabel(f'{signal_type} ({units})')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            self.current_axes.append(ax1)

            # Additional plots based on signal type
            plot_idx = 2

            # Heart rate or instantaneous rate
            if 'heart_rate' in results or 'inst_rate' in results:
                ax = self.figure.add_subplot(n_plots, 1, plot_idx)
                rate = results.get('heart_rate', results.get('inst_rate', []))
                rate_ts = results.get('heart_rate_ts', ts[:len(rate)])
                ax.plot(rate_ts, rate, 'r-', linewidth=1)
                ax.set_ylabel('Rate (bpm)')
                ax.grid(True, alpha=0.3)
                self.current_axes.append(ax)
                plot_idx += 1

            # Templates
            if 'templates' in results:
                ax = self.figure.add_subplot(n_plots, 1, plot_idx)
                templates = results['templates']
                for template in templates:
                    ax.plot(template, alpha=0.5, linewidth=0.5)
                ax.set_ylabel('Templates')
                ax.grid(True, alpha=0.3)
                self.current_axes.append(ax)
                plot_idx += 1

            # Set x-label on last plot
            if self.current_axes:
                self.current_axes[-1].set_xlabel('Time (s)')

        else:
            # Simple single plot
            ax = self.figure.add_subplot(111)
            ax.plot(ts, signal, 'b-', linewidth=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'{signal_type} ({units})')
            ax.set_title(f'{signal_type} Signal')
            ax.grid(True, alpha=0.3)
            self.current_axes = [ax]

        self.figure.tight_layout()
        self.canvas.draw()

    def _count_result_plots(self, results):
        """Count how many subplots needed for results."""
        count = 1  # Always have main signal
        if 'heart_rate' in results or 'inst_rate' in results:
            count += 1
        if 'templates' in results:
            count += 1
        return count

    def _on_press(self, event):
        """Handle mouse button press."""
        if not self.selection_mode or event.inaxes is None:
            return

        self.selection_active = True
        self.selection_start = event.xdata

    def _on_release(self, event):
        """Handle mouse button release."""
        if not self.selection_active or event.inaxes is None:
            return

        self.selection_active = False
        self.selection_end = event.xdata

        if self.selection_start is not None and self.selection_end is not None:
            # Ensure start < end
            start = min(self.selection_start, self.selection_end)
            end = max(self.selection_start, self.selection_end)

            self._process_selection(start, end)

        # Clear selection rectangle
        if self.selection_rect is not None:
            self.selection_rect.remove()
            self.selection_rect = None
            self.canvas.draw()

    def _on_motion(self, event):
        """Handle mouse motion."""
        if not self.selection_active or event.inaxes is None:
            return

        # Draw selection rectangle
        if self.selection_rect is not None:
            self.selection_rect.remove()

        ax = event.inaxes
        y_min, y_max = ax.get_ylim()

        self.selection_rect = ax.axvspan(self.selection_start, event.xdata,
                                         alpha=0.3, color='yellow')
        self.canvas.draw()

    def _on_scroll(self, event):
        """Handle mouse scroll for zoom."""
        if event.inaxes is None:
            return

        # Get current axis limits
        ax = event.inaxes
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        # Zoom factor
        zoom_factor = 1.2 if event.button == 'down' else 0.8

        # Get mouse position in data coordinates
        xdata = event.xdata
        ydata = event.ydata

        # Calculate new limits
        new_xmin = xdata - (xdata - xmin) * zoom_factor
        new_xmax = xdata + (xmax - xdata) * zoom_factor
        new_ymin = ydata - (ydata - ymin) * zoom_factor
        new_ymax = ydata + (ymax - ydata) * zoom_factor

        # Apply to all axes
        for ax in self.current_axes:
            ax.set_xlim(new_xmin, new_xmax)
            ax.set_ylim(new_ymin, new_ymax)

        self.canvas.draw()

    def _process_selection(self, start_time, end_time):
        """Process selected time segment.

        Parameters
        ----------
        start_time : float
            Start time in seconds.
        end_time : float
            End time in seconds.
        """
        if self.current_signal is None:
            return

        sampling_rate = self.current_signal.get('sampling_rate', 1000)
        signal = self.current_signal['signal']

        # Convert time to indices
        start_idx = int(start_time * sampling_rate)
        end_idx = int(end_time * sampling_rate)

        # Clip to valid range
        start_idx = max(0, start_idx)
        end_idx = min(len(signal), end_idx)

        # Store selection
        self.current_signal['selection'] = {
            'start_time': start_time,
            'end_time': end_time,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'segment': signal[start_idx:end_idx]
        }

        # Notify main window
        msg = f"Selected: {start_time:.2f}s to {end_time:.2f}s ({end_idx-start_idx} samples)"
        self.main_window.statusbar.set_message(msg)

        # Highlight selection
        for ax in self.current_axes:
            ax.axvspan(start_time, end_time, alpha=0.2, color='green')

        self.canvas.draw()

    def clear(self):
        """Clear the plot."""
        self.figure.clear()
        self.canvas.draw()
        self.current_signal = None
        self.current_axes = []

    def save_figure(self, filename):
        """Save current figure to file.

        Parameters
        ----------
        filename : str
            Output filename.
        """
        self.figure.savefig(filename, dpi=300, bbox_inches='tight')

    def get_selection(self):
        """Get current selection.

        Returns
        -------
        dict or None
            Selection information or None if no selection.
        """
        if self.current_signal is None:
            return None
        return self.current_signal.get('selection')
