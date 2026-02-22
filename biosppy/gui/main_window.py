"""
Main GUI Window for BioSPPy
============================

Professional interface with menu bar, toolbar, and interactive plotting.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from .menubar import MenuBar
from .toolbar import Toolbar
from .plot_widget import PlotWidget
from .signal_manager import SignalManager
from .status_bar import StatusBar


class BioSPPyGUI:
    """Main BioSPPy GUI Application.

    Features:
    - Menu bar for file operations and signal processing
    - Toolbar with quick access buttons
    - Interactive plotting with zoom, pan, and selection
    - Context menu (right-click) on plots
    - Modular architecture for easy extension
    """

    def __init__(self, root):
        """Initialize the main application window.

        Parameters
        ----------
        root : tk.Tk
            The root Tkinter window.
        """
        self.root = root
        self.root.title("BioSPPy - Biosignal Processing in Python")
        self.root.geometry("1400x900")

        # Set application icon if available
        try:
            # self.root.iconbitmap('icon.ico')  # Add icon if available
            pass
        except:
            pass

        # Initialize signal manager
        self.signal_manager = SignalManager()

        # Create menu bar
        self.menubar = MenuBar(self.root, self)

        # Create toolbar
        self.toolbar = Toolbar(self.root, self)
        self.toolbar.pack(side=tk.TOP, fill=tk.X, padx=2, pady=2)

        # Create main content area with splitter
        self._create_main_content()

        # Create status bar
        self.statusbar = StatusBar(self.root)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Keyboard shortcuts
        self._setup_shortcuts()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Update status
        self.statusbar.set_message("Ready")

    def _create_main_content(self):
        """Create the main content area with signal list and plot."""
        # Main paned window (horizontal split)
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Left panel - Signal list and properties
        self.left_panel = ttk.Frame(self.main_paned, width=250)
        self.main_paned.add(self.left_panel, weight=0)

        # Signal list
        list_label = ttk.Label(self.left_panel, text="Loaded Signals:",
                              font=("Arial", 10, "bold"))
        list_label.pack(pady=(5, 2), padx=5, anchor=tk.W)

        # Listbox with scrollbar
        list_frame = ttk.Frame(self.left_panel)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.signal_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                        selectmode=tk.SINGLE)
        self.signal_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.signal_listbox.yview)

        self.signal_listbox.bind('<<ListboxSelect>>', self.on_signal_select)
        self.signal_listbox.bind('<Button-3>', self.show_signal_context_menu)

        # Signal properties
        props_label = ttk.Label(self.left_panel, text="Signal Properties:",
                               font=("Arial", 10, "bold"))
        props_label.pack(pady=(10, 2), padx=5, anchor=tk.W)

        props_frame = ttk.Frame(self.left_panel)
        props_frame.pack(fill=tk.BOTH, padx=5, pady=5)

        self.props_text = tk.Text(props_frame, height=10, width=30,
                                 state=tk.DISABLED, wrap=tk.WORD)
        self.props_text.pack(fill=tk.BOTH, expand=True)

        # Right panel - Plot area with tabs
        self.right_panel = ttk.Frame(self.main_paned)
        self.main_paned.add(self.right_panel, weight=1)

        # Notebook for multiple plots
        self.notebook = ttk.Notebook(self.right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create initial plot tab
        self.plot_widget = PlotWidget(self.notebook, self)
        self.notebook.add(self.plot_widget.frame, text="Signal View")

        # Bind right-click on plot
        self.plot_widget.canvas.mpl_connect('button_press_event',
                                           self._on_plot_click)

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        self.root.bind('<Control-o>', lambda e: self.menubar.import_signal())
        self.root.bind('<Control-s>', lambda e: self.menubar.save_signal())
        self.root.bind('<Control-q>', lambda e: self.on_close())
        self.root.bind('<Control-z>', lambda e: self.undo())
        self.root.bind('<Control-y>', lambda e: self.redo())
        self.root.bind('<Delete>', lambda e: self.remove_signal())
        self.root.bind('<F1>', lambda e: self.menubar.show_help())
        self.root.bind('<F5>', lambda e: self.refresh_plot())

    def _on_plot_click(self, event):
        """Handle mouse clicks on plot."""
        if event.button == 3:  # Right click
            self.show_plot_context_menu(event)

    def show_plot_context_menu(self, event):
        """Show context menu on plot right-click."""
        from .context_menu import PlotContextMenu
        menu = PlotContextMenu(self, event)
        try:
            menu.post(event.guiEvent.x_root, event.guiEvent.y_root)
        finally:
            menu.grab_release()

    def show_signal_context_menu(self, event):
        """Show context menu on signal list right-click."""
        from .context_menu import SignalContextMenu
        menu = SignalContextMenu(self, event)
        try:
            menu.post(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def on_signal_select(self, event):
        """Handle signal selection from list."""
        selection = self.signal_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        signal_name = self.signal_listbox.get(idx)

        # Update properties display
        self.update_signal_properties(signal_name)

        # Update plot
        self.plot_signal(signal_name)

    def update_signal_properties(self, signal_name):
        """Update the signal properties display."""
        signal_data = self.signal_manager.get_signal(signal_name)
        if signal_data is None:
            return

        props = []
        props.append(f"Name: {signal_name}")
        props.append(f"Type: {signal_data.get('type', 'Unknown')}")
        props.append(f"Length: {len(signal_data['signal'])} samples")
        props.append(f"Sampling Rate: {signal_data.get('sampling_rate', 'N/A')} Hz")

        duration = len(signal_data['signal']) / signal_data.get('sampling_rate', 1)
        props.append(f"Duration: {duration:.2f} s")

        if 'units' in signal_data:
            props.append(f"Units: {signal_data['units']}")

        if signal_data.get('processed', False):
            props.append("\nProcessing Applied:")
            for proc in signal_data.get('processing_history', []):
                props.append(f"  - {proc}")

        # Update text widget
        self.props_text.config(state=tk.NORMAL)
        self.props_text.delete(1.0, tk.END)
        self.props_text.insert(1.0, "\n".join(props))
        self.props_text.config(state=tk.DISABLED)

    def plot_signal(self, signal_name):
        """Plot the selected signal."""
        signal_data = self.signal_manager.get_signal(signal_name)
        if signal_data is None:
            return

        self.plot_widget.plot_signal(signal_data)
        self.statusbar.set_message(f"Displaying: {signal_name}")

    def add_signal_to_list(self, signal_name):
        """Add a signal to the listbox."""
        self.signal_listbox.insert(tk.END, signal_name)

    def remove_signal(self):
        """Remove selected signal."""
        selection = self.signal_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        signal_name = self.signal_listbox.get(idx)

        # Confirm deletion
        if messagebox.askyesno("Confirm", f"Remove signal '{signal_name}'?"):
            self.signal_manager.remove_signal(signal_name)
            self.signal_listbox.delete(idx)
            self.plot_widget.clear()
            self.statusbar.set_message(f"Removed: {signal_name}")

    def refresh_plot(self):
        """Refresh the current plot."""
        selection = self.signal_listbox.curselection()
        if selection:
            idx = selection[0]
            signal_name = self.signal_listbox.get(idx)
            self.plot_signal(signal_name)

    def undo(self):
        """Undo last operation."""
        if self.signal_manager.undo():
            self.statusbar.set_message("Undo successful")
            self.refresh_plot()
        else:
            self.statusbar.set_message("Nothing to undo")

    def redo(self):
        """Redo last undone operation."""
        if self.signal_manager.redo():
            self.statusbar.set_message("Redo successful")
            self.refresh_plot()
        else:
            self.statusbar.set_message("Nothing to redo")

    def on_close(self):
        """Handle window close event."""
        if self.signal_manager.has_unsaved_changes():
            if messagebox.askyesno("Confirm Exit",
                                  "You have unsaved changes. Exit anyway?"):
                self.root.destroy()
        else:
            self.root.destroy()
