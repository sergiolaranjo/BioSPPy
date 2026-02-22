# BioSPPy GUI

Professional graphical user interface for BioSPPy biosignal processing.

## Features

### 🎯 Core Functionality
- **Signal Import**: Support for multiple formats (TXT, EDF, HDF5, CSV)
- **Interactive Visualization**: Pan, zoom, and select signal regions with mouse
- **Signal Processing**: Process ECG, EDA, EMG, EEG, PPG, Respiration, and more
- **Feature Extraction**: Time, frequency, and non-linear domain features
- **Export Results**: Save processed signals and analysis results

### 🖱️ Interactive Features
- **Mouse Wheel Zoom**: Zoom in/out with scroll wheel
- **Selection Mode**: Click and drag to select signal segments for analysis
- **Context Menus**: Right-click for quick access to common operations
- **Undo/Redo**: Full undo/redo support for all operations

### 📊 Menu Bar
- **File**: Import/Export signals, save/load sessions
- **Edit**: Undo/Redo, preferences
- **Process**: Signal-specific processing, filtering, preprocessing
- **Analyze**: Feature extraction, HRV analysis, signal quality assessment
- **View**: Zoom controls, display options
- **Tools**: Selection mode, batch processing, signal comparison
- **Help**: Documentation, keyboard shortcuts, about

### 🔧 Toolbar
Quick access buttons for:
- Open and save files
- Undo/Redo operations
- Zoom controls
- Signal type processing (ECG, EDA, EMG, etc.)
- Filtering and feature extraction

### 🔌 Plugin System
Modular architecture allows easy extension with custom:
- Signal processing algorithms
- Visualization methods
- Analysis techniques
- Import/Export formats

## Usage

### Launching the GUI

```bash
# From project root
python biosppy_gui.py

# Or directly via module
python -m biosppy.gui
```

### Basic Workflow

1. **Import Signal**
   - File → Import Signal → Select format
   - Choose file and configure parameters
   - Signal appears in the signal list

2. **Process Signal**
   - Select signal from list
   - Click appropriate processing button (ECG, EDA, etc.)
   - Or: Process → Process as [Signal Type]
   - Configure parameters and click Process

3. **Analyze Results**
   - View processed signal with detected features
   - Use selection mode to analyze specific regions
   - Extract features via Analyze menu

4. **Export**
   - File → Export → Choose format
   - Save processed signals or analysis results

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+O` | Import signal |
| `Ctrl+S` | Save signal |
| `Ctrl+Q` | Exit application |
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Delete` | Remove selected signal |
| `F5` | Refresh view |
| `F1` | Show help |

### Mouse Controls

- **Scroll Wheel**: Zoom in/out at cursor position
- **Toolbar Pan/Zoom**: Use matplotlib navigation toolbar
- **Right-Click**: Open context menu
- **Click and Drag** (in selection mode): Select signal region

## Signal Types Supported

| Signal | Description | Module |
|--------|-------------|--------|
| ECG | Electrocardiogram | `biosppy.signals.ecg` |
| EDA/GSR | Electrodermal Activity | `biosppy.signals.eda` |
| EMG | Electromyogram | `biosppy.signals.emg` |
| EEG | Electroencephalogram | `biosppy.signals.eeg` |
| PPG | Photoplethysmogram | `biosppy.signals.ppg` |
| Respiration | Respiratory signal | `biosppy.signals.resp` |
| BVP | Blood Volume Pulse | `biosppy.signals.bvp` |
| ABP | Arterial Blood Pressure | `biosppy.signals.abp` |
| PCG | Phonocardiogram | `biosppy.signals.pcg` |
| Accelerometer | 3-axis acceleration | `biosppy.signals.acc` |

## File Format Support

### Import Formats
- **Text (.txt)**: Plain text files with signal data
- **EDF (.edf)**: European Data Format for biosignals
- **HDF5 (.h5)**: Hierarchical data format
- **CSV (.csv)**: Comma-separated values

### Export Formats
- **Text (.txt)**: Save processed signals
- **JSON (.json)**: Export analysis results
- **PNG/PDF**: Export figures

## Extending with Plugins

### Creating a Plugin

Create a Python file in `~/.biosppy/plugins/` or `biosppy/gui/plugins/`:

```python
# my_custom_plugin.py

plugin_info = {
    'name': 'My Custom Plugin',
    'version': '1.0.0',
    'author': 'Your Name',
    'description': 'Description of plugin functionality',
    'category': 'processing'  # or 'visualization', 'analysis', etc.
}

def register(main_window):
    """Called when plugin loads."""
    # Add menu item
    main_window.menubar.process_menu.add_command(
        label="My Custom Function",
        command=lambda: my_custom_function(main_window)
    )

def unregister(main_window):
    """Called when plugin unloads."""
    pass

def my_custom_function(main_window):
    """Your custom functionality."""
    # Access selected signal
    signal_data = main_window.signal_manager.get_signal(
        main_window.signal_listbox.get(tk.ACTIVE)
    )

    # Process signal
    processed = process_signal(signal_data['signal'])

    # Update display
    main_window.signal_manager.update_signal(
        name,
        processed,
        processing_step='My Custom Processing'
    )
    main_window.refresh_plot()
```

### Plugin Categories
- **processing**: Signal processing algorithms
- **visualization**: Custom plot types
- **analysis**: Analysis and feature extraction
- **import**: Custom file importers
- **export**: Custom file exporters

## Architecture

### Module Structure

```
biosppy/gui/
├── __init__.py           # Main entry point
├── main_window.py        # Main application window
├── menubar.py            # Menu bar implementation
├── toolbar.py            # Toolbar with quick access buttons
├── plot_widget.py        # Interactive matplotlib plotting
├── signal_manager.py     # Signal management with undo/redo
├── status_bar.py         # Status bar for messages
├── context_menu.py       # Right-click context menus
├── dialogs.py            # Various dialog windows
├── plugins.py            # Plugin system
└── README.md             # This file
```

### Key Classes

- **BioSPPyGUI**: Main application window
- **SignalManager**: Manages loaded signals with undo/redo
- **PlotWidget**: Interactive matplotlib-based plotting
- **MenuBar**: Complete menu system
- **Toolbar**: Quick access toolbar
- **PluginManager**: Plugin loading and management

## Requirements

- Python >= 3.7
- numpy
- matplotlib
- tkinter (usually included with Python)
- biosppy (all signal processing modules)

## Examples

### Example 1: Load and Process ECG

```python
from biosppy.gui import run_gui
run_gui()

# In GUI:
# 1. File → Import Signal → Text File
# 2. Select examples/ecg.txt
# 3. Click "ECG" button in toolbar
# 4. View processed signal with R-peaks
```

### Example 2: Analyze Selection

```python
# In GUI:
# 1. Load a signal
# 2. Enable selection mode (toolbar or Tools menu)
# 3. Click and drag to select region
# 4. Right-click → Analyze Selection
# 5. View results for selected segment
```

### Example 3: Batch Processing

```python
# In GUI:
# 1. Tools → Batch Processing
# 2. Add multiple files
# 3. Configure processing parameters
# 4. Process all files automatically
```

## Troubleshooting

### GUI doesn't start
- Ensure tkinter is installed: `python -m tkinter`
- Check matplotlib backend: Should use TkAgg

### Import errors
- Verify BioSPPy is installed: `pip install biosppy`
- Check file format compatibility

### Plot not responding
- Try refreshing: Press F5
- Reset view: View → Reset View

## Future Enhancements

- [ ] Real-time signal acquisition
- [ ] Multi-signal comparison view
- [ ] Advanced annotation tools
- [ ] Signal quality metrics dashboard
- [ ] Machine learning integration
- [ ] Cloud storage integration
- [ ] Collaborative features

## Contributing

To contribute to the GUI:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

See CONTRIBUTING.md for detailed guidelines.

## License

BioSPPy GUI is part of the BioSPPy project.
See LICENSE file for details.

## Contact

- GitHub: https://github.com/scientisst/BioSPPy
- Issues: https://github.com/scientisst/BioSPPy/issues

## Acknowledgments

BioSPPy GUI builds upon the excellent BioSPPy signal processing library
and uses matplotlib for visualization and tkinter for the GUI framework.
