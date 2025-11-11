# BioSPPy GUI Installation Guide

## System Requirements

### Operating Systems
- Linux (Ubuntu, Debian, Fedora, etc.)
- macOS
- Windows 10/11

### Python Version
- Python 3.7 or higher

## Dependencies

### Python Packages
Install via pip:

```bash
pip install biosppy
pip install matplotlib numpy scipy
```

All dependencies are automatically installed when you install BioSPPy:
- numpy
- matplotlib
- scipy
- scikit-learn
- h5py
- bidict
- shortuuid
- joblib
- opencv-python
- pywavelets
- peakutils

### System Packages (for GUI)

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install python3-tk
```

#### Fedora/RHEL/CentOS
```bash
sudo dnf install python3-tkinter
```

#### macOS
Tkinter is usually included with Python on macOS. If not:
```bash
brew install python-tk
```

#### Windows
Tkinter is typically included with the Python installer.
If missing, reinstall Python and ensure "tcl/tk and IDLE" is checked.

## Installation

### Option 1: From PyPI (when released)
```bash
pip install biosppy[gui]
```

### Option 2: From Source
```bash
git clone https://github.com/scientisst/BioSPPy.git
cd BioSPPy
pip install -e .
```

## Verification

Verify installation:

```bash
# Test Python packages
python -c "import numpy, matplotlib, scipy; print('Packages OK')"

# Test tkinter
python -m tkinter
```

If tkinter test opens a small window, you're ready to go!

## Running the GUI

```bash
# From project directory
python biosppy_gui.py

# Or via Python module
python -m biosppy.gui

# Or from Python
python
>>> from biosppy.gui import run_gui
>>> run_gui()
```

## Troubleshooting

### "No module named 'tkinter'"

**Linux:**
```bash
sudo apt-get install python3-tk  # Ubuntu/Debian
sudo dnf install python3-tkinter  # Fedora
```

**macOS:**
```bash
brew install python-tk@3.x  # Replace x with your Python version
```

**Windows:**
Reinstall Python with tkinter support enabled.

### "No module named 'numpy'" or similar

```bash
pip install numpy matplotlib scipy
```

Or install all BioSPPy dependencies:
```bash
pip install -r requirements.txt
```

### GUI doesn't display

**Remote server (SSH):**
Enable X11 forwarding:
```bash
ssh -X user@server
```

**WSL (Windows Subsystem for Linux):**
Install an X server like VcXsrv or install VcXsrv and set DISPLAY:
```bash
export DISPLAY=:0
```

**Docker:**
GUI applications require X11 forwarding:
```bash
docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix biosppy
```

### matplotlib backend issues

If you see matplotlib backend warnings, set the backend:

```python
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
```

This is already configured in the GUI code.

## Testing Without Display

For headless environments (CI, Docker, etc.), you can test the code structure:

```bash
# Syntax check
python -m py_compile biosppy/gui/*.py

# Import test (without actually creating windows)
python -c "from biosppy.gui import signal_manager, plot_widget; print('Structure OK')"
```

## Performance Tips

### Large Files
For large signal files:
- Use HDF5 format for better performance
- Enable downsampling for visualization
- Use batch processing for multiple files

### Memory Usage
- Close unused signal tabs
- Use "Clear All" periodically
- Process signals in smaller segments

### Plotting Performance
- Reduce signal length for real-time visualization
- Disable interactive mode for batch processing
- Use PNG instead of PDF for faster exports

## Next Steps

After installation:
1. Read the [GUI README](README.md) for usage guide
2. Try the examples in `examples/` directory
3. Check out the [tutorial](TUTORIAL.md) (if available)

## Support

If you encounter issues:
1. Check this troubleshooting guide
2. Search [GitHub Issues](https://github.com/scientisst/BioSPPy/issues)
3. Create a new issue with:
   - Operating system
   - Python version (`python --version`)
   - Error message
   - Steps to reproduce

## Optional Components

### Additional Features

**EDA processing with CVXopt:**
```bash
pip install biosppy[eda]
```

**Development tools:**
```bash
pip install pytest pytest-cov black flake8
```

## Minimal Working Example

Test your installation:

```python
# test_gui.py
import sys
import os

# Add project to path if running from source
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all GUI modules can be imported."""
    try:
        from biosppy.gui import signal_manager
        from biosppy.gui import plot_widget
        from biosppy.gui import menubar
        from biosppy.gui import toolbar
        from biosppy.gui import dialogs
        from biosppy.gui import plugins
        print("✓ All GUI modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_signal_manager():
    """Test signal manager functionality."""
    try:
        from biosppy.gui.signal_manager import SignalManager
        import numpy as np

        manager = SignalManager()

        # Add a test signal
        signal = np.random.randn(1000)
        manager.add_signal('test', signal, 'ECG', 1000)

        # Retrieve signal
        data = manager.get_signal('test')
        assert data is not None
        assert len(data['signal']) == 1000

        # Test undo
        manager.remove_signal('test')
        assert manager.undo()

        print("✓ Signal manager tests passed")
        return True
    except Exception as e:
        print(f"✗ Signal manager error: {e}")
        return False

if __name__ == '__main__':
    print("Testing BioSPPy GUI installation...\n")

    results = []
    results.append(test_imports())
    results.append(test_signal_manager())

    print(f"\nResults: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("\n✓ Installation verified! You can run the GUI with:")
        print("  python biosppy_gui.py")
    else:
        print("\n✗ Some tests failed. Check the errors above.")
        sys.exit(1)
```

Run the test:
```bash
python test_gui.py
```
