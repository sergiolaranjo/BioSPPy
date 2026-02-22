# BioSPPy GUI - Quick Start Guide

Get started with the BioSPPy GUI in 5 minutes!

## Installation

```bash
# Install system dependencies (Linux)
sudo apt-get install python3-tk

# Install BioSPPy with dependencies
pip install numpy matplotlib scipy h5py peakutils

# Clone repository if not already done
git clone https://github.com/scientisst/BioSPPy.git
cd BioSPPy
```

## Launch

```bash
python biosppy_gui.py
```

## Your First Signal Analysis

### 1. Import a Signal

**Method A: Using Menu**
- File → Import Signal → Text File
- Navigate to `examples/ecg.txt`
- Configure:
  - Signal Name: `my_ecg`
  - Signal Type: `ECG`
  - Sampling Rate: `1000` Hz
  - Units: `mV`
- Click **Import**

**Method B: Using Toolbar**
- Click 📂 **Open** button
- Select file
- Configure and import

### 2. Process the Signal

**Quick Processing:**
- Click ⚡ **ECG** button in toolbar
- Click **Process** in dialog

**Or via Menu:**
- Process → Process as ECG
- Select R-peak detector (default: hamilton)
- Click **Process**

### 3. View Results

The plot will show:
- Raw ECG signal (blue)
- Filtered signal (green)
- Detected R-peaks (red dots)
- Heart rate over time (bottom panel)

### 4. Interactive Exploration

**Zoom:**
- Scroll mouse wheel to zoom in/out
- Or use 🔍 buttons in toolbar

**Pan:**
- Use matplotlib navigation toolbar
- Click and drag to move around

**Select Region:**
- Click ✂️ **Select** button in toolbar
- Click and drag on plot to select region
- Selected region highlighted in yellow

**Right-Click Menu:**
- Right-click on plot for quick actions:
  - Zoom In/Out
  - Reset View
  - Analyze Selection
  - Export Plot

### 5. Analyze Selection

After selecting a region:
- Right-click → Analyze Selection
- Or: Analyze → Analyze Selection

### 6. Export Results

**Save Figure:**
- File → Export → Export Figure (.png)
- Choose location and save

**Save Signal:**
- File → Export → Export Signal (.txt)

## Common Workflows

### Workflow 1: ECG Heart Rate Analysis

```
1. Import ECG signal
2. Click ECG button
3. View detected R-peaks and heart rate
4. Select interesting region (e.g., high variability)
5. Right-click → Analyze Selection
6. Analyze → HRV Analysis (for full HRV metrics)
7. Export results
```

### Workflow 2: EDA Stress Detection

```
1. Import EDA/GSR signal
2. Click 💧 EDA button
3. View phasic and tonic components
4. Identify stress events (rapid rises)
5. Select event regions
6. Analyze → Extract Features
7. Compare features across events
```

### Workflow 3: EMG Muscle Activity

```
1. Import EMG signal
2. Click 💪 EMG button
3. View envelope and activations
4. Select activation periods
5. Extract time-domain features
6. Export for further analysis
```

### Workflow 4: Batch Processing

```
1. Tools → Batch Processing
2. Add multiple files
3. Select signal type
4. Configure parameters
5. Process all
6. Export results to folder
```

## Keyboard Shortcuts Cheat Sheet

| Action | Shortcut |
|--------|----------|
| Import | `Ctrl+O` |
| Save | `Ctrl+S` |
| Undo | `Ctrl+Z` |
| Redo | `Ctrl+Y` |
| Delete Signal | `Del` |
| Refresh | `F5` |
| Help | `F1` |
| Quit | `Ctrl+Q` |

## Tips & Tricks

### 1. Multiple Signals
- Import multiple signals
- Switch between them in the signal list (left panel)
- Compare by viewing properties

### 2. Undo/Redo
- Made a mistake? Press `Ctrl+Z`
- All operations are reversible

### 3. Signal Properties
- Click signal in list to see properties in left panel
- Shows: length, sampling rate, duration, processing history

### 4. Custom Processing
- Use Process menu for specific filters
- Adjust cutoff frequencies as needed
- Try different detectors for best results

### 5. Selection Mode
- Toggle selection mode on/off as needed
- Create multiple selections for comparison
- Clear selection: Right-click → Clear Selection

## Troubleshooting Quick Fixes

### GUI doesn't start
```bash
# Test tkinter
python -m tkinter

# If fails, install:
sudo apt-get install python3-tk  # Linux
brew install python-tk           # macOS
```

### Import fails
```bash
# Check file format
head -n 5 your_signal.txt

# Try different import options
# Use column/channel selection for multi-column files
```

### Processing errors
- Check sampling rate is correct
- Ensure signal is not too short (min ~5 seconds)
- Try different detector algorithms

### Plot not updating
- Press `F5` to refresh
- Or: View → Refresh

## Example Data

Try the included examples:

```bash
cd examples/
ls *.txt

# You'll find:
# - ecg.txt      : ECG signal
# - eda.txt      : EDA signal
# - emg.txt      : EMG signal
# - ... and more
```

## Next Steps

After this quick start:

1. **Read Full Documentation**
   - [README.md](README.md) - Complete feature list
   - [INSTALLATION.md](INSTALLATION.md) - Detailed setup

2. **Explore All Signal Types**
   - Try EEG, PPG, Respiration
   - Each has specific features

3. **Advanced Features**
   - HRV analysis
   - Feature extraction
   - Signal quality assessment
   - Batch processing

4. **Extend with Plugins**
   - Create custom processing algorithms
   - Add your own visualizations
   - See plugins system in [README.md](README.md)

## Getting Help

- **Built-in Help**: Press `F1`
- **Documentation**: [README.md](README.md)
- **Issues**: [GitHub Issues](https://github.com/scientisst/BioSPPy/issues)
- **Examples**: Check `examples/` directory

## Video Tutorial (Conceptual)

### Typical 5-Minute Session

```
00:00 - Launch GUI
00:30 - Import ECG signal
01:00 - Process with ECG algorithm
01:30 - View results (R-peaks, HR)
02:00 - Enable selection mode
02:30 - Select interesting region
03:00 - Analyze selection
03:30 - Export figure
04:00 - Try different signal (EDA)
04:30 - Compare results
05:00 - Save work
```

## Summary

You now know how to:
- ✓ Launch the GUI
- ✓ Import signals
- ✓ Process signals (ECG, EDA, EMG, etc.)
- ✓ Use interactive features (zoom, pan, select)
- ✓ Analyze regions
- ✓ Export results

**Happy signal processing! 🎉**

---

For more details, see the complete [README.md](README.md) and [INSTALLATION.md](INSTALLATION.md).
