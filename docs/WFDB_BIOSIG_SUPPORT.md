# WFDB and BioSig Format Support

BioSPPy now supports importing biomedical signal files from PhysioNet's WFDB format and various other formats through BioSig.

## Installation

### WFDB Support (PhysioNet)

To load WFDB format files from PhysioNet:

```bash
pip install wfdb
```

Or install BioSPPy with WFDB support:

```bash
pip install biosppy[wfdb]
```

### BioSig Support

To load various biomedical signal formats using BioSig:

```bash
pip install pyBioSig
```

Or install BioSPPy with BioSig support:

```bash
pip install biosppy[biosig]
```

### Install All Optional Dependencies

```bash
pip install biosppy[all]
```

Or using the requirements file:

```bash
pip install -r requirements-optional.txt
```

## Usage

### Loading WFDB Files

```python
from biosppy import storage

# Load all channels
signals, metadata = storage.load_wfdb('path/to/record')

# Load specific channel by index
signal, metadata = storage.load_wfdb('path/to/record', channel=0)

# Load specific channel by name
signal, metadata = storage.load_wfdb('path/to/record', channel='MLII')

# Load multiple channels
signals, metadata = storage.load_wfdb('path/to/record', channel=[0, 2])
```

### Loading BioSig Files

```python
from biosppy import storage

# Load all channels
signals, metadata = storage.load_biosig('path/to/file.gdf')

# Load specific channel
signal, metadata = storage.load_biosig('path/to/file.gdf', channel=0)

# Load multiple channels
signals, metadata = storage.load_biosig('path/to/file.gdf', channel=[0, 2, 3])
```

### Processing Loaded Signals

```python
from biosppy import storage
from biosppy.signals import ecg

# Load ECG signal from WFDB
ecg_signal, metadata = storage.load_wfdb('path/to/ecg_record', channel=0)

# Process with BioSPPy
output = ecg.ecg(
    signal=ecg_signal,
    sampling_rate=metadata['sampling_rate'],
    show=True
)

print(f"Heart rate: {output['heart_rate']}")
print(f"R-peaks: {output['rpeaks']}")
```

## Supported Formats

### WFDB (PhysioNet)

- `.dat` / `.hea` - WFDB format files
- Used by PhysioNet databases (MIT-BIH, PTB, etc.)
- More info: https://physionet.org/

### BioSig

BioSig supports numerous biomedical signal formats:

- **EDF/EDF+** - European Data Format
- **BDF/BDF+** - BioSemi Data Format
- **GDF** - General Data Format
- **BrainVision** - BrainVision format
- **BioSemi** - BioSemi format
- And many others

More info: https://biosig.sourceforge.net/

## Metadata

Both loaders return signals and metadata dictionaries.

### WFDB Metadata

```python
{
    'sampling_rate': float,      # Sampling frequency (Hz)
    'units': list,               # Physical units for each signal
    'labels': list,              # Signal names
    'num_signals': int,          # Number of signals
    'num_samples': int,          # Number of samples per signal
    'base_date': datetime,       # Recording start date
    'base_time': datetime,       # Recording start time
    'comments': list,            # Comments from header
    'sig_name': list            # Original signal names
}
```

### BioSig Metadata

```python
{
    'sampling_rate': float,      # Sampling frequency (Hz)
    'units': list,               # Physical units for each signal
    'labels': list,              # Channel labels
    'num_signals': int,          # Number of signals
    'num_samples': int,          # Number of samples
    'patient_id': str,           # Patient identifier
    'recording_id': str,         # Recording identifier
    'start_datetime': datetime,  # Recording start time
    'file_type': str            # File format type
}
```

## Examples

See `examples/load_wfdb_biosig_example.py` for complete examples.

### Download Example Data

PhysioNet provides many public databases with WFDB format:

```bash
# Example: Download MIT-BIH Arrhythmia Database record
wget https://physionet.org/files/mitdb/1.0.0/100.dat
wget https://physionet.org/files/mitdb/1.0.0/100.hea
```

Then load:

```python
from biosppy import storage

signals, metadata = storage.load_wfdb('100')
print(f"Loaded {metadata['num_signals']} signals")
print(f"Signal names: {metadata['labels']}")
```

## References

1. **WFDB Software Package**: https://physionet.org/content/wfdb/
2. **PhysioNet**: https://physionet.org/
3. **BioSig**: https://biosig.sourceforge.net/
4. **WFDB Python Package**: https://github.com/MIT-LCP/wfdb-python

## Troubleshooting

### ImportError: No module named 'wfdb'

Install the wfdb package:
```bash
pip install wfdb
```

### ImportError: No module named 'biosig'

Install the pyBioSig package:
```bash
pip install pyBioSig
```

### IOError: Unable to read WFDB file

- Ensure both `.dat` and `.hea` files are present
- The path should point to the base name without extension
- Example: Use `'100'` not `'100.dat'` or `'100.hea'`

### BioSig installation issues

For some systems, you may need to install BioSig from source. See:
https://biosig.sourceforge.net/download.html
