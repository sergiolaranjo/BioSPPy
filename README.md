<a href="https://biosppy.readthedocs.org/">
<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/logo/logo_400.png">
  <source media="(prefers-color-scheme: dark)" srcset="docs/logo/logo_inverted_400.png">
  <img alt="Image" title="I know you're listening! - xkcd.com/525">
</picture>
</a>

*A toolbox for biosignal processing written in Python.*

[![PyPI version](https://badgen.net/pypi/v/biosppy)](https://pypi.org/project/biosppy/)
[![PyPI downloads](https://badgen.net/pypi/dm/biosppy/?color=blue)](https://pypi.org/project/biosppy/)
[![License](https://badgen.net/pypi/license/biosppy?color=grey)](https://github.com/scientisst/BioSPPy/blob/main/LICENSE)

[![GitHub stars](https://badgen.net/github/stars/scientisst/BioSPPy?color=yellow)]()
[![GitHub issues](https://badgen.net/github/open-issues/scientisst/BioSPPy?color=cyan)](https://github.com/scientisst/BioSPPy/issues)


# BioSPPy - Biosignal Processing in Python

BioSPPy is a toolbox for biosignal processing written in Python. It bundles together various signal processing and pattern recognition methods geared towards the analysis of biosignals.

## Highlights

- **Signal Processing**: ECG, EDA, EEG, EMG, PCG, PPG, ABP, BVP, Respiration, Accelerometry
- **Heart Rate Variability**: Time-domain, frequency-domain, and nonlinear HRV analysis
- **Feature Extraction**: Time, frequency, cepstral, phase-space, and wavelet coherence domains
- **Signal Quality Assessment**: Automated quality evaluation for EDA and ECG
- **Nonlinear Dynamics**: Entropy measures (Shannon, sample, approximate, multiscale), fractal dimensions (Higuchi, Katz, correlation dimension), Lyapunov exponents, and recurrence analysis
- **Empirical Mode Decomposition**: EMD, EEMD, and CEEMDAN with Hilbert spectral analysis
- **Baroreflex Sensitivity**: Sequential method (Di Rienzo) with multi-lag analysis
- **Multichannel Analysis**: Synchronized multi-signal import and analysis
- **Dimensionality Reduction**: PCA, ICA, NMF, t-SNE, MDS, Isomap, and UMAP
- **Clustering**: K-Means, DBSCAN, Gaussian Mixture Models, and hierarchical clustering
- **Biometrics**: ECG-based biometric identification
- **Signal Synthesizers**: Generate synthetic biosignals for testing and development
- **Storage**: Load signals from TXT, HDF5, EDF, WFDB, and BioSig formats

Documentation: <https://biosppy.readthedocs.org/>

## Installation

Installation can be easily done with `pip`:

```bash
$ pip install biosppy
```

Alternatively, install the latest development version from GitHub:

```bash
$ pip install git+https://github.com/scientisst/BioSPPy.git
```

## Quick Start

### ECG Processing

```python
from biosppy import storage
from biosppy.signals import ecg

# load raw ECG signal
signal, mdata = storage.load_txt('./examples/ecg.txt')

# process it and plot
out = ecg.ecg(signal=signal, sampling_rate=1000., show=True)

# access results
print("R-peak indices:", out['rpeaks'])
print("Heart rate:", out['heart_rate'])
```

![ECG summary example](docs/images/ECG_summary.png)

### EDA Processing

```python
from biosppy import storage
from biosppy.signals import eda

signal, mdata = storage.load_txt('./examples/eda.txt')
out = eda.eda(signal=signal, sampling_rate=1000., show=True)
```

### Heart Rate Variability

```python
from biosppy.signals import ecg, hrv

# process ECG to get R-peaks
signal, _ = storage.load_txt('./examples/ecg.txt')
out = ecg.ecg(signal=signal, sampling_rate=1000., show=False)

# compute HRV from R-peak intervals
rri = np.diff(out['rpeaks']) / 1000.  # convert to seconds
hrv_results = hrv.hrv(rri=rri, sampling_rate=1000.)
```

For a complete usage guide with more examples, see [HOWTO.md](HOWTO.md).

## Supported Signals

| Signal | Module | Description |
|--------|--------|-------------|
| ECG | `biosppy.signals.ecg` | Electrocardiography (R-peak detection, heart rate, wave delineation) |
| EDA | `biosppy.signals.eda` | Electrodermal Activity (SCR detection, tonic/phasic decomposition) |
| EEG | `biosppy.signals.eeg` | Electroencephalography (band-power extraction) |
| EMG | `biosppy.signals.emg` | Electromyography (onset detection, activation patterns) |
| PPG | `biosppy.signals.ppg` | Photoplethysmography (pulse detection, pulse wave analysis) |
| PCG | `biosppy.signals.pcg` | Phonocardiography (heart sound segmentation) |
| ABP | `biosppy.signals.abp` | Arterial Blood Pressure (systolic/diastolic detection) |
| BVP | `biosppy.signals.bvp` | Blood Volume Pulse |
| Respiration | `biosppy.signals.resp` | Respiratory signal analysis (breath detection, rate) |
| ACC | `biosppy.signals.acc` | Accelerometry (activity recognition) |
| HRV | `biosppy.signals.hrv` | Heart Rate Variability (time/frequency/nonlinear) |

## Project Structure

```
biosppy/
    signals/          # Signal-specific processing modules
        ecg.py        # ECG processing and R-peak detection
        eda.py        # EDA processing and SCR detection
        eeg.py        # EEG band-power analysis
        emg.py        # EMG onset detection
        ppg.py        # PPG processing and pulse wave analysis
        pcg.py        # PCG heart sound analysis
        abp.py        # Arterial blood pressure
        bvp.py        # Blood volume pulse
        resp.py       # Respiratory signal analysis
        acc.py        # Accelerometry
        hrv.py        # Heart rate variability
        emd.py        # Empirical Mode Decomposition
        baroreflex.py # Baroreflex sensitivity analysis
        multichannel.py # Multi-signal synchronized analysis
        tools.py      # Signal processing primitives
    features/         # Feature extraction modules
        time.py       # Time-domain features
        frequency.py  # Frequency-domain features
        cepstral.py   # Cepstral features
        time_freq.py  # Time-frequency features
        phase_space.py          # Phase-space features
        wavelet_coherence.py    # Wavelet coherence analysis
    chaos.py          # Nonlinear dynamics and entropy measures
    clustering.py     # Clustering algorithms
    dimensionality_reduction.py  # PCA, t-SNE, UMAP, etc.
    quality.py        # Signal quality assessment
    biometrics.py     # ECG-based biometric identification
    storage.py        # File I/O (TXT, HDF5, EDF, WFDB)
    plotting.py       # Visualization utilities
    utils.py          # General utilities
examples/             # Example scripts and sample data
```

## Dependencies

- numpy
- scipy
- scikit-learn
- matplotlib
- bidict
- h5py
- shortuuid
- six
- joblib

## Citing

Please use the following if you need to cite BioSPPy:

P. Bota, R. Silva, C. Carreiras, A. Fred, and H. P. da Silva, "BioSPPy: A Python toolbox for physiological signal processing," SoftwareX, vol. 26, pp. 101712, 2024, doi: 10.1016/j.softx.2024.101712.

```latex
@article{biosppy,
    title = {BioSPPy: A Python toolbox for physiological signal processing},
    author = {Patrícia Bota and Rafael Silva and Carlos Carreiras and Ana Fred and Hugo Plácido {da Silva}},
    journal = {SoftwareX},
    volume = {26},
    pages = {101712},
    year = {2024},
    issn = {2352-7110},
    doi = {https://doi.org/10.1016/j.softx.2024.101712},
    url = {https://www.sciencedirect.com/science/article/pii/S2352711024000839},
}
```

However, if you want to cite a specific version of BioSPPy, you can use Zenodo's DOI:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11048615.svg)](https://doi.org/10.5281/zenodo.11048615)


## License
BioSPPy is released under the BSD 3-clause license. See LICENSE for more details.

## Disclaimer

This program is distributed in the hope it will be useful and provided
to you "as is", but WITHOUT ANY WARRANTY, without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. This
program is NOT intended for medical diagnosis. We expressly disclaim any
liability whatsoever for any direct, indirect, consequential, incidental
or special damages, including, without limitation, lost revenues, lost
profits, losses resulting from business interruption or loss of data,
regardless of the form of action or legal theory under which the
liability may be asserted, even if advised of the possibility of such
damages.
