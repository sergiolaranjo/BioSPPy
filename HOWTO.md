# BioSPPy How-To Guide

A practical guide for biosignal processing with BioSPPy. Each section is self-contained with copy-paste ready examples.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Loading Data](#2-loading-data)
3. [ECG Processing](#3-ecg-processing)
4. [EDA Processing](#4-eda-processing)
5. [PPG Processing](#5-ppg-processing)
6. [EMG Processing](#6-emg-processing)
7. [EEG Processing](#7-eeg-processing)
8. [Respiration Processing](#8-respiration-processing)
9. [Heart Rate Variability (HRV)](#9-heart-rate-variability-hrv)
10. [Empirical Mode Decomposition (EMD)](#10-empirical-mode-decomposition-emd)
11. [Baroreflex Sensitivity](#11-baroreflex-sensitivity)
12. [Multichannel Analysis](#12-multichannel-analysis)
13. [Feature Extraction](#13-feature-extraction)
14. [Signal Quality Assessment](#14-signal-quality-assessment)
15. [Nonlinear Dynamics and Chaos](#15-nonlinear-dynamics-and-chaos)
16. [Clustering](#16-clustering)
17. [Dimensionality Reduction](#17-dimensionality-reduction)
18. [Signal Processing Primitives](#18-signal-processing-primitives)
19. [Saving Results](#19-saving-results)

---

## 1. Installation

```bash
# from PyPI
pip install biosppy

# from GitHub (latest development version)
pip install git+https://github.com/scientisst/BioSPPy.git
```

---

## 2. Loading Data

BioSPPy supports several file formats through the `storage` module.

### From text files

```python
from biosppy import storage

# load a single-column signal
signal, mdata = storage.load_txt('./examples/ecg.txt')
print(f"Signal shape: {signal.shape}")
print(f"Metadata: {mdata}")
```

### From HDF5 files

```python
signal = storage.load_h5('recording.h5', label='ecg')
```

### From EDF files

```python
data = storage.load_edf('recording.edf')
```

### From WFDB records

```python
# see examples/load_wfdb_biosig_example.py for full details
from biosppy import storage
signal, fields = storage.load_txt('record.dat')
```

---

## 3. ECG Processing

One-line processing with filtering, R-peak detection, heart rate computation, and optional plotting.

### Basic usage

```python
import numpy as np
from biosppy import storage
from biosppy.signals import ecg

# load signal
signal, mdata = storage.load_txt('./examples/ecg.txt')

# process with defaults (Hamilton segmenter, 1000 Hz)
out = ecg.ecg(signal=signal, sampling_rate=1000., show=True)

# access results
print("Time axis:", out['ts'][:5])
print("R-peak indices:", out['rpeaks'][:5])
print("Heart rate (bpm):", out['heart_rate'][:5])
print("Number of templates:", out['templates'].shape[0])
```

### Choose a different R-peak detector

```python
# available segmenters: hamilton, ASI, gamboa, engzee, ssf,
#                       pan-tompkins, elgendi, kalidas, christov
out = ecg.ecg(signal=signal, sampling_rate=1000., segmenter='pan-tompkins', show=False)
```

### Use individual functions

```python
# just filter the signal
filtered, _, _ = ecg.st.filter_signal(
    signal=signal, ftype='FIR', band='bandpass',
    order=150, frequency=[3, 45], sampling_rate=1000.
)

# just detect R-peaks
rpeaks, = ecg.hamilton_segmenter(signal=filtered, sampling_rate=1000.)
rpeaks, = ecg.correct_rpeaks(signal=filtered, rpeaks=rpeaks,
                               sampling_rate=1000., tol=0.05)
```

### ECG wave delineation

```python
# detect P, QRS, T wave boundaries using the wavelet method (Martinez 2004)
out = ecg.ecg(signal=signal, sampling_rate=1000., show=False)
waves = ecg.ecg_wavelet_delineation(
    signal=signal, rpeaks=out['rpeaks'], sampling_rate=1000.
)
```

### Save the plot

```python
out = ecg.ecg(signal=signal, sampling_rate=1000., show=True, path='ecg_report.png')
```

---

## 4. EDA Processing

Electrodermal Activity processing with skin conductance response (SCR) detection.

### Basic usage

```python
from biosppy import storage
from biosppy.signals import eda

signal, mdata = storage.load_txt('./examples/eda.txt')
out = eda.eda(signal=signal, sampling_rate=1000., show=True)

print("Filtered signal shape:", out['filtered'].shape)
print("Number of SCR onsets:", len(out['onsets']))
print("Number of SCR peaks:", len(out['peaks']))
```

### SCR detection with custom parameters

```python
# adjust minimum SCR amplitude threshold
out = eda.eda(signal=signal, sampling_rate=1000., min_amplitude=0.05, show=False)
```

### SCR event analysis

```python
# get detailed SCR event information (onsets, peaks, amplitudes, recovery times)
events = eda.eda_events(
    signal=out['filtered'],
    sampling_rate=1000.,
    min_amplitude=0.1
)

print("SCR onsets:", events['onsets'])
print("SCR peaks:", events['peaks'])
print("SCR amplitudes:", events['amplitudes'])
print("Half recovery times:", events['half_rec'])
```

---

## 5. PPG Processing

Photoplethysmography analysis with pulse detection and pulse wave analysis.

### Basic usage

```python
from biosppy import storage
from biosppy.signals import ppg

signal, mdata = storage.load_txt('./examples/ppg.txt')
out = ppg.ppg(signal=signal, sampling_rate=1000., show=True)

print("Pulse peak indices:", out['peaks'][:5])
print("Heart rate:", out['heart_rate'][:5])
```

### Pulse wave analysis

```python
# access pulse wave features (dicrotic notch, augmentation index, etc.)
pwa = ppg.pulse_wave_analysis(
    signal=out['filtered'], peaks=out['peaks'], sampling_rate=1000.
)
```

---

## 6. EMG Processing

Electromyography with muscle activation onset detection.

### Basic usage

```python
from biosppy import storage
from biosppy.signals import emg

signal, mdata = storage.load_txt('./examples/emg.txt')
out = emg.emg(signal=signal, sampling_rate=1000., show=True)

print("Activation onsets:", out['onsets'])
```

---

## 7. EEG Processing

Electroencephalography with band-power extraction (delta, theta, alpha, beta, gamma).

### Basic usage

```python
from biosppy import storage
from biosppy.signals import eeg

signal, mdata = storage.load_txt('./examples/eeg_eo.txt')
out = eeg.eeg(signal=signal, sampling_rate=256., show=True)

# band powers for each channel
print("Theta power:", out['theta'][:3])
print("Alpha power:", out['alpha_low'][:3])
```

---

## 8. Respiration Processing

### Basic usage

```python
from biosppy import storage
from biosppy.signals import resp

signal, mdata = storage.load_txt('./examples/resp.txt')
out = resp.resp(signal=signal, sampling_rate=1000., show=True)

print("Breath zeros:", out['zeros'][:5])
print("Respiration rate:", out['resp_rate'][:5])
```

---

## 9. Heart Rate Variability (HRV)

Comprehensive HRV analysis including time-domain, frequency-domain, nonlinear, and wavelet features.

### From ECG R-peaks

```python
import numpy as np
from biosppy import storage
from biosppy.signals import ecg, hrv

# get R-peaks from ECG
signal, _ = storage.load_txt('./examples/ecg.txt')
ecg_out = ecg.ecg(signal=signal, sampling_rate=1000., show=False)

# compute all HRV features
hrv_results = hrv.hrv(rpeaks=ecg_out['rpeaks'], sampling_rate=1000., show=True)

# access specific features
for key, val in hrv_results.items():
    print(f"{key}: {val}")
```

### From pre-computed RR intervals

```python
# if you already have RR intervals in milliseconds
rri = np.array([800, 810, 790, 820, 805, 815, 795, 830])  # ms
hrv_results = hrv.hrv(rri=rri, show=False)
```

### Select feature domains

```python
# only time-domain features
hrv_time = hrv.hrv(rpeaks=ecg_out['rpeaks'], sampling_rate=1000.,
                   parameters='time', show=False)

# only frequency-domain features
hrv_freq = hrv.hrv(rpeaks=ecg_out['rpeaks'], sampling_rate=1000.,
                   parameters='frequency', show=False)

# only non-linear features
hrv_nl = hrv.hrv(rpeaks=ecg_out['rpeaks'], sampling_rate=1000.,
                 parameters='non-linear', show=False)

# all features
hrv_all = hrv.hrv(rpeaks=ecg_out['rpeaks'], sampling_rate=1000.,
                  parameters='all', show=False)
```

---

## 10. Empirical Mode Decomposition (EMD)

Decompose signals into Intrinsic Mode Functions (IMFs) for time-frequency analysis.

### Basic EMD

```python
import numpy as np
from biosppy.signals import emd

# create a signal
t = np.linspace(0, 1, 1000)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)

# decompose
result = emd.emd(signal=signal)
imfs = result['imfs']
residue = result['residue']

print(f"Number of IMFs: {len(imfs)}")
```

### CEEMDAN (recommended for noisy signals)

```python
result = emd.ceemdan(signal=signal, num_ensemble=100, noise_std=0.2)
imfs = result['imfs']
```

### Hilbert spectral analysis

```python
result = emd.emd(signal=signal)
hilbert = emd.hilbert_spectrum(imfs=result['imfs'], sampling_rate=1000.)
```

---

## 11. Baroreflex Sensitivity

Compute baroreflex sensitivity (BRS) from simultaneous blood pressure and RR interval data.

### Sequential method

```python
import numpy as np
from biosppy.signals import baroreflex

# synthetic data (replace with real measurements)
np.random.seed(42)
n = 200
sbp = 120 + 10 * np.sin(2 * np.pi * 0.1 * np.arange(n) / n) + np.random.randn(n)
rri = 800 + 50 * np.sin(2 * np.pi * 0.1 * np.arange(n) / n + 0.3) + np.random.randn(n) * 5

result = baroreflex.baroreflex_sensitivity(
    sbp=sbp, rri=rri, method='sequence', show=False
)
print(f"BRS: {result['brs_mean']:.2f} ms/mmHg")
```

### Baroreflex Effectiveness Index

```python
bei = baroreflex.baroreflex_effectiveness_index(sbp=sbp, rri=rri)
print(f"BEI: {bei['bei']:.2f}")
```

---

## 12. Multichannel Analysis

Process multiple synchronized biosignal channels together.

```python
import numpy as np
from biosppy import storage
from biosppy.signals.multichannel import MultiChannelSignal

# create a multichannel container
mc = MultiChannelSignal(sampling_rate=1000.)

# add channels
ecg_signal, _ = storage.load_txt('./examples/ecg.txt')
resp_signal, _ = storage.load_txt('./examples/resp.txt')

# trim to same length
min_len = min(len(ecg_signal), len(resp_signal))
mc.add_channel('ecg', ecg_signal[:min_len], signal_type='ecg')
mc.add_channel('resp', resp_signal[:min_len], signal_type='resp')

# process all channels
results = mc.process_all()
```

---

## 13. Feature Extraction

BioSPPy provides feature extraction across multiple domains.

### Time-domain features

```python
import numpy as np
from biosppy.features import time as time_features

signal = np.random.randn(1000)
feats = time_features.time_features(signal=signal)
```

### Frequency-domain features

```python
from biosppy.features import frequency as freq_features

feats = freq_features.frequency_features(signal=signal, sampling_rate=1000.)
```

### Cepstral features

```python
from biosppy.features import cepstral

feats = cepstral.cepstral_features(signal=signal, sampling_rate=1000.)
```

### Phase-space features

```python
from biosppy.features import phase_space

feats = phase_space.phase_space_features(signal=signal)
```

### Wavelet coherence

Analyze the time-frequency relationship between two signals.

```python
import numpy as np
from biosppy.features import wavelet_coherence as wc

fs = 100
t = np.linspace(0, 10, fs * 10)
signal1 = np.sin(2 * np.pi * 2 * t) + 0.3 * np.random.randn(len(t))
signal2 = np.sin(2 * np.pi * 2 * t + 0.5) + 0.3 * np.random.randn(len(t))

result = wc.wavelet_coherence(signal1=signal1, signal2=signal2,
                               sampling_rate=fs)
```

---

## 14. Signal Quality Assessment

Evaluate signal quality before downstream analysis.

### EDA quality

```python
import numpy as np
from biosppy import quality

eda_segment = np.random.randn(2000) + 5  # replace with real EDA
sqi = quality.quality_eda(x=eda_segment, sampling_rate=1000.)
```

### ECG quality

```python
ecg_segment = np.random.randn(10000)  # replace with real ECG
sqi = quality.quality_ecg(segment=ecg_segment, sampling_rate=1000.)
```

### Individual SQI metrics

```python
from biosppy.signals import ecg
from biosppy import quality

# coefficient of variation SQI from R-peaks
signal, _ = storage.load_txt('./examples/ecg.txt')
out = ecg.ecg(signal=signal, sampling_rate=1000., show=False)
c_sqi = quality.cSQI(rpeaks=out['rpeaks'])

# higher-order statistics SQI
hos_sqi = quality.hosSQI(signal=signal)
```

---

## 15. Nonlinear Dynamics and Chaos

Entropy measures, fractal dimensions, and Lyapunov exponents for complexity analysis.

### Entropy measures

```python
import numpy as np
from biosppy import chaos

signal = np.random.randn(1000)

# Shannon entropy
se = chaos.shannon_entropy(signal=signal)
print(f"Shannon Entropy: {se['shannon_entropy']:.4f}")

# Sample entropy
sampen = chaos.sample_entropy(signal=signal, m=2)
print(f"Sample Entropy: {sampen['sample_entropy']:.4f}")

# Approximate entropy
apen = chaos.approximate_entropy(signal=signal, m=2)
print(f"Approximate Entropy: {apen['approximate_entropy']:.4f}")
```

### Fractal dimensions

```python
# Higuchi Fractal Dimension
hfd = chaos.higuchi_fd(signal=signal, k_max=10)
print(f"Higuchi FD: {hfd['higuchi_fd']:.4f}")
```

### Lyapunov exponent

```python
# largest Lyapunov exponent (Rosenstein's algorithm)
lle = chaos.lyapunov_exponent(signal=signal, emb_dim=10, tau=1)
print(f"Largest Lyapunov Exponent: {lle['lyapunov_exponent']:.4f}")
```

---

## 16. Clustering

Cluster biosignal features or heartbeat templates.

```python
import numpy as np
from biosppy import clustering

# example: cluster heartbeat templates
data = np.random.randn(100, 50)  # 100 templates, 50 features each

# K-Means
result = clustering.kmeans(data=data, n_clusters=3)

# DBSCAN
result = clustering.dbscan(data=data, eps=1.5, min_samples=5)
```

---

## 17. Dimensionality Reduction

Reduce high-dimensional feature spaces for visualization or modeling.

```python
import numpy as np
from biosppy import dimensionality_reduction as dr

data = np.random.randn(100, 50)  # 100 samples, 50 features

# PCA
result = dr.pca(data=data, n_components=2)
reduced = result['reduced']

# t-SNE
result = dr.tsne(data=data, n_components=2)
```

---

## 18. Signal Processing Primitives

Low-level tools for filtering, frequency analysis, and peak detection.

### Filtering

```python
import numpy as np
from biosppy.signals import tools as st

signal = np.random.randn(1000)

# bandpass FIR filter
filtered, _, _ = st.filter_signal(
    signal=signal, ftype='FIR', band='bandpass',
    order=100, frequency=[1, 40], sampling_rate=1000.
)

# lowpass Butterworth filter
filtered, _, _ = st.filter_signal(
    signal=signal, ftype='butter', band='lowpass',
    order=4, frequency=50, sampling_rate=1000.
)
```

### Power spectrum

```python
freqs, power = st.power_spectrum(
    signal=signal, sampling_rate=1000., pad=1024
)
```

### Peak detection

```python
# find signal peaks
peaks = st.find_extrema(signal=signal, mode='max')
```

### Zero crossings

```python
zeros, = st.zero_cross(signal=signal, detrend=False)
```

### Normalize signal

```python
normalized = st.normalize(signal=signal)
```

---

## 19. Saving Results

### Save to text file

```python
from biosppy import storage
import numpy as np

storage.store_txt('output.txt', signal=np.array([1, 2, 3]), mdata={'fs': 1000})
```

### Save to HDF5

```python
storage.store_h5('output.h5', label='ecg', data=signal)
```

### Save plots

All top-level processing functions accept a `path` parameter:

```python
from biosppy.signals import ecg

out = ecg.ecg(signal=signal, sampling_rate=1000., show=True, path='ecg_summary.png')
```

---

## Tips and Best Practices

1. **Always specify the correct sampling rate.** All algorithms depend on it for filter design, peak detection thresholds, and frequency analysis.

2. **Check signal quality first.** Use `quality.quality_ecg()` or `quality.quality_eda()` before running feature extraction.

3. **Use `show=False` in batch processing.** Plotting slows down computation significantly when processing many files.

4. **Prefer CEEMDAN over EMD.** CEEMDAN provides better spectral separation and reduced mode mixing for noisy physiological signals.

5. **Handle NaN values.** Some algorithms (especially HRV) include outlier handling, but it's good practice to check for NaN or infinite values in your input signals.

6. **Use the right units.** BioSPPy expects signals in their raw units (e.g., mV for ECG, microsiemens for EDA). The `units` parameter is for labeling plots, not for conversion.
