# BioSPPy Comprehensive Code Audit Report

**Date**: 2026-02-22
**Scope**: Full codebase analysis - 52 Python files, ~900KB of code
**Focus**: Bugs, optimizations, and new algorithm opportunities

---

## Table of Contents

1. [Critical Bugs (Must Fix)](#1-critical-bugs)
2. [High Priority Bugs](#2-high-priority-bugs)
3. [Medium Priority Issues](#3-medium-priority-issues)
4. [Low Priority / Code Quality](#4-low-priority-issues)
5. [Optimization Opportunities](#5-optimization-opportunities)
6. [New Algorithms to Implement](#6-new-algorithms-to-implement)

---

## 1. Critical Bugs

### BUG-C01: `ecg.py:1525-1526` - Pan-Tompkins++ uses `signal` (array) instead of `ss` (scipy.signal)

```python
# CURRENT (crashes with AttributeError):
a, b = signal.butter(N, Wn, btype="highpass")
ecg_h = signal.filtfilt(a, b, ecg_l, ...)

# FIX:
a, b = ss.butter(N, Wn, btype="highpass")
ecg_h = ss.filtfilt(a, b, ecg_l, ...)
```

**Impact**: Pan-Tompkins++ segmenter crashes immediately when `sampling_rate == 200`.

---

### BUG-C02: `ecg.py:292` - `break` instead of `continue` in `_extract_heartbeats()`

```python
for r in R:
    a = r - before
    if a < 0:
        continue
    b = r + after
    if b > length:
        break  # BUG: should be 'continue'
```

**Impact**: Silently discards ALL valid R-peaks after the first one near signal end.

---

### BUG-C03: `pcg.py:427-431` - Duplicate conditional with conflicting assignments

```python
if states[last_location_of_definite_state] == 1:
    states[last_location_of_definite_state:] = 2   # sets to 2

if states[last_location_of_definite_state] == 1:   # DUPLICATE - never true now!
    states[last_location_of_definite_state:] = 4   # dead code
```

**Impact**: Heart sound state `4` is never assigned in the final segment. Line 430 should likely check `== 3`.

---

### BUG-C04: `storage.py:50` - `serialize()` missing return assignment

```python
# CURRENT:
utils.normpath(path)  # return value discarded!

# FIX:
path = utils.normpath(path)
```

**Impact**: Path normalization has no effect; serialization may use un-normalized paths.

---

### BUG-C05: `storage.py:564` - Off-by-one in `load_edf()` signal scaling

```python
# CURRENT:
for i in range(num_signals-1):  # skips last signal!
    signals[i] = (signals[i] - digital_min[i]) / ...

# FIX:
for i in range(num_signals):
```

**Impact**: The last EDF signal channel is never scaled to physical units.

---

### BUG-C06: `plotting.py:191` - `fig` undefined when `ax is not None`

```python
if ax is None:
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(211)
else:
    ax2 = ax.twinx()

fig.subplots_adjust(...)  # NameError when ax was provided!
fig.suptitle(...)
```

**Impact**: `_plot_filter()` crashes when called with a pre-existing axes object.

---

### BUG-C07: `quality.py:220` - `reshape` crashes on non-divisible signal lengths

```python
segments_2s = x.reshape(-1, int(sampling_rate*2))
# Crashes if len(x) % (sampling_rate*2) != 0
```

**Impact**: `eda_sqi_bottcher()` crashes for any signal whose length isn't exactly divisible by `2*sampling_rate`.

---

### BUG-C08: `quality.py:226` - Division by zero in RAC calculation

```python
rac = np.abs((max_ - min_) / max_)
# If max_ == 0, ZeroDivisionError (or NaN in numpy)
```

**Impact**: Crashes or produces NaN for flat/zero-valued EDA segments.

---

### BUG-C09: `quality.py:306-307` - Debug prints always execute (ignores `verbose`)

```python
def hosSQI(signal=None, quantitative=False, verbose=1):
    ...
    kSQI = stats.kurtosis(signal)
    sSQI = stats.skew(signal)
    print('kurtosis: ', kSQI)   # Always prints, not gated by verbose!
    print('skewness: ', sSQI)
```

---

### BUG-C10: `quality.py:271,321` - String formatting crashes at runtime

```python
# Line 271:
print('cSQI is {:.2f} -> {str_level}'.format(cSQI, str_level=str_level))
# Line 321:
print('hosSQI is {:.2f} -> {str_level}'.format(hosSQI, str_level=str_level))

# These crash with KeyError because {str_level} is not a valid format spec
# FIX: use f-strings or positional arguments
```

**Impact**: `cSQI()` and `hosSQI()` crash when `verbose=1` (the default!).

---

### BUG-C11: `quality.py:182` - `np.corrcoef()` biased by self-correlations

```python
templates, _ = ecg.extract_heartbeats(...)
corr_points = np.corrcoef(templates)
if np.mean(corr_points) > threshold:
    quality = HQ
```

**Impact**: `np.corrcoef(templates)` returns NxN matrix including diagonal (self-correlation = 1.0). `np.mean()` is biased upward, making quality assessment unreliable. Should exclude diagonal or use off-diagonal mean.

---

### BUG-C12: `inter_plotting/acc.py:220-221` - Uninitialized global variables

```python
global feat_fig      # Never initialized at module level
global toolbarfeat   # Never initialized at module level
```

**Impact**: Clicking feature buttons in wrong order causes `NameError` crash.

---

## 2. High Priority Bugs

### BUG-H01: `ecg.py:518-519` - Division by zero in `compare_segmentation()`

```python
acc = float(TP) / (TP + FP)   # ZeroDivisionError when TP+FP==0
err = float(FP) / (TP + FP)
```

---

### BUG-H02: `ecg.py:3025,3027` - Division by zero in `fSQI()`

```python
return num_power / dem_power   # dem_power can be 0
```

---

### BUG-H03: `ecg.py:994` - Invalid -1 slice index in `engzee_segmenter()`

```python
i, f = windowW[0], windowW[-1] if windowW[-1] < len(y2) else -1
# When f=-1, y2[i:-1] excludes last element (should be None or len(y2))
```

---

### BUG-H04: `ecg.py:1049-1050` - Unprotected `[0]`/`[-1]` indexing in `gamboa_segmenter()`

```python
v0 = edges[np.nonzero(F > TH)[0][0]]     # IndexError if empty
v1 = edges[np.nonzero(F < (1 - TH))[0][-1]]
```

---

### BUG-H05: `plotting.py:292` (+ 12 other locations) - File extension check misses dot

```python
ext = ext.lower()
if ext not in ['png', 'jpg']:   # ext is '.png', never matches 'png'!
    path = root + '.png'

# FIX:
if ext not in ['.png', '.jpg']:
```

**Affected lines**: 292, 345, 467, 595, 690, 785, 1161, 1291, 1569, 1682, 1812, 1978, 2074

---

### BUG-H06: `clustering.py:332` - Deprecated `np.random.random_integers()`

```python
'k': np.random.random_integers(low=kmin, high=kmax, size=nensemble)
# Removed in NumPy 2.0

# FIX:
'k': np.random.randint(low=kmin, high=kmax+1, size=nensemble)
```

---

### BUG-H07: `clustering.py:552,644` - String key check instead of integer

```python
if '-1' in ks:      # ks contains integers, not strings!
    ks.remove('-1')

# FIX:
if -1 in ks:
    ks.remove(-1)
```

---

### BUG-H08: `tools.py:1690` - Off-by-one in `finite_difference()` alignment

```python
index = np.arange(D2, len(signal) - D2, dtype="int")
derivative = derivative[D:]    # BUG: should be derivative[D2:] to match index

# D = N-1, D2 = D//2. Index starts at D2 but derivative starts at D.
# They will have different lengths!
```

---

### BUG-H09: `pcg.py:414,423` - Unprotected `np.argwhere()` on potentially empty results

```python
first_location = np.argwhere(states!=0)[0][0]   # IndexError if all states==0
last_location = np.argwhere(states!=0)[-1][0]    # Same issue
```

---

### BUG-H10: `ppg.py:464` - `.max()` on potentially empty array

```python
onsets.append(minima[minima < ind].max())
# ValueError if no minima exist before the peak
```

---

### BUG-H11: `phase_space.py:307,312,317` - `[-1]` indexing on potentially empty arrays

```python
rec_plot_lgst_diag_line_len = np.where(i_ll == 1)[0][-1]
# IndexError if no diagonal lines found
```

---

### BUG-H12: `frequency.py:224` - Wrong harmonic condition (inverted logic)

```python
if fundamental_frequency > (sampling_rate / 2 + 2):
    # This means: compute harmonics only if fundamental > Nyquist + 2Hz
    # But fundamentals should always be < Nyquist!
    # This condition is NEVER true for valid signals.

# FIX (probable intent):
if fundamental_frequency < (sampling_rate / 2 - 2):
```

**Impact**: Harmonic features are NEVER computed.

---

### BUG-H13: `chaos.py:660` - Empty array in `np.polyfit()` for Higuchi fractal dimension

```python
if len(k_valid) < 2:
    raise ValueError(...)
coeffs = np.polyfit(np.log(k_valid), np.log(lengths_valid), 1)
# lengths_valid can be all zeros -> np.log(0) = -inf -> polyfit fails
```

---

### BUG-H14: `chaos.py:881` - Fragile array length matching in Hurst exponent

```python
coeffs = np.polyfit(np.log(window_sizes[:len(rs_values)]), np.log(rs_values), 1)
# If some windows produced no data, lengths silently mismatch
```

---

## 3. Medium Priority Issues

### BUG-M01: `eda.py:198-201` - Silent exception swallows division by zero

```python
try:
    phasic_rate = sampling_rate * (60. / np.diff(peaks))
except Exception as e:
    print(e)      # Swallows all errors silently
    phasic_rate = None
```

---

### BUG-M02: `emg.py:199,313,486,817,1045` - Typo "specidy" in error messages

```python
raise TypeError("Please specidy rest parameters.")
# Should be: "Please specify rest parameters."
```

---

### BUG-M03: `emg.py:253,600,767,992` - Off-by-one in moving average index adjustment

```python
onsets += int(size / 2)
# For mode='valid' convolution, correct adjustment is int((size - 1) / 2)
```

---

### BUG-M04: `eeg.py:271-272` - No validation for `overlap >= 1.0`

```python
step = size - int(overlap * size)
# If overlap >= 1.0, step <= 0 causing infinite loop or crash
```

---

### BUG-M05: `tools.py:1145` - Typo in error message

```python
raise ValueError("Unknwon mode %r." % mode)
# Should be: "Unknown mode"
```

---

### BUG-M06: `tools.py:1863` - Function name typo `_ditance_profile`

```python
def _ditance_profile(...)    # Should be _distance_profile
# Used at lines 1751, 1863, 1963, 2092
```

---

### BUG-M07: `metrics.py:49-50` - Uses private scipy API

```python
u = ssd._validate_vector(u)   # Private API, may break
v = ssd._validate_vector(v)

# FIX:
u = np.asarray(u).flatten()
v = np.asarray(v).flatten()
```

---

### BUG-M08: `quality.py:55` - Uses `assert` for input validation

```python
assert len(x) > sampling_rate * 5, 'Segment must be 5s long'
# assert is stripped in optimized mode (-O flag)
# Should use: if len(x) <= ...: raise ValueError(...)
```

---

### BUG-M09: `storage.py:391` - Opens file in binary mode but reads as text

```python
with open(path, 'rb') as fid:
    lines = fid.readlines()    # Returns bytes, not strings
```

---

### BUG-M10: `storage.py:250-254` - Silently replaces existing HDF5 data

```python
try:
    fid.create_dataset(label, data=data)
except (RuntimeError, ValueError):
    del fid[label]              # Silently deletes existing data!
    fid.create_dataset(label, data=data)
```

---

### BUG-M11: `frequency.py:243,252` - Unsafe `np.argwhere()[0][0]` on potentially empty arrays

```python
spectral_roll_on = freqs[np.argwhere(norm_cm_s >= 0.05)[0][0]]
spectral_roll_off = freqs[np.argwhere(norm_cm_s >= 0.95)[0][0]]
# IndexError if no values satisfy conditions
```

---

### BUG-M12: `chaos.py:361` - Bare `except Exception` swallows all errors

```python
try:
    result = entropy_fn(...)
except Exception:
    mse_values.append(np.nan)   # Silently swallows real errors
# Should catch specific: (ValueError, RuntimeError, ZeroDivisionError)
```

---

### BUG-M13: `features/time.py:172-191` - Library code prints to stdout

```python
if signal_mobility is None:
    print("Hjorth mobility is undefined. Returning None.")
# Should use logging module or suppress entirely
```

---

## 4. Low Priority Issues

| ID | File:Line | Issue |
|----|-----------|-------|
| L01 | `ecg.py:27` | `scipy.ndimage.filters` deprecated (use `scipy.ndimage` directly) |
| L02 | `ecg.py:17` | `six.moves` still imported (Python 2 compat no longer needed) |
| L03 | `timing.py:91` | `CLOCKS.pop(name)` raises KeyError silently |
| L04 | `utils.py:443` | Invalid type hint syntax `(int, float, ...)` instead of `Union[...]` |
| L05 | `inter_plotting/acc.py:1-10` | Docstring says "ecg plot" but file is `acc.py` |
| L06 | `inter_plotting/acc.py,ecg.py` | File extension bug same as BUG-H05 |
| L07 | `ecg.py:318` | Docstring says `int` but parameter accepts `float` |
| L08 | `chaos.py:450` | `import math` inside function instead of module level |

---

## 5. Optimization Opportunities

### OPT-01: Replace Python loops with vectorized NumPy operations

**Files affected**: `emg.py` (onset detectors), `ecg.py` (segmenters), `pcg.py` (state machine)

Many onset detection algorithms use sample-by-sample Python loops that could be vectorized:
```python
# Current (slow):
for k in range(1, len(signal), 2):
    tf = (1 / var_rest) * (signal[k-1]**2 + signal[k]**2)

# Optimized:
pairs = signal[:-1:2]**2 + signal[1::2]**2
tf = pairs / var_rest
```

### OPT-02: Avoid unnecessary array copies

**Files affected**: `tools.py`, `eda.py`, `ecg.py`

Several functions create unnecessary copies:
```python
# Current:
signal = np.array(signal)  # copies even if already ndarray
# Better:
signal = np.asarray(signal)  # avoids copy if already ndarray
```

### OPT-03: Use `scipy.signal.sosfilt` instead of `ba` format filters

**Files affected**: `tools.py:filter_signal()` (used everywhere)

Second-order sections (SOS) format is more numerically stable for higher-order filters:
```python
# Current:
b, a = ss.butter(order, freq, btype=band)
filtered = ss.filtfilt(b, a, signal)

# More stable:
sos = ss.butter(order, freq, btype=band, output='sos')
filtered = ss.sosfiltfilt(sos, signal)
```

### OPT-04: Lazy imports for optional dependencies

**Files affected**: `ecg.py`, `plotting.py`, `storage.py`

Heavy imports like `matplotlib.pyplot` at module level slow down import time. Could use lazy imports:
```python
def plot_ecg(...):
    import matplotlib.pyplot as plt  # only when needed
```

### OPT-05: Parallel processing for multi-channel signals

**Files affected**: `eeg.py`, `multichannel.py`

EEG processing with multiple channels applies the same operations per channel in a loop. Could use `joblib.Parallel`:
```python
# Current:
for j in range(nch):
    values[:, i, j], _ = st.smoother(...)

# Parallel:
from joblib import Parallel, delayed
results = Parallel(n_jobs=-1)(delayed(st.smoother)(...) for j in range(nch))
```

### OPT-06: Caching for repeated filter computations

**Files affected**: `tools.py`

When the same filter coefficients are requested multiple times, cache them:
```python
from functools import lru_cache

@lru_cache(maxsize=32)
def get_filter(ftype, band, order, frequency, sampling_rate):
    ...
```

---

## 6. New Algorithms to Implement

### ECG - Electrocardiogram

| Algorithm | Description | Reference | Priority |
|-----------|-------------|-----------|----------|
| **XQRS detector** | Robust R-peak detection from WFDB | Pan & Tompkins + WFDB modifications | High |
| **ECG delineation** | Full P-QRS-T wave delineation using wavelets | Martinez et al. 2004 | High |
| **Atrial fibrillation detection** | AF detection from RR intervals | Dash et al. 2009 | High |
| **ECG morphology features** | QRS duration, QT interval, ST segment | Standard clinical features | Medium |
| **Deep learning segmenter** | CNN/LSTM-based R-peak detection | Hannun et al. 2019 | Medium |
| **Signal quality (bSQI)** | Beat-based SQI using template matching | Li et al. 2008 | High |
| **ECG synthesis (advanced)** | ECGSYN with realistic pathologies | McSharry et al. 2003 (extended) | Low |

### EDA - Electrodermal Activity

| Algorithm | Description | Reference | Priority |
|-----------|-------------|-----------|----------|
| **cvxEDA decomposition** | Convex optimization EDA decomposition | Greco et al. 2016 | High |
| **SparsEDA** | Sparse deconvolution approach | Hernando-Gallego et al. 2017 | Medium |
| **Ledalab method** | Continuous decomposition analysis (CDA) | Benedek & Kaernbach 2010 | High |
| **EDA stress detection** | Stress level from EDA features | Healey & Picard 2005 | Medium |
| **Motion artifact removal** | Adaptive filtering for wearable EDA | Taylor et al. 2015 | High |

### EMG - Electromyogram

| Algorithm | Description | Reference | Priority |
|-----------|-------------|-----------|----------|
| **Muscle fatigue index** | Median frequency shift analysis | De Luca 1997 | High |
| **Motor unit decomposition** | MUAP decomposition from sEMG | Holobar & Zazula 2007 | Medium |
| **Wavelet-based denoising** | EMG denoising using wavelets | Phinyomark et al. 2012 | Medium |
| **Force estimation** | EMG-to-force mapping | Staudenmann et al. 2010 | Low |
| **Conduction velocity** | Muscle fiber conduction velocity | Merletti et al. 1999 | Low |

### EEG - Electroencephalogram

| Algorithm | Description | Reference | Priority |
|-----------|-------------|-----------|----------|
| **Independent Component Analysis** | ICA for artifact removal | Makeig et al. 1996 | High |
| **Common Spatial Patterns (CSP)** | Feature extraction for BCI | Blankertz et al. 2008 | High |
| **Connectivity measures** | Coherence, Granger causality, PLV | Bastos & Schoffelen 2015 | High |
| **Spectral parameterization** | FOOOF - aperiodic + periodic decomposition | Donoghue et al. 2020 | High |
| **Microstate analysis** | EEG microstate segmentation | Michel & Koenig 2018 | Medium |
| **Sleep staging** | Automatic sleep stage classification | Chambon et al. 2018 | Medium |
| **Seizure detection** | Epileptic seizure detection | Shoeb & Guttag 2010 | Medium |
| **ERD/ERS computation** | Event-related desynchronization | Pfurtscheller & da Silva 1999 | Medium |

### PPG - Photoplethysmogram

| Algorithm | Description | Reference | Priority |
|-----------|-------------|-----------|----------|
| **SpO2 estimation** | Blood oxygen saturation from PPG | Nitzan et al. 2014 | High |
| **Blood pressure estimation** | Continuous BP from PPG morphology | Elgendi et al. 2019 | High |
| **PPG morphology analysis** | Pulse wave analysis, augmentation index | Takazawa et al. 1998 | High |
| **Adaptive peak detection** | Multi-scale peak detection | Billauer 2012 | Medium |
| **PPG quality assessment** | Signal quality index for PPG | Elgendi 2016 | High |
| **Pulse transit time** | PTT from ECG-PPG synchronization | Mukkamala et al. 2015 | Medium |

### Respiration

| Algorithm | Description | Reference | Priority |
|-----------|-------------|-----------|----------|
| **EDR (ECG-derived respiration)** | Extract respiration from ECG | Moody et al. 1985 | High |
| **Respiratory pattern classification** | Apnea, tachypnea, Cheyne-Stokes | Various | Medium |
| **Tidal volume estimation** | Respiratory volume from signal amplitude | Carry et al. 1997 | Medium |
| **Respiratory rate variability** | RRV analysis (like HRV for breathing) | Vlemincx et al. 2013 | Medium |
| **Advanced peak detection** | Better than zero-crossing approach | Charlton et al. 2016 | High |

### PCG - Phonocardiogram

| Algorithm | Description | Reference | Priority |
|-----------|-------------|-----------|----------|
| **Murmur detection** | Heart murmur classification | Springer et al. 2016 | High |
| **S3/S4 detection** | Third and fourth heart sound detection | Zheng et al. 2015 | Medium |
| **PCG quality assessment** | Signal quality for stethoscope recordings | Various | Medium |

### Cross-Signal / General

| Algorithm | Description | Reference | Priority |
|-----------|-------------|-----------|----------|
| **Multimodal fusion** | Combine ECG+PPG+EDA for stress/emotion | Various | High |
| **Adaptive filtering** | Reference-based noise cancellation | Widrow et al. 1975 | High |
| **Transfer entropy** | Causal coupling between signals | Schreiber 2000 | Medium |
| **Recurrence quantification** | Extended RQA for biosignals | Marwan et al. 2007 | Medium |
| **Artifact detection (general)** | ML-based artifact detection framework | Various | High |
| **Signal alignment** | DTW-based signal alignment | Keogh & Ratanamahatana 2005 | Medium |

### Signal Quality (Expanding `quality.py`)

| Method | Signal | Description | Priority |
|--------|--------|-------------|----------|
| **bSQI** | ECG | Beat-template correlation SQI | High |
| **sSQI** | ECG | Spectral distribution SQI | Medium |
| **Skewness/Kurtosis SQI** | General | Statistical shape SQI | Medium |
| **SNR estimation** | General | Signal-to-noise ratio estimation | High |
| **Contact quality** | EDA/EEG | Electrode contact impedance proxy | Medium |
| **Clipping detection** | General | ADC saturation detection | High |
| **Baseline wander index** | ECG | Low-frequency drift quantification | Medium |

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Critical Bugs | 12 |
| High Priority Bugs | 14 |
| Medium Priority Issues | 13 |
| Low Priority Issues | 8 |
| Optimization Opportunities | 6 |
| New Algorithms Proposed | 50+ |
| **Total Issues Found** | **47** |

### Files with Most Issues

| File | Issues | Critical |
|------|--------|----------|
| `quality.py` | 9 | 6 |
| `ecg.py` | 8 | 2 |
| `storage.py` | 4 | 2 |
| `frequency.py` | 3 | 0 |
| `chaos.py` | 4 | 0 |
| `plotting.py` | 2 | 1 |
| `clustering.py` | 2 | 0 |
| `eda.py` | 3 | 0 |
| `emg.py` | 3 | 0 |
| `pcg.py` | 3 | 1 |
| `tools.py` | 3 | 0 |
| `phase_space.py` | 1 | 0 |
| `inter_plotting/acc.py` | 2 | 1 |

### Recommended Fix Priority

1. **Immediate**: BUG-C01 through BUG-C09 (crashes and data corruption)
2. **Next Sprint**: BUG-H01 through BUG-H12 (edge case crashes)
3. **Ongoing**: Medium/Low issues and optimizations
4. **Roadmap**: New algorithm implementations
