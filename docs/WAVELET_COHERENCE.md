# Wavelet Coherence Analysis

## Overview

The `wavelet_coherence` module provides tools for analyzing the relationship between two signals in the time-frequency domain using wavelet coherence analysis. This is particularly useful for biosignal processing where you need to understand how two signals are related at different frequencies and how this relationship changes over time.

## Features

- **Wavelet Coherence**: Measure cross-correlation between two signals as a function of frequency and time
- **Phase Analysis**: Compute phase differences to understand the temporal relationship between signals
- **Temporal Delay**: Calculate time delays between signals at different frequencies
- **Cross-Wavelet Spectrum**: Compute cross-wavelet power and phase
- **Statistical Significance Testing**: Test the significance of coherence values

## Theoretical Background

### Wavelet Coherence

Wavelet coherence (WTC) measures the cross-correlation between two time series as a function of frequency and time. It is analogous to the coherence between two signals in the frequency domain, but localized in time.

The wavelet coherence is defined as:

```
WTC = |S(W₁ · W₂*)|² / (S(|W₁|²) · S(|W₂|²))
```

Where:
- W₁ and W₂ are the continuous wavelet transforms of the two signals
- W₂* is the complex conjugate of W₂
- S is a smoothing operator in time and scale
- The result is a value between 0 and 1, where 1 indicates perfect coherence

### Phase Difference

The phase difference φ indicates the temporal relationship between the two signals:

```
φ = angle(S(W₁ · W₂*))
```

- φ = 0: signals are in phase
- φ = π/2: signal 1 leads signal 2 by a quarter cycle
- φ = π: signals are in anti-phase
- φ = -π/2: signal 2 leads signal 1 by a quarter cycle

### Temporal Delay

The temporal delay τ represents the time shift between signals:

```
τ = φ / (2πf)
```

Where f is the frequency. Positive values indicate signal 1 leads signal 2, negative values indicate signal 2 leads signal 1.

## Installation

The module requires the following dependencies (automatically installed with BioSPPy):

```bash
pip install numpy scipy pywavelets matplotlib
```

## Usage Examples

### Basic Wavelet Coherence

```python
import numpy as np
from biosppy.features import wavelet_coherence as wc

# Generate two example signals
fs = 100  # sampling rate in Hz
t = np.linspace(0, 10, fs * 10)
signal1 = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
signal2 = np.sin(2 * np.pi * 5 * t + 0.3) + 0.3 * np.sin(2 * np.pi * 10 * t)

# Compute wavelet coherence
result = wc.wavelet_coherence(signal1, signal2, sampling_rate=fs)

# Access results
coherence = result['coherence']  # coherence matrix (frequencies x time)
frequencies = result['frequencies']  # frequency values
phase = result['phase']  # phase difference matrix
delay = result['delay']  # temporal delay matrix
```

### Cross-Wavelet Spectrum

```python
# Compute cross-wavelet spectrum
result = wc.cross_wavelet_spectrum(signal1, signal2, sampling_rate=fs)

cross_spectrum = result['cross_spectrum']  # complex cross-spectrum
power = result['power']  # magnitude of cross-spectrum
phase = result['phase']  # phase of cross-spectrum
frequencies = result['frequencies']
```

### Computing Temporal Delay

```python
# Compute temporal delay from phase
delay = wc.compute_temporal_delay(phase, frequencies, unwrap=True)
# Result is in seconds
# Positive values: signal1 leads signal2
# Negative values: signal2 leads signal1
```

### Significance Testing

```python
# Test statistical significance
is_significant, threshold = wc.significance_test(coherence, n_samples=1000, alpha=0.05)
# is_significant is a boolean matrix indicating where coherence is significant
# threshold is the coherence value for 95% confidence
```

### Plotting

```python
import matplotlib.pyplot as plt

# Simple visualization
fig, ax = wc.plot_wavelet_coherence(coherence, frequencies, time_axis=t,
                                     phase=phase, phase_step=10)
plt.show()
```

### Complete Example

See `examples/wavelet_coherence_example.py` for comprehensive examples including:

1. **Synthetic Signals**: Analysis of signals with multiple frequency components
2. **Time Delay Detection**: Detecting and quantifying time delays between signals
3. **Cross-Wavelet Spectrum**: Visualizing common power between signals

Run the examples:

```bash
cd examples
python wavelet_coherence_example.py
```

## Function Reference

### `wavelet_coherence(signal1, signal2, sampling_rate, wavelet='morl', scales=None, compute_phase=True, compute_delay=True)`

Compute wavelet coherence between two signals.

**Parameters:**
- `signal1`, `signal2` (array): Input signals (must have same length)
- `sampling_rate` (float): Sampling rate in Hz
- `wavelet` (str): Wavelet name ('morl', 'cmor', 'cgau', 'mexh', 'gaus'). Default: 'morl'
- `scales` (array): Scales for wavelet transform. If None, automatically computed
- `compute_phase` (bool): Whether to compute phase difference. Default: True
- `compute_delay` (bool): Whether to compute temporal delay. Default: True

**Returns:**
- `coherence` (array): Wavelet coherence matrix (scales × time)
- `frequencies` (array): Corresponding frequencies in Hz
- `phase` (array): Phase difference in radians (if compute_phase=True)
- `delay` (array): Temporal delay in seconds (if compute_delay=True)
- `cross_spectrum` (array): Complex cross-wavelet spectrum

### `cross_wavelet_spectrum(signal1, signal2, sampling_rate, wavelet='morl', scales=None)`

Compute cross-wavelet spectrum between two signals.

**Parameters:**
- `signal1`, `signal2` (array): Input signals
- `sampling_rate` (float): Sampling rate in Hz
- `wavelet` (str): Wavelet name. Default: 'morl'
- `scales` (array): Scales for wavelet transform

**Returns:**
- `cross_spectrum` (array): Complex cross-wavelet spectrum
- `frequencies` (array): Frequencies in Hz
- `power` (array): Magnitude of cross-spectrum
- `phase` (array): Phase of cross-spectrum in radians

### `compute_temporal_delay(phase, frequencies, unwrap=True)`

Compute temporal delay from phase difference.

**Parameters:**
- `phase` (array): Phase difference matrix in radians
- `frequencies` (array): Frequencies in Hz
- `unwrap` (bool): Whether to unwrap phase. Default: True

**Returns:**
- `delay` (array): Temporal delay in seconds

### `significance_test(coherence, n_samples, alpha=0.05)`

Test statistical significance of wavelet coherence.

**Parameters:**
- `coherence` (array): Wavelet coherence values
- `n_samples` (int): Number of independent samples
- `alpha` (float): Significance level. Default: 0.05

**Returns:**
- `is_significant` (array): Boolean matrix indicating significance
- `threshold` (float): Coherence threshold for significance

## Applications in Biosignal Processing

### ECG-Respiration Coupling

Analyze the relationship between heart rate and respiration:

```python
from biosppy.signals import ecg, resp

# Process ECG and respiration signals
ecg_signal = ...  # your ECG data
resp_signal = ...  # your respiration data

# Compute wavelet coherence
result = wc.wavelet_coherence(ecg_signal, resp_signal, sampling_rate=1000)
# High coherence at respiratory frequency indicates cardiorespiratory coupling
```

### EEG Coherence

Analyze synchronization between brain regions:

```python
# Compare EEG from two electrodes
eeg_channel1 = ...
eeg_channel2 = ...

result = wc.wavelet_coherence(eeg_channel1, eeg_channel2, sampling_rate=256)
# High coherence indicates synchronized activity between regions
```

### EMG Synchronization

Study muscle coordination:

```python
# Compare EMG from two muscles
emg_muscle1 = ...
emg_muscle2 = ...

result = wc.wavelet_coherence(emg_muscle1, emg_muscle2, sampling_rate=1000)
# Analyze phase to understand timing of muscle activation
```

## Interpretation Guidelines

### Coherence Values

- **0.0 - 0.3**: Low coherence (signals are largely independent)
- **0.3 - 0.7**: Moderate coherence (some relationship between signals)
- **0.7 - 1.0**: High coherence (strong relationship between signals)

### Phase Interpretation

- **In-phase (φ ≈ 0)**: Signals peak together
- **Anti-phase (φ ≈ ±π)**: When one signal peaks, the other is at minimum
- **Quadrature (φ ≈ ±π/2)**: One signal leads the other by a quarter cycle

### Time-Frequency Localization

- **High frequency, localized in time**: Transient events
- **Low frequency, persistent**: Long-term trends or oscillations
- **Varying coherence**: Intermittent relationship between signals

## Limitations and Considerations

1. **Edge Effects**: Wavelet transform may have edge effects at the beginning and end of the signal. Consider using signals longer than the phenomena of interest.

2. **Cone of Influence**: Low-frequency components require longer time windows and may be affected by edge effects.

3. **Statistical Significance**: High coherence values don't automatically imply causality. Use significance testing and domain knowledge.

4. **Computational Cost**: Wavelet coherence is computationally intensive for long signals. Consider downsampling if appropriate.

5. **Wavelet Choice**: Different wavelets have different time-frequency localization properties:
   - **Morlet ('morl')**: Good balance, commonly used (default)
   - **Complex Morlet ('cmor')**: Better frequency resolution
   - **Mexican hat ('mexh')**: Good time localization
   - **Gaussian derivatives ('gaus')**: Fast, simple

## References

1. Torrence, C. and Compo, G.P., 1998. A practical guide to wavelet analysis. *Bulletin of the American Meteorological Society*, 79(1), pp.61-78.

2. Grinsted, A., Moore, J.C. and Jevrejeva, S., 2004. Application of the cross wavelet transform and wavelet coherence to geophysical time series. *Nonlinear Processes in Geophysics*, 11(5/6), pp.561-566.

3. Aguiar-Conraria, L. and Soares, M.J., 2014. The continuous wavelet transform: Moving beyond uni‐and bivariate analysis. *Journal of Economic Surveys*, 28(2), pp.344-375.

## Support and Contributing

For issues, questions, or contributions, please visit the [BioSPPy GitHub repository](https://github.com/scientisst/BioSPPy).

## License

This module is part of BioSPPy and is distributed under the BSD 3-clause license.
