#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hilbert-Huang Transform (HHT) for Heart Rate Variability Analysis
==================================================================

This example demonstrates how to use the Hilbert-Huang Transform (HHT)
for analyzing heart rate variability (HRV) using the CEEMDAN method.

The HHT consists of:
1. Empirical Mode Decomposition (EMD) - decomposes signal into IMFs
2. Hilbert Spectral Analysis - extracts instantaneous frequency/amplitude

CEEMDAN (Complete Ensemble EMD with Adaptive Noise) is the most recent
and improved version of EMD, providing better spectral separation.

Author: BioSPPy Development Team
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from biosppy.signals import ecg, hrv, emd

# Generate or load ECG signal
# For demonstration, we'll create a synthetic RR-interval series
print("=" * 70)
print("HHT-based Heart Rate Variability Analysis Example")
print("=" * 70)

# Method 1: Using real ECG data (if available)
# Uncomment and provide your ECG file:
# signal, mdata = st.load('path/to/ecg_signal.txt')
# ecg_result = ecg.ecg(signal=signal, sampling_rate=1000, show=False)
# rpeaks = ecg_result['rpeaks']
# rri = np.diff(rpeaks) / 1000.0 * 1000  # Convert to ms

# Method 2: Generate synthetic RR-interval series
print("\n1. Generating synthetic RR-interval series...")
np.random.seed(42)
n_beats = 300
t = np.arange(n_beats)

# Create synthetic RRI with multiple components:
# - Baseline: ~800 ms
# - Respiratory sinus arrhythmia (HF): ~0.25 Hz
# - LF oscillations: ~0.1 Hz
# - VLF trend: ~0.02 Hz
# - Random noise

baseline = 800
respiratory = 30 * np.sin(2 * np.pi * 0.25 * t / 4.0)  # HF component
lf_component = 20 * np.sin(2 * np.pi * 0.1 * t / 4.0)   # LF component
vlf_trend = 15 * np.sin(2 * np.pi * 0.02 * t / 4.0)     # VLF component
noise = np.random.randn(n_beats) * 5

rri = baseline + respiratory + lf_component + vlf_trend + noise

print(f"   - Number of beats: {n_beats}")
print(f"   - Mean RRI: {np.mean(rri):.2f} ms")
print(f"   - Std RRI: {np.std(rri):.2f} ms")

# =============================================================================
# Part 1: CEEMDAN Decomposition
# =============================================================================
print("\n2. Performing CEEMDAN decomposition...")
print("   (This may take a minute...)")

# Decompose using CEEMDAN (most recent HHT method)
ceemdan_result = emd.ceemdan(
    signal=rri,
    num_ensemble=50,      # Number of ensemble members (trade-off: accuracy vs speed)
    noise_std=0.2,        # Noise standard deviation
    max_imf=None,         # Auto-determine number of IMFs
    random_seed=42        # For reproducibility
)

imfs = ceemdan_result['imfs']
residue = ceemdan_result['residue']

print(f"   - Number of IMFs extracted: {len(imfs)}")
print(f"   - Residue std: {np.std(residue):.2f} ms")

# Verify reconstruction
reconstructed = np.sum(imfs, axis=0) + residue
reconstruction_error = np.sqrt(np.mean((rri - reconstructed) ** 2))
print(f"   - Reconstruction error: {reconstruction_error:.4f} ms (should be ~0)")

# =============================================================================
# Part 2: Hilbert Spectral Analysis
# =============================================================================
print("\n3. Computing Hilbert spectrum...")

# Compute instantaneous amplitude and frequency
hilbert_result = emd.hilbert_spectrum(imfs=imfs, sampling_rate=4.0)
inst_amplitude = hilbert_result['inst_amplitude']
inst_frequency = hilbert_result['inst_frequency']
inst_phase = hilbert_result['inst_phase']

print(f"   - Instantaneous amplitude shape: {inst_amplitude.shape}")
print(f"   - Instantaneous frequency shape: {inst_frequency.shape}")

# =============================================================================
# Part 3: HRV Features using HHT
# =============================================================================
print("\n4. Extracting HRV features using HHT...")

# 4a. Basic variability features
print("\n   4a. Basic HHT variability features:")
hht_var = hrv.hht_variability(rri=rri, method='ceemdan', num_ensemble=50,
                              random_seed=42)

print(f"      - Total energy: {hht_var['total_energy']:.2f}")
print(f"      - Energy per IMF:")
for i, (energy, ratio) in enumerate(zip(hht_var['imf_energy'],
                                        hht_var['energy_ratio'])):
    freq_mean = hht_var['imf_frequency_mean'][i]
    print(f"        IMF {i+1}: Energy={energy:.2f}, Ratio={ratio:.3f}, "
          f"Mean Freq={freq_mean:.4f} Hz")

# 4b. Frequency band analysis
print("\n   4b. HHT-based frequency band analysis:")
hht_bands = hrv.hht_frequency_bands(rri=rri, method='ceemdan',
                                    num_ensemble=50, random_seed=42)

print(f"      - VLF power: {hht_bands['vlf_power']:.2f} ms²")
print(f"      - LF power:  {hht_bands['lf_power']:.2f} ms²")
print(f"      - HF power:  {hht_bands['hf_power']:.2f} ms²")
print(f"      - Total power: {hht_bands['total_power']:.2f} ms²")
print(f"      - LF/HF ratio: {hht_bands['lf_hf_ratio']:.2f}")
print(f"      - LF norm: {hht_bands['lf_norm']:.2f}%")
print(f"      - HF norm: {hht_bands['hf_norm']:.2f}%")

print("\n      - IMF to frequency band mapping:")
for imf_idx, band_info in hht_bands['imf_to_band'].items():
    print(f"        IMF {imf_idx+1}: {band_info['band'].upper()} "
          f"(mean freq: {band_info['mean_freq']:.4f} Hz)")

# 4c. Non-linear features
print("\n   4c. HHT-based non-linear features:")
hht_nonlinear = hrv.hht_nonlinear_features(rri=rri, method='ceemdan',
                                           num_ensemble=50, random_seed=42)

print(f"      - Number of IMFs: {hht_nonlinear['n_imfs']}")
print(f"      - Complexity index: {hht_nonlinear['complexity_index']:.3f}")
print(f"      - Energy entropy: {hht_nonlinear['energy_entropy']:.3f}")
print(f"      - Frequency entropy: {hht_nonlinear['frequency_entropy']:.3f}")
print(f"      - Residue std: {hht_nonlinear['residue_std']:.2f} ms")
print(f"      - Reconstruction error: {hht_nonlinear['reconstruction_error']:.4f} ms")

# =============================================================================
# Part 4: Visualization
# =============================================================================
print("\n5. Creating visualizations...")

fig = plt.figure(figsize=(14, 12))

# Plot 1: Original RRI and Reconstruction
ax1 = plt.subplot(4, 2, 1)
ax1.plot(rri, 'b-', linewidth=1, label='Original RRI')
ax1.plot(reconstructed, 'r--', linewidth=1, alpha=0.7, label='Reconstructed')
ax1.set_ylabel('RRI (ms)')
ax1.set_title('Original vs Reconstructed RRI')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: IMFs
for i in range(min(6, len(imfs))):
    ax = plt.subplot(4, 2, i + 2)
    ax.plot(imfs[i], linewidth=0.8)
    mean_freq = hht_var['imf_frequency_mean'][i]
    energy_ratio = hht_var['energy_ratio'][i]
    ax.set_ylabel(f'IMF {i+1}')
    ax.set_title(f'IMF {i+1} (f={mean_freq:.3f}Hz, E={energy_ratio:.2f})')
    ax.grid(True, alpha=0.3)

    if i == 5:
        ax.set_xlabel('Beat number')

# Plot 7: Residue (trend)
ax7 = plt.subplot(4, 2, 7)
ax7.plot(residue, 'k-', linewidth=1.5)
ax7.set_ylabel('Residue (ms)')
ax7.set_xlabel('Beat number')
ax7.set_title('Residue (Trend Component)')
ax7.grid(True, alpha=0.3)

# Plot 8: Energy distribution
ax8 = plt.subplot(4, 2, 8)
imf_labels = [f'IMF{i+1}' for i in range(len(imfs))]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
bars = ax8.bar(imf_labels, hht_var['energy_ratio'],
               color=colors[:len(imfs)])
ax8.set_ylabel('Energy Ratio')
ax8.set_xlabel('IMF')
ax8.set_title('Energy Distribution Across IMFs')
ax8.grid(True, alpha=0.3, axis='y')
plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('/home/user/BioSPPy/examples/hht_hrv_analysis.png', dpi=150,
            bbox_inches='tight')
print("   - Figure saved as: examples/hht_hrv_analysis.png")

# Additional plot: Hilbert spectrum (marginal spectrum)
print("\n6. Computing marginal Hilbert spectrum...")
freq_bins = np.linspace(0, 0.5, 100)
marginal_result = emd.marginal_spectrum(
    inst_amplitude=inst_amplitude,
    inst_frequency=inst_frequency,
    freq_bins=freq_bins,
    sampling_rate=4.0
)

frequencies = marginal_result['frequencies']
amplitudes = marginal_result['amplitudes']

fig2, ax = plt.subplots(figsize=(10, 5))
ax.plot(frequencies, amplitudes, 'b-', linewidth=2)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Marginal Amplitude')
ax.set_title('Marginal Hilbert Spectrum')
ax.grid(True, alpha=0.3)
ax.axvspan(0.003, 0.04, alpha=0.2, color='blue', label='VLF')
ax.axvspan(0.04, 0.15, alpha=0.2, color='green', label='LF')
ax.axvspan(0.15, 0.4, alpha=0.2, color='red', label='HF')
ax.legend()
plt.tight_layout()
plt.savefig('/home/user/BioSPPy/examples/hht_marginal_spectrum.png',
            dpi=150, bbox_inches='tight')
print("   - Figure saved as: examples/hht_marginal_spectrum.png")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY OF HHT-BASED HRV ANALYSIS")
print("=" * 70)
print(f"Signal characteristics:")
print(f"  - Length: {len(rri)} beats")
print(f"  - Mean RRI: {np.mean(rri):.2f} ms")
print(f"  - SDNN: {np.std(rri):.2f} ms")
print()
print(f"CEEMDAN Decomposition:")
print(f"  - Number of IMFs: {len(imfs)}")
print(f"  - Reconstruction error: {reconstruction_error:.4f} ms")
print()
print(f"Frequency-domain features (HHT-based):")
print(f"  - VLF power: {hht_bands['vlf_power']:.2f} ms²")
print(f"  - LF power: {hht_bands['lf_power']:.2f} ms²")
print(f"  - HF power: {hht_bands['hf_power']:.2f} ms²")
print(f"  - LF/HF ratio: {hht_bands['lf_hf_ratio']:.2f}")
print()
print(f"Non-linear features (HHT-based):")
print(f"  - Complexity index: {hht_nonlinear['complexity_index']:.3f}")
print(f"  - Energy entropy: {hht_nonlinear['energy_entropy']:.3f}")
print(f"  - Frequency entropy: {hht_nonlinear['frequency_entropy']:.3f}")
print()
print("Advantages of HHT over traditional methods:")
print("  ✓ Adaptive and data-driven (no fixed basis functions)")
print("  ✓ Suitable for non-linear and non-stationary signals")
print("  ✓ Provides instantaneous frequency information")
print("  ✓ Better time-frequency resolution than STFT/Wavelets")
print("  ✓ CEEMDAN reduces mode mixing problem")
print("=" * 70)

plt.show()
