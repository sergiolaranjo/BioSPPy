#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example: HRV Analysis using Discrete Wavelet Transform (DWT)
==============================================================

This example demonstrates how to perform Heart Rate Variability (HRV)
analysis using Discrete Wavelet Transform with Daubechies 12 wavelet.

The wavelet-based HRV analysis provides a multi-resolution time-frequency
representation of the RR-interval sequence, which can reveal different
components related to autonomic nervous system activity.

"""

import sys
import os

# Add the current directory to path to import biosppy from source
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from biosppy.signals import ecg, hrv

# Example 1: Direct wavelet analysis from RRI sequence
print("=" * 70)
print("Example 1: Direct Wavelet Analysis from RRI")
print("=" * 70)

# Load RRI data directly (which has enough data points)
rri_data = np.loadtxt('./examples/rri.txt')

print(f"\nRRI sequence length: {len(rri_data)} intervals")
print(f"Mean RR interval: {np.mean(rri_data):.2f} ms")
print(f"Mean heart rate: {60000/np.mean(rri_data):.2f} bpm")
print(f"Signal duration: {np.sum(rri_data)/1000:.2f} seconds")

# Compute wavelet-based HRV features using Daubechies 12
print("\nComputing wavelet-based HRV features (Daubechies 12)...")
hrv_out = hrv.hrv_wavelet(rri=rri_data,
                          wavelet='db12',
                          level=6,
                          detrend_rri=True,
                          show=False)

print("\nWavelet HRV Features:")
print("-" * 70)
for key in hrv_out.keys():
    value = hrv_out[key]
    if isinstance(value, (int, float, np.integer, np.floating)):
        print(f"{key:30s}: {value:.6f}")
    elif isinstance(value, str):
        print(f"{key:30s}: {value}")
    elif isinstance(value, (int, np.integer)):
        print(f"{key:30s}: {value}")

# Example 2: Compare different wavelets
print("\n" + "=" * 70)
print("Example 2: Comparing Different Wavelets")
print("=" * 70)

# Analyze with different wavelets for comparison
wavelets = ['db12', 'db8', 'db4', 'sym5']

print("\nComparing different wavelets:")
print("-" * 70)

for wavelet_name in wavelets:
    hrv_out = hrv.hrv_wavelet(rri=rri_data,
                              wavelet=wavelet_name,
                              level=6,
                              detrend_rri=True,
                              show=False)

    print(f"\nWavelet: {wavelet_name}")
    actual_level = hrv_out['decomposition_level']
    print(f"  Decomposition Level: {actual_level}")
    print(f"  Total Energy: {hrv_out['dwt_energy_total']:.2f}")
    print(f"  Wavelet Entropy: {hrv_out['dwt_entropy']:.6f}")

    # Show energy distribution across levels (dynamically based on actual level)
    print(f"  Energy distribution:")
    for i in range(1, actual_level + 1):
        rel_energy = hrv_out[f'dwt_energy_rel_d{i}']
        print(f"    Detail {i}: {rel_energy*100:.2f}%")
    rel_energy_app = hrv_out[f'dwt_energy_rel_a{actual_level}']
    print(f"    Approx {actual_level}: {rel_energy_app*100:.2f}%")

# Example 3: Compute all HRV features including wavelet
print("\n" + "=" * 70)
print("Example 3: Complete HRV Analysis (All Features)")
print("=" * 70)

print("\nComputing all HRV features (time, frequency, non-linear, wavelet)...")
# For this we need to first get ECG data and extract R-peaks
# Since the example ECG is too short, we'll use the RRI directly with hrv() function
print("Note: Using RRI data directly for demonstration")
hrv_all = hrv.hrv(rri=rri_data,
                  parameters='all',
                  wavelet='db12',
                  wavelet_level=6,
                  show=False)

print(f"\nTotal number of features extracted: {len(hrv_all)}")

# Group features by domain
print("\nFeatures by domain:")
print("-" * 70)

time_features = [k for k in hrv_all.keys() if k.startswith(('hr_', 'rr_', 'rmssd', 'nn', 'pnn', 'sdnn', 'hti', 'tinn'))]
freq_features = [k for k in hrv_all.keys() if k.endswith(('_pwr', '_peak', '_rpwr', '_nu')) or k == 'lf_hf']
nonlinear_features = [k for k in hrv_all.keys() if k in ['s', 'sd1', 'sd2', 'sd12', 'sd21', 'sampen', 'appen']]
wavelet_features = [k for k in hrv_all.keys() if k.startswith('dwt_')]

print(f"\nTime-domain features ({len(time_features)}):")
for k in sorted(time_features):
    if isinstance(hrv_all[k], (int, float, np.integer, np.floating)):
        print(f"  {k:25s}: {hrv_all[k]:.4f}")

print(f"\nFrequency-domain features ({len(freq_features)}):")
for k in sorted(freq_features):
    if isinstance(hrv_all[k], (int, float, np.integer, np.floating)):
        print(f"  {k:25s}: {hrv_all[k]:.4f}")

print(f"\nNon-linear features ({len(nonlinear_features)}):")
for k in sorted(nonlinear_features):
    if isinstance(hrv_all[k], (int, float, np.integer, np.floating)):
        print(f"  {k:25s}: {hrv_all[k]:.4f}")

print(f"\nWavelet features ({len(wavelet_features)}):")
for k in sorted(wavelet_features)[:10]:  # Show first 10
    if isinstance(hrv_all[k], (int, float, np.integer, np.floating)):
        print(f"  {k:25s}: {hrv_all[k]:.4f}")
if len(wavelet_features) > 10:
    print(f"  ... and {len(wavelet_features) - 10} more wavelet features")

# Example 4: Visualize wavelet decomposition
print("\n" + "=" * 70)
print("Example 4: Visualize Wavelet Decomposition")
print("=" * 70)
print("\nNote: Set show=True to display wavelet decomposition plots")
print("      This will show the approximation and detail coefficients")
print("      along with their energy distributions.")

# Uncomment to visualize:
# hrv.hrv_wavelet(rri=rri_data, wavelet='db12', level=6, show=True)

print("\n" + "=" * 70)
print("Examples completed successfully!")
print("=" * 70)
