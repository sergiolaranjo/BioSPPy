#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-channel Signal Analysis Example
======================================

This example demonstrates how to:
1. Import and analyze multiple signal channels simultaneously
2. Perform individual channel analysis
3. Combine channels for baroreflex analysis

Author: BioSPPy Development Team
"""

import numpy as np
from biosppy.signals import multichannel, baroreflex
from biosppy import storage

# ==============================================================================
# EXAMPLE 1: Basic Multi-Channel Signal Creation and Analysis
# ==============================================================================

print("=" * 70)
print("EXAMPLE 1: Basic Multi-Channel Signal Analysis")
print("=" * 70)

# Load example ECG signal
ecg_signal, mdata = storage.load_txt('./examples/ecg.txt')

# Generate synthetic ABP signal for demonstration
# In real applications, load actual ABP data
sampling_rate = 1000.0
duration = len(ecg_signal) / sampling_rate

# Synthetic ABP: baseline + oscillation
t = np.arange(len(ecg_signal)) / sampling_rate
abp_signal = 100 + 20 * np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 2, len(t))

# Create multi-channel signal using dictionary
signals = {
    'ECG': ecg_signal,
    'ABP': abp_signal
}

channel_types = {
    'ECG': 'ecg',
    'ABP': 'abp'
}

# Initialize multi-channel signal
mc = multichannel.MultiChannelSignal(sampling_rate=sampling_rate)

# Add channels
mc.add_channel('ECG', ecg_signal, channel_type='ecg')
mc.add_channel('ABP', abp_signal, channel_type='abp')

print(f"\nMulti-channel signal created:")
print(mc)

# ==============================================================================
# EXAMPLE 2: Individual Channel Analysis
# ==============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 2: Individual Channel Analysis")
print("=" * 70)

# Process ECG channel
print("\nProcessing ECG channel...")
ecg_results = mc.process_channel('ECG')
print(f"  - Detected {len(ecg_results['rpeaks'])} R-peaks")
print(f"  - Mean heart rate: {np.mean(ecg_results['heart_rate']):.2f} bpm")

# Process ABP channel
print("\nProcessing ABP channel...")
abp_results = mc.process_channel('ABP')
print(f"  - Detected {len(abp_results['onsets'])} pulse onsets")
print(f"  - Mean heart rate from ABP: {np.mean(abp_results['heart_rate']):.2f} bpm")

# ==============================================================================
# EXAMPLE 3: Synchronization and Time Alignment
# ==============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 3: Channel Synchronization")
print("=" * 70)

# Synchronize channels using cross-correlation
offsets = mc.synchronize(reference_channel='ECG', method='cross_correlation')
print("\nSynchronization offsets (in seconds):")
for channel, offset in offsets.items():
    print(f"  - {channel}: {offset:.4f} s")

# Get time vectors for each channel
time_vectors = mc.get_time_vector()
print(f"\nTime vector lengths:")
for channel, tv in time_vectors.items():
    print(f"  - {channel}: {len(tv)} samples, {tv[-1]:.2f} s duration")

# ==============================================================================
# EXAMPLE 4: Baroreflex Analysis (ECG + ABP)
# ==============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 4: Baroreflex Sensitivity Analysis")
print("=" * 70)

# Analyze baroreflex using the integrated function
try:
    brs_results = baroreflex.analyze_multichannel_baroreflex(
        mc,
        ecg_channel='ECG',
        abp_channel='ABP',
        method='all',
        show=False  # Set to True to show plots
    )

    print("\nBaroreflex Sensitivity Results:")
    print("-" * 70)

    # Sequence method
    if 'brs_sequence' in brs_results.keys() and not np.isnan(brs_results['brs_sequence']):
        print(f"  Sequence Method:")
        print(f"    - BRS: {brs_results['brs_sequence']:.2f} ms/mmHg")
        print(f"    - Up sequences: {brs_results['n_sequences_up']}")
        print(f"    - Down sequences: {brs_results['n_sequences_down']}")

    # Spectral method
    if 'brs_spectral_lf' in brs_results.keys():
        print(f"\n  Spectral Method:")
        if not np.isnan(brs_results['brs_spectral_lf']):
            print(f"    - BRS LF: {brs_results['brs_spectral_lf']:.2f} ms/mmHg")
            print(f"    - Coherence LF: {brs_results['coherence_lf']:.3f}")
        else:
            print(f"    - BRS LF: N/A (low coherence)")

        if not np.isnan(brs_results['brs_spectral_hf']):
            print(f"    - BRS HF: {brs_results['brs_spectral_hf']:.2f} ms/mmHg")
            print(f"    - Coherence HF: {brs_results['coherence_hf']:.3f}")
        else:
            print(f"    - BRS HF: N/A (low coherence)")

    # Alpha method
    if 'brs_alpha_lf' in brs_results.keys():
        print(f"\n  Alpha Coefficient Method:")
        if not np.isnan(brs_results['brs_alpha_lf']):
            print(f"    - Alpha LF: {brs_results['brs_alpha_lf']:.2f} ms/mmHg")
        if not np.isnan(brs_results['brs_alpha_hf']):
            print(f"    - Alpha HF: {brs_results['brs_alpha_hf']:.2f} ms/mmHg")

except Exception as e:
    print(f"\nWarning: Baroreflex analysis failed: {str(e)}")
    print("This may be due to synthetic ABP data. Try with real ABP signals.")

# ==============================================================================
# EXAMPLE 5: Using Convenience Function
# ==============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 5: Using Convenience Function")
print("=" * 70)

# Create and process multi-channel signal in one step
mc2 = multichannel.multichannel(
    signals={'ECG': ecg_signal, 'ABP': abp_signal},
    sampling_rate=sampling_rate,
    channel_types={'ECG': 'ecg', 'ABP': 'abp'},
    process=True,
    synchronize=True
)

print(f"\nCreated and processed multi-channel signal:")
print(mc2)

# Get heart rate from all cardiovascular channels
hr_signals = mc2.get_heart_rate_signals()
print(f"\nHeart rate signals available:")
for channel, hr_data in hr_signals.items():
    print(f"  - {channel}: {len(hr_data['hr'])} values, "
          f"mean = {np.mean(hr_data['hr']):.2f} bpm")

# ==============================================================================
# EXAMPLE 6: Direct Baroreflex Analysis with Pre-computed Values
# ==============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 6: Direct Baroreflex Analysis")
print("=" * 70)

# Extract RR intervals
rpeaks = ecg_results['rpeaks']
rri = np.diff(rpeaks) / sampling_rate * 1000.0  # Convert to ms

# Extract systolic blood pressure values
# For demonstration, create synthetic SBP values
# In real applications, extract from actual ABP signal
onsets = abp_results['onsets']
sbp_values = []

for i in range(len(onsets) - 1):
    start = onsets[i]
    end = onsets[i + 1]
    segment = abp_signal[start:end]
    if len(segment) > 0:
        sbp_values.append(np.max(segment))

sbp_values = np.array(sbp_values)

# Compute baroreflex sensitivity using sequence method only
print("\nComputing baroreflex sensitivity (sequence method)...")
brs_direct = baroreflex.baroreflex_sensitivity(
    rri=rri[:len(sbp_values)],
    sbp=sbp_values,
    method='sequence',
    min_sequences=3,
    show=False
)

if 'brs_sequence' in brs_direct.keys() and not np.isnan(brs_direct['brs_sequence']):
    print(f"  - BRS: {brs_direct['brs_sequence']:.2f} ms/mmHg")
    print(f"  - Number of sequences: {brs_direct['n_sequences_up'] + brs_direct['n_sequences_down']}")
else:
    print("  - Could not compute BRS (insufficient sequences)")

# ==============================================================================
# EXAMPLE 7: Process All Channels at Once
# ==============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 7: Batch Processing All Channels")
print("=" * 70)

# Add a third channel (respiratory signal)
resp_signal, _ = storage.load_txt('./examples/resp.txt')
# Truncate or pad to match ECG length
if len(resp_signal) > len(ecg_signal):
    resp_signal = resp_signal[:len(ecg_signal)]
else:
    resp_signal = np.pad(resp_signal, (0, len(ecg_signal) - len(resp_signal)))

mc3 = multichannel.MultiChannelSignal(sampling_rate=sampling_rate)
mc3.add_channel('ECG', ecg_signal, channel_type='ecg')
mc3.add_channel('ABP', abp_signal, channel_type='abp')
mc3.add_channel('RESP', resp_signal, channel_type='resp')

print(f"\nMulti-channel signal with 3 channels:")
print(mc3)

# Process all channels
print("\nProcessing all channels...")
results = mc3.process_all()

print(f"\nProcessing results:")
for channel, result in results.items():
    if result is not None:
        print(f"  - {channel}: Successfully processed")
    else:
        print(f"  - {channel}: Failed to process")

# ==============================================================================
# EXAMPLE 8: Alignment and Resampling
# ==============================================================================

print("\n" + "=" * 70)
print("EXAMPLE 8: Channel Alignment")
print("=" * 70)

# Align all channels to the same length
aligned_signals = mc3.align_channels()
print(f"\nAligned channels to {len(aligned_signals['ECG'])} samples")
for channel, signal in aligned_signals.items():
    print(f"  - {channel}: {len(signal)} samples")

print("\n" + "=" * 70)
print("Examples completed successfully!")
print("=" * 70)

# ==============================================================================
# NOTES FOR REAL-WORLD USAGE
# ==============================================================================

print("\n" + "=" * 70)
print("NOTES FOR REAL-WORLD USAGE:")
print("=" * 70)
print("""
1. Signal Quality:
   - Ensure signals are properly filtered and free of artifacts
   - Use BioSPPy's quality assessment modules for signal quality checks

2. Synchronization:
   - If signals are recorded simultaneously, synchronization may not be needed
   - For non-simultaneous recordings, use cross-correlation synchronization

3. Baroreflex Analysis:
   - Requires at least 3-5 minutes of good quality data
   - Subject should be in supine position and resting
   - BRS values typically range from 5-30 ms/mmHg in healthy adults
   - Lower values indicate reduced baroreflex function

4. Signal Processing Parameters:
   - Adjust filtering parameters based on signal quality
   - For ECG, you can specify the R-peak detection algorithm
   - For ABP, adjust thresholds based on signal characteristics

5. Clinical Interpretation:
   - Always consider clinical context when interpreting results
   - Consult with medical professionals for clinical applications

6. File Formats:
   - Use storage.load_txt(), load_h5(), or load_edf() to load signals
   - Multi-channel recordings can be stored in HDF5 format for efficiency

Example loading from HDF5:
>>> from biosppy import storage
>>> with storage.HDF('data.h5') as hdf:
...     ecg = hdf.load_signal('ecg')
...     abp = hdf.load_signal('abp')
""")
