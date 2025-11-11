#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Wavelet Coherence Example
--------------------------

This example demonstrates the use of wavelet coherence analysis
to analyze the relationship between two signals in the time-frequency domain.

"""

import numpy as np
import matplotlib.pyplot as plt
from biosppy.features import wavelet_coherence as wc


def example_synthetic_signals():
    """Example with synthetic signals."""

    print("=" * 60)
    print("Wavelet Coherence Analysis - Synthetic Signals Example")
    print("=" * 60)

    # Generate time vector
    fs = 100  # sampling frequency (Hz)
    duration = 10  # seconds
    t = np.linspace(0, duration, int(fs * duration))

    # Generate two signals with some correlation
    # Signal 1: combination of two frequencies
    freq1 = 5  # Hz
    freq2 = 10  # Hz
    signal1 = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)

    # Signal 2: similar to signal1 but with phase shift and different amplitudes
    phase_shift = 0.3  # phase shift in radians
    signal2 = np.sin(2 * np.pi * freq1 * t + phase_shift) + 0.3 * np.sin(2 * np.pi * freq2 * t + phase_shift)

    # Add some noise
    signal1 += 0.1 * np.random.randn(len(t))
    signal2 += 0.1 * np.random.randn(len(t))

    print(f"\nSignal properties:")
    print(f"  Duration: {duration} seconds")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Signal length: {len(t)} samples")
    print(f"  Frequency components: {freq1} Hz and {freq2} Hz")
    print(f"  Phase shift: {phase_shift:.2f} radians ({np.degrees(phase_shift):.1f} degrees)")

    # Compute wavelet coherence
    print("\nComputing wavelet coherence...")
    result = wc.wavelet_coherence(signal1, signal2,
                                   sampling_rate=fs,
                                   compute_phase=True,
                                   compute_delay=True)

    coherence = result['coherence']
    frequencies = result['frequencies']
    phase = result['phase']
    delay = result['delay']

    print("Done!")
    print(f"\nResult dimensions:")
    print(f"  Coherence matrix: {coherence.shape}")
    print(f"  Frequencies: {len(frequencies)}")
    print(f"  Frequency range: {frequencies.min():.2f} - {frequencies.max():.2f} Hz")

    # Find maximum coherence and its location
    max_coh_idx = np.unravel_index(np.argmax(coherence), coherence.shape)
    max_coh_value = coherence[max_coh_idx]
    max_coh_freq = frequencies[max_coh_idx[0]]
    max_coh_time = t[max_coh_idx[1]]

    print(f"\nMaximum coherence:")
    print(f"  Value: {max_coh_value:.4f}")
    print(f"  Frequency: {max_coh_freq:.2f} Hz")
    print(f"  Time: {max_coh_time:.2f} seconds")

    # Compute average coherence at the main frequency components
    freq_tolerance = 1.0  # Hz
    for freq_target in [freq1, freq2]:
        freq_mask = np.abs(frequencies - freq_target) < freq_tolerance
        if np.any(freq_mask):
            avg_coh = np.mean(coherence[freq_mask, :])
            print(f"\nAverage coherence at {freq_target} Hz: {avg_coh:.4f}")

            # Average delay at this frequency
            avg_delay = np.mean(delay[freq_mask, :])
            print(f"  Average delay: {avg_delay*1000:.2f} ms")

    # Plot results
    print("\nGenerating plots...")
    fig = plt.figure(figsize=(15, 10))

    # Plot signals
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(t, signal1, 'b-', linewidth=0.5, label='Signal 1')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Signal 1')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(t, signal2, 'r-', linewidth=0.5, label='Signal 2')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Signal 2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot wavelet coherence
    ax3 = plt.subplot(3, 2, 3)
    extent = [t[0], t[-1], frequencies[0], frequencies[-1]]
    im1 = ax3.imshow(coherence, aspect='auto', origin='lower',
                     extent=extent, cmap='viridis', vmin=0, vmax=1)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_title('Wavelet Coherence')
    ax3.set_ylim([0, 20])  # Focus on relevant frequency range
    plt.colorbar(im1, ax=ax3, label='Coherence')

    # Plot phase
    ax4 = plt.subplot(3, 2, 4)
    im2 = ax4.imshow(phase, aspect='auto', origin='lower',
                     extent=extent, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_title('Phase Difference')
    ax4.set_ylim([0, 20])
    plt.colorbar(im2, ax=ax4, label='Phase (rad)')

    # Plot delay
    ax5 = plt.subplot(3, 2, 5)
    # Clip delay for better visualization
    delay_clipped = np.clip(delay * 1000, -500, 500)  # convert to ms and clip
    im3 = ax5.imshow(delay_clipped, aspect='auto', origin='lower',
                     extent=extent, cmap='RdBu_r', vmin=-200, vmax=200)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Frequency (Hz)')
    ax5.set_title('Temporal Delay')
    ax5.set_ylim([0, 20])
    plt.colorbar(im3, ax=ax5, label='Delay (ms)')

    # Plot coherence at specific frequency
    ax6 = plt.subplot(3, 2, 6)
    freq_idx = np.argmin(np.abs(frequencies - freq1))
    ax6.plot(t, coherence[freq_idx, :], 'b-', linewidth=1, label=f'Coherence at {frequencies[freq_idx]:.1f} Hz')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Coherence')
    ax6.set_title(f'Coherence Time Series at ~{freq1} Hz')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('wavelet_coherence_example.png', dpi=150, bbox_inches='tight')
    print("Figure saved as 'wavelet_coherence_example.png'")

    plt.show()

    return result


def example_with_time_delay():
    """Example with two signals with a known time delay."""

    print("\n" + "=" * 60)
    print("Wavelet Coherence Analysis - Time Delay Example")
    print("=" * 60)

    # Generate time vector
    fs = 200  # sampling frequency (Hz)
    duration = 5  # seconds
    t = np.linspace(0, duration, int(fs * duration))

    # Generate signal 1
    freq = 8  # Hz
    signal1 = np.sin(2 * np.pi * freq * t)

    # Generate signal 2 with a time delay
    time_delay = 0.05  # seconds (50 ms)
    delay_samples = int(time_delay * fs)
    signal2 = np.zeros_like(signal1)
    signal2[delay_samples:] = signal1[:-delay_samples]

    # Add noise
    signal1 += 0.1 * np.random.randn(len(t))
    signal2 += 0.1 * np.random.randn(len(t))

    print(f"\nSignal properties:")
    print(f"  Duration: {duration} seconds")
    print(f"  Sampling rate: {fs} Hz")
    print(f"  Frequency: {freq} Hz")
    print(f"  True time delay: {time_delay*1000:.1f} ms")

    # Compute wavelet coherence
    print("\nComputing wavelet coherence...")
    result = wc.wavelet_coherence(signal1, signal2,
                                   sampling_rate=fs,
                                   compute_phase=True,
                                   compute_delay=True)

    coherence = result['coherence']
    frequencies = result['frequencies']
    delay = result['delay']

    # Find delay at the main frequency
    freq_idx = np.argmin(np.abs(frequencies - freq))
    estimated_delay = np.mean(delay[freq_idx, delay_samples:]) * 1000  # ms

    print(f"\nDelay estimation:")
    print(f"  True delay: {time_delay*1000:.1f} ms")
    print(f"  Estimated delay at {frequencies[freq_idx]:.1f} Hz: {estimated_delay:.1f} ms")
    print(f"  Error: {abs(estimated_delay - time_delay*1000):.1f} ms")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Signals
    axes[0, 0].plot(t[:200], signal1[:200], 'b-', label='Signal 1')
    axes[0, 0].plot(t[:200], signal2[:200], 'r--', label='Signal 2 (delayed)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Signals (first 200 samples)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Coherence
    extent = [t[0], t[-1], frequencies[0], frequencies[-1]]
    im1 = axes[0, 1].imshow(coherence, aspect='auto', origin='lower',
                            extent=extent, cmap='viridis', vmin=0, vmax=1)
    axes[0, 1].axhline(y=freq, color='r', linestyle='--', linewidth=1, alpha=0.5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    axes[0, 1].set_title('Wavelet Coherence')
    axes[0, 1].set_ylim([0, 20])
    plt.colorbar(im1, ax=axes[0, 1], label='Coherence')

    # Delay
    delay_ms = delay * 1000
    delay_clipped = np.clip(delay_ms, -200, 200)
    im2 = axes[1, 0].imshow(delay_clipped, aspect='auto', origin='lower',
                            extent=extent, cmap='RdBu_r', vmin=-100, vmax=100)
    axes[1, 0].axhline(y=freq, color='r', linestyle='--', linewidth=1, alpha=0.5)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_title('Temporal Delay')
    axes[1, 0].set_ylim([0, 20])
    plt.colorbar(im2, ax=axes[1, 0], label='Delay (ms)')

    # Delay at main frequency
    axes[1, 1].plot(t, delay_ms[freq_idx, :], 'b-', linewidth=1)
    axes[1, 1].axhline(y=time_delay*1000, color='r', linestyle='--',
                       linewidth=2, label=f'True delay ({time_delay*1000:.1f} ms)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Delay (ms)')
    axes[1, 1].set_title(f'Delay at {frequencies[freq_idx]:.1f} Hz')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([-100, 100])

    plt.tight_layout()
    plt.savefig('wavelet_coherence_delay_example.png', dpi=150, bbox_inches='tight')
    print("Figure saved as 'wavelet_coherence_delay_example.png'")

    plt.show()

    return result


def example_cross_wavelet_spectrum():
    """Example using cross-wavelet spectrum."""

    print("\n" + "=" * 60)
    print("Cross-Wavelet Spectrum Example")
    print("=" * 60)

    # Generate signals
    fs = 100
    t = np.linspace(0, 10, fs * 10)
    signal1 = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    signal2 = np.sin(2 * np.pi * 5 * t + 0.2) + 0.3 * np.sin(2 * np.pi * 10 * t)

    # Add noise
    signal1 += 0.1 * np.random.randn(len(t))
    signal2 += 0.1 * np.random.randn(len(t))

    print("\nComputing cross-wavelet spectrum...")
    result = wc.cross_wavelet_spectrum(signal1, signal2, sampling_rate=fs)

    cross_spectrum = result['cross_spectrum']
    frequencies = result['frequencies']
    power = result['power']
    phase = result['phase']

    print(f"Cross-spectrum shape: {cross_spectrum.shape}")
    print(f"Maximum power: {power.max():.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    extent = [t[0], t[-1], frequencies[0], frequencies[-1]]

    # Power
    im1 = axes[0].imshow(power, aspect='auto', origin='lower',
                         extent=extent, cmap='hot')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_title('Cross-Wavelet Power')
    axes[0].set_ylim([0, 20])
    plt.colorbar(im1, ax=axes[0], label='Power')

    # Phase
    im2 = axes[1].imshow(phase, aspect='auto', origin='lower',
                         extent=extent, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('Cross-Wavelet Phase')
    axes[1].set_ylim([0, 20])
    plt.colorbar(im2, ax=axes[1], label='Phase (rad)')

    plt.tight_layout()
    plt.savefig('cross_wavelet_spectrum_example.png', dpi=150, bbox_inches='tight')
    print("Figure saved as 'cross_wavelet_spectrum_example.png'")

    plt.show()

    return result


if __name__ == '__main__':
    # Run examples
    print("\n\nRunning Wavelet Coherence Examples\n")

    # Example 1: Synthetic signals with multiple frequencies
    result1 = example_synthetic_signals()

    # Example 2: Signals with time delay
    result2 = example_with_time_delay()

    # Example 3: Cross-wavelet spectrum
    result3 = example_cross_wavelet_spectrum()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
