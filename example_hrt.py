#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example script demonstrating Heart Rate Turbulence (HRT) analysis.

This script shows how to use the heart_rate_turbulence function to analyze
the autonomic response to ventricular premature complexes (VPCs).
"""

import numpy as np
from biosppy.signals import hrv, ecg

def example_hrt_with_synthetic_data():
    """Example using synthetic RR intervals with simulated VPCs."""
    print("Example 1: Heart Rate Turbulence with synthetic data")
    print("=" * 60)

    # Create synthetic RR intervals (normal sinus rhythm around 800ms)
    np.random.seed(42)
    n_intervals = 200
    baseline_rr = 800  # ms

    # Generate normal RR intervals with some variability
    rri = baseline_rr + np.random.randn(n_intervals) * 30

    # Add some simulated VPCs at specific positions
    vpc_positions = [50, 100, 150]

    for vpc_pos in vpc_positions:
        if vpc_pos < len(rri) - 1:
            # Simulate premature beat (shorter interval)
            rri[vpc_pos] = baseline_rr * 0.7  # 30% shorter
            # Simulate compensatory pause (longer interval)
            rri[vpc_pos + 1] = baseline_rr * 1.3  # 30% longer

    # Compute HRT parameters
    print("\nComputing HRT parameters...")
    hrt_results = hrv.heart_rate_turbulence(rri=rri, show=False)

    print(f"\nResults:")
    print(f"  Turbulence Onset (TO): {hrt_results['to']:.2f}%")
    print(f"  Turbulence Slope (TS): {hrt_results['ts']:.2f} ms/RR")
    print(f"  Number of VPCs detected: {hrt_results['vpc_count']}")
    print(f"  VPC indices: {hrt_results['vpc_indices']}")

    print("\nInterpretation:")
    if hrt_results['to'] < 0:
        print("  TO < 0%: Normal (heart rate accelerates after VPC)")
    else:
        print("  TO >= 0%: Abnormal (reduced heart rate acceleration)")

    if hrt_results['ts'] > 2.5:
        print("  TS > 2.5 ms/RR: Normal (adequate heart rate deceleration)")
    else:
        print("  TS <= 2.5 ms/RR: Abnormal (reduced heart rate deceleration)")

    print("\n" + "=" * 60 + "\n")

    return hrt_results


def example_hrt_with_ecg_data():
    """Example using real ECG data (if available)."""
    print("Example 2: Heart Rate Turbulence with ECG data")
    print("=" * 60)

    try:
        # Load sample ECG data from biosppy
        from biosppy import storage
        signal, mdata = storage.load_txt('./examples/ecg.txt')
        sampling_rate = mdata['sampling_rate']

        # Process ECG to extract R-peaks
        print("\nProcessing ECG signal...")
        ecg_results = ecg.ecg(signal=signal, sampling_rate=sampling_rate, show=False)

        # Compute RR intervals
        rri = hrv.compute_rri(rpeaks=ecg_results['rpeaks'],
                             sampling_rate=sampling_rate,
                             filter_rri=False)

        # Compute HRT parameters
        print("Computing HRT parameters...")
        hrt_results = hrv.heart_rate_turbulence(rri=rri, show=False)

        print(f"\nResults:")
        print(f"  Turbulence Onset (TO): {hrt_results['to']:.2f}%")
        print(f"  Turbulence Slope (TS): {hrt_results['ts']:.2f} ms/RR")
        print(f"  Number of VPCs detected: {hrt_results['vpc_count']}")

        if hrt_results['vpc_count'] > 0:
            print(f"  VPC indices: {hrt_results['vpc_indices']}")
        else:
            print("  No VPCs detected in this ECG segment")

    except Exception as e:
        print(f"\nCould not load ECG data: {e}")
        print("This example requires sample ECG data file.")

    print("\n" + "=" * 60 + "\n")


def example_hrt_with_manual_vpc_indices():
    """Example specifying VPC locations manually."""
    print("Example 3: Heart Rate Turbulence with manual VPC specification")
    print("=" * 60)

    # Create synthetic RR intervals
    np.random.seed(123)
    n_intervals = 150
    baseline_rr = 850  # ms
    rri = baseline_rr + np.random.randn(n_intervals) * 25

    # Manually specify VPC locations
    vpc_indices = [40, 80, 120]

    # Add VPC patterns at specified locations
    for vpc_idx in vpc_indices:
        if vpc_idx < len(rri) - 1:
            rri[vpc_idx] = baseline_rr * 0.65
            rri[vpc_idx + 1] = baseline_rr * 1.35

    print(f"\nManually specified VPC indices: {vpc_indices}")

    # Compute HRT with manual VPC indices
    print("Computing HRT parameters with manual VPC indices...")
    hrt_results = hrv.heart_rate_turbulence(rri=rri,
                                           vpc_indices=vpc_indices,
                                           show=False)

    print(f"\nResults:")
    print(f"  Turbulence Onset (TO): {hrt_results['to']:.2f}%")
    print(f"  Turbulence Slope (TS): {hrt_results['ts']:.2f} ms/RR")
    print(f"  Number of VPCs used: {hrt_results['vpc_count']}")

    print("\n" + "=" * 60 + "\n")

    return hrt_results


if __name__ == '__main__':
    print("\n")
    print("=" * 60)
    print("Heart Rate Turbulence (HRT) Analysis Examples")
    print("=" * 60)
    print("\n")

    # Run examples
    example_hrt_with_synthetic_data()
    example_hrt_with_manual_vpc_indices()
    example_hrt_with_ecg_data()

    print("Examples completed!")
