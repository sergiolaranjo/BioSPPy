#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example: Baroreflex Sensitivity Analysis using Sequential Method
==================================================================

This example demonstrates how to use the baroreflex module to compute
baroreflex sensitivity (BRS) using the sequential method by Di Rienzo et al.

The example shows both single-lag and multi-lag analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from biosppy.signals import baroreflex


def generate_synthetic_data(n_beats=200, brs_true=10.0, noise_level=0.1):
    """Generate synthetic SBP and RRI data with baroreflex coupling.

    Parameters
    ----------
    n_beats : int
        Number of cardiac cycles to generate.
    brs_true : float
        True baroreflex sensitivity (ms/mmHg).
    noise_level : float
        Amount of noise to add (0-1).

    Returns
    -------
    sbp : array
        Synthetic systolic blood pressure (mmHg).
    rri : array
        Synthetic RR intervals (ms).
    """
    # Generate baseline SBP signal with slow oscillations
    t = np.linspace(0, n_beats * 0.8, n_beats)  # time in seconds
    sbp_baseline = 120  # mmHg

    # Add Mayer wave oscillation (~0.1 Hz) and respiratory oscillation (~0.25 Hz)
    sbp = sbp_baseline + 5 * np.sin(2 * np.pi * 0.1 * t) + 3 * np.sin(2 * np.pi * 0.25 * t)

    # Add random variations
    sbp += np.random.randn(n_beats) * 2

    # Generate RRI based on SBP with baroreflex coupling
    rri_baseline = 850  # ms
    rri = rri_baseline + brs_true * (sbp - sbp_baseline)

    # Add noise to RRI
    rri += np.random.randn(n_beats) * (noise_level * 20)

    return sbp, rri


def main():
    """Main example function."""

    print("=" * 70)
    print("Baroreflex Sensitivity Analysis - Sequential Method")
    print("=" * 70)
    print()

    # Generate synthetic data
    print("Generating synthetic data...")
    np.random.seed(42)  # For reproducibility
    n_beats = 300
    brs_true = 12.0  # True BRS in ms/mmHg
    sbp, rri = generate_synthetic_data(n_beats=n_beats, brs_true=brs_true, noise_level=0.15)

    print(f"  - Number of beats: {n_beats}")
    print(f"  - True BRS: {brs_true:.2f} ms/mmHg")
    print(f"  - SBP range: {sbp.min():.1f} - {sbp.max():.1f} mmHg")
    print(f"  - RRI range: {rri.min():.1f} - {rri.max():.1f} ms")
    print()

    # Example 1: Single-lag analysis (lag = 0)
    print("-" * 70)
    print("Example 1: Single-lag analysis (lag = 0)")
    print("-" * 70)

    result = baroreflex.sequential_method(
        sbp=sbp,
        rri=rri,
        threshold_sbp=1.0,
        threshold_rri=5.0,
        min_sequence_length=3,
        max_lag=0,
        correlation_threshold=0.85
    )

    print(f"Baroreflex Sensitivity (BRS): {result['brs']:.2f} ms/mmHg")
    print(f"Number of up-up sequences: {result['n_sequences_up']}")
    print(f"Number of down-down sequences: {result['n_sequences_down']}")
    print(f"Total number of sequences: {result['n_sequences_total']}")
    print()

    # Show details of first 3 sequences
    if len(result['sequences_info']) > 0:
        print("First 3 sequence details:")
        for i, seq in enumerate(result['sequences_info'][:3]):
            print(f"  Sequence {i+1}:")
            print(f"    - Type: {seq['type']}")
            print(f"    - Length: {seq['length']} beats")
            print(f"    - Slope (BRS): {seq['slope']:.2f} ms/mmHg")
            print(f"    - Correlation: {seq['correlation']:.3f}")
            print(f"    - P-value: {seq['p_value']:.4f}")
    print()

    # Example 2: Multi-lag analysis (lag = 0, 1, 2, 3)
    print("-" * 70)
    print("Example 2: Multi-lag analysis (lag = 0, 1, 2, 3)")
    print("-" * 70)

    result_multilag = baroreflex.sequential_method(
        sbp=sbp,
        rri=rri,
        threshold_sbp=1.0,
        threshold_rri=5.0,
        min_sequence_length=3,
        max_lag=3,
        correlation_threshold=0.85
    )

    print(f"{'Lag':<6} {'BRS (ms/mmHg)':<16} {'N_up':<8} {'N_down':<8} {'N_total':<8}")
    print("-" * 50)
    for lag in range(4):
        print(f"{lag:<6} {result_multilag['brs'][lag]:<16.2f} "
              f"{result_multilag['n_sequences_up'][lag]:<8} "
              f"{result_multilag['n_sequences_down'][lag]:<8} "
              f"{result_multilag['n_sequences_total'][lag]:<8}")
    print()

    # Example 3: Baroreflex Effectiveness Index
    print("-" * 70)
    print("Example 3: Baroreflex Effectiveness Index (BEI)")
    print("-" * 70)

    bei_result = baroreflex.baroreflex_effectiveness_index(
        sbp=sbp,
        rri=rri,
        threshold_sbp=1.0,
        threshold_rri=5.0
    )

    print(f"Baroreflex Effectiveness Index (BEI): {bei_result['bei']:.1f}%")
    print(f"Total SBP ramps detected: {bei_result['n_sbp_ramps']}")
    print(f"Effective ramps (with RRI response): {bei_result['n_effective_ramps']}")
    print()

    # Visualization
    print("-" * 70)
    print("Generating visualizations...")
    print("-" * 70)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: SBP and RRI time series
    beat_numbers = np.arange(len(sbp))
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    line1 = ax1.plot(beat_numbers, sbp, 'b-', label='SBP', linewidth=1)
    line2 = ax1_twin.plot(beat_numbers, rri, 'r-', label='RRI', linewidth=1)

    ax1.set_xlabel('Beat Number')
    ax1.set_ylabel('SBP (mmHg)', color='b')
    ax1_twin.set_ylabel('RRI (ms)', color='r')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    ax1.set_title('Systolic Blood Pressure and RR Intervals')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: SBP vs RRI scatter plot with sequences
    ax2 = axes[1]
    ax2.scatter(sbp, rri, alpha=0.3, s=20, label='All beats')

    # Highlight sequences
    colors = {'up': 'green', 'down': 'orange'}
    for seq in result['sequences_info'][:10]:  # Show first 10 sequences
        idx_start = seq['start_idx']
        idx_end = seq['end_idx'] + 1
        sbp_seq = sbp[idx_start:idx_end]
        rri_seq = rri[idx_start:idx_end]
        ax2.plot(sbp_seq, rri_seq, 'o-', color=colors[seq['type']],
                alpha=0.7, linewidth=2, markersize=6)

    ax2.set_xlabel('SBP (mmHg)')
    ax2.set_ylabel('RRI (ms)')
    ax2.set_title(f'SBP-RRI Relationship (BRS = {result["brs"]:.2f} ms/mmHg)')
    ax2.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
               markersize=8, label='Up sequences'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
               markersize=8, label='Down sequences')
    ]
    ax2.legend(handles=legend_elements, loc='best')

    # Plot 3: BRS vs Lag
    ax3 = axes[2]
    lags = np.arange(len(result_multilag['brs']))
    ax3.plot(lags, result_multilag['brs'], 'o-', linewidth=2, markersize=8)
    ax3.axhline(y=brs_true, color='r', linestyle='--', label=f'True BRS = {brs_true:.2f}')
    ax3.set_xlabel('Lag (beats)')
    ax3.set_ylabel('BRS (ms/mmHg)')
    ax3.set_title('Baroreflex Sensitivity vs Lag')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xticks(lags)

    plt.tight_layout()
    plt.savefig('baroreflex_analysis.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'baroreflex_analysis.png'")
    plt.show()

    print()
    print("=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
