#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example: Chaos Theory Analysis for Biological Signals

This example demonstrates nonlinear dynamics analysis including:
1. Entropy measures (Shannon, Sample, Approximate, Permutation)
2. Fractal dimension (Higuchi, Petrosian, Katz)
3. DFA (Detrended Fluctuation Analysis)
4. Hurst exponent
5. Lyapunov exponent

"""

import numpy as np
import matplotlib.pyplot as plt
from biosppy import chaos

print("=" * 70)
print("Chaos Theory Analysis for Biological Signals")
print("=" * 70)

# ============================================================================
# Generate Test Signals
# ============================================================================
print("\n" + "=" * 70)
print("1. Generating Test Signals")
print("=" * 70)

np.random.seed(42)
n_points = 2000

# Signal 1: Regular periodic signal (low complexity)
t = np.linspace(0, 10, n_points)
regular_signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
print("\n✓ Generated regular periodic signal")

# Signal 2: Random noise (high complexity, no structure)
random_signal = np.random.randn(n_points)
print("✓ Generated random noise signal")

# Signal 3: Correlated noise (physiological-like)
# Simulates heart rate variability
correlated_signal = np.cumsum(np.random.randn(n_points)) * 0.1
correlated_signal = correlated_signal - np.mean(correlated_signal)
print("✓ Generated correlated noise signal (HRV-like)")

# Signal 4: Chaotic signal (logistic map)
x = [0.1]
r = 3.9  # Chaotic regime
for _ in range(n_points - 1):
    x.append(r * x[-1] * (1 - x[-1]))
chaotic_signal = np.array(x)
print("✓ Generated chaotic signal (Logistic map)")

# ============================================================================
# 2. Shannon Entropy Analysis
# ============================================================================
print("\n" + "=" * 70)
print("2. Shannon Entropy Analysis")
print("=" * 70)
print("\nShannon Entropy (normalized):")

for name, signal in [("Regular", regular_signal),
                     ("Random", random_signal),
                     ("Correlated", correlated_signal),
                     ("Chaotic", chaotic_signal)]:
    result = chaos.shannon_entropy(signal=signal)
    print(f"  {name:12s}: {result['entropy']:.4f}")

print("\nInterpretation: Higher entropy = more complexity/randomness")

# ============================================================================
# 3. Sample Entropy Analysis
# ============================================================================
print("\n" + "=" * 70)
print("3. Sample Entropy Analysis")
print("=" * 70)
print("\nSample Entropy (m=2, r=0.2*std):")

for name, signal in [("Regular", regular_signal),
                     ("Random", random_signal),
                     ("Correlated", correlated_signal),
                     ("Chaotic", chaotic_signal)]:
    result = chaos.sample_entropy(signal=signal, m=2, r=0.2)
    print(f"  {name:12s}: {result['sampen']:.4f}")

print("\nInterpretation: Lower SampEn = more regular/predictable")

# ============================================================================
# 4. Approximate Entropy Analysis
# ============================================================================
print("\n" + "=" * 70)
print("4. Approximate Entropy Analysis")
print("=" * 70)
print("\nApproximate Entropy (m=2, r=0.2*std):")

for name, signal in [("Regular", regular_signal),
                     ("Random", random_signal),
                     ("Correlated", correlated_signal),
                     ("Chaotic", chaotic_signal)]:
    result = chaos.approximate_entropy(signal=signal, m=2, r=0.2)
    print(f"  {name:12s}: {result['apen']:.4f}")

print("\nInterpretation: Similar to Sample Entropy")

# ============================================================================
# 5. Permutation Entropy Analysis
# ============================================================================
print("\n" + "=" * 70)
print("5. Permutation Entropy Analysis")
print("=" * 70)
print("\nPermutation Entropy (order=3):")

for name, signal in [("Regular", regular_signal),
                     ("Random", random_signal),
                     ("Correlated", correlated_signal),
                     ("Chaotic", chaotic_signal)]:
    result = chaos.permutation_entropy(signal=signal, order=3)
    print(f"  {name:12s}: {result['pe']:.4f}")

print("\nInterpretation: Normalized to [0,1], higher = more complex")

# ============================================================================
# 6. Multiscale Entropy Analysis
# ============================================================================
print("\n" + "=" * 70)
print("6. Multiscale Entropy Analysis")
print("=" * 70)
print("\nComputing MSE for correlated signal...")

mse_result = chaos.multiscale_entropy(signal=correlated_signal, max_scale=20)
print(f"✓ Computed MSE for {len(mse_result['scales'])} scales")
print(f"  MSE at scale 1: {mse_result['mse'][0]:.4f}")
print(f"  MSE at scale 10: {mse_result['mse'][9]:.4f}")
print(f"  MSE at scale 20: {mse_result['mse'][19]:.4f}")

# ============================================================================
# 7. Fractal Dimension Analysis
# ============================================================================
print("\n" + "=" * 70)
print("7. Fractal Dimension Analysis")
print("=" * 70)

# Higuchi Fractal Dimension
print("\nHiguchi Fractal Dimension:")
for name, signal in [("Regular", regular_signal),
                     ("Random", random_signal),
                     ("Correlated", correlated_signal),
                     ("Chaotic", chaotic_signal)]:
    result = chaos.higuchi_fd(signal=signal, k_max=10)
    print(f"  {name:12s}: {result['hfd']:.4f}")

print("\nInterpretation: Range [1, 2], higher = more irregular")

# Petrosian Fractal Dimension
print("\nPetrosian Fractal Dimension:")
for name, signal in [("Regular", regular_signal),
                     ("Random", random_signal),
                     ("Correlated", correlated_signal),
                     ("Chaotic", chaotic_signal)]:
    result = chaos.petrosian_fd(signal=signal)
    print(f"  {name:12s}: {result['pfd']:.4f}")

# Katz Fractal Dimension
print("\nKatz Fractal Dimension:")
for name, signal in [("Regular", regular_signal),
                     ("Random", random_signal),
                     ("Correlated", correlated_signal),
                     ("Chaotic", chaotic_signal)]:
    result = chaos.katz_fd(signal=signal)
    print(f"  {name:12s}: {result['kfd']:.4f}")

# ============================================================================
# 8. DFA Analysis
# ============================================================================
print("\n" + "=" * 70)
print("8. Detrended Fluctuation Analysis (DFA)")
print("=" * 70)
print("\nDFA Scaling Exponent (α):")

for name, signal in [("Regular", regular_signal),
                     ("Random", random_signal),
                     ("Correlated", correlated_signal),
                     ("Chaotic", chaotic_signal)]:
    result = chaos.dfa(signal=signal)
    print(f"  {name:12s}: α = {result['alpha']:.4f}")

print("\nInterpretation:")
print("  α < 0.5  : Anti-correlated")
print("  α = 0.5  : White noise (uncorrelated)")
print("  α = 1.0  : 1/f noise (pink noise)")
print("  α > 1.0  : Non-stationary")

# ============================================================================
# 9. Hurst Exponent Analysis
# ============================================================================
print("\n" + "=" * 70)
print("9. Hurst Exponent Analysis")
print("=" * 70)
print("\nHurst Exponent (H):")

for name, signal in [("Regular", regular_signal),
                     ("Random", random_signal),
                     ("Correlated", correlated_signal),
                     ("Chaotic", chaotic_signal)]:
    result = chaos.hurst_exponent(signal=signal)
    print(f"  {name:12s}: H = {result['H']:.4f}")

print("\nInterpretation:")
print("  H < 0.5  : Mean-reverting (anti-persistent)")
print("  H = 0.5  : Random walk")
print("  H > 0.5  : Trending (persistent)")

# ============================================================================
# 10. Lyapunov Exponent Analysis
# ============================================================================
print("\n" + "=" * 70)
print("10. Lyapunov Exponent Analysis")
print("=" * 70)
print("\nLargest Lyapunov Exponent (λ):")

for name, signal in [("Regular", regular_signal),
                     ("Random", random_signal),
                     ("Chaotic", chaotic_signal)]:
    try:
        result = chaos.lyapunov_exponent(signal=signal, emb_dim=10)
        print(f"  {name:12s}: λ = {result['lambda_max']:.4f}")
    except Exception as e:
        print(f"  {name:12s}: Could not compute")

print("\nInterpretation:")
print("  λ > 0  : Chaotic (exponential divergence)")
print("  λ = 0  : Periodic/quasi-periodic")
print("  λ < 0  : Stable fixed point")

# ============================================================================
# 11. Visualization
# ============================================================================
print("\n" + "=" * 70)
print("11. Generating Visualizations")
print("=" * 70)

fig = plt.figure(figsize=(16, 12))

# Plot signals
signals = [
    ("Regular Signal", regular_signal),
    ("Random Signal", random_signal),
    ("Correlated Signal", correlated_signal),
    ("Chaotic Signal", chaotic_signal)
]

for idx, (name, sig) in enumerate(signals, 1):
    ax = plt.subplot(4, 3, (idx - 1) * 3 + 1)
    ax.plot(sig[:500], linewidth=0.8)
    ax.set_title(name)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)

# Plot Multiscale Entropy for correlated signal
ax = plt.subplot(4, 3, 2)
ax.plot(mse_result['scales'], mse_result['mse'], marker='o', linewidth=2)
ax.set_xlabel('Scale')
ax.set_ylabel('Sample Entropy')
ax.set_title('Multiscale Entropy\n(Correlated Signal)')
ax.grid(True, alpha=0.3)

# Plot DFA for correlated signal
dfa_corr = chaos.dfa(signal=correlated_signal)
ax = plt.subplot(4, 3, 3)
ax.loglog(dfa_corr['window_sizes'], dfa_corr['fluctuations'], 'o-', linewidth=2)
ax.set_xlabel('Window Size (log)')
ax.set_ylabel('Fluctuation F(n) (log)')
ax.set_title(f'DFA Analysis\nα = {dfa_corr["alpha"]:.3f}')
ax.grid(True, alpha=0.3)

# Comparison bar plots
ax = plt.subplot(4, 3, 5)
names = ['Regular', 'Random', 'Corr.', 'Chaotic']
shannon_vals = [chaos.shannon_entropy(signal=s)['entropy']
                for _, s in signals]
ax.bar(names, shannon_vals, color='steelblue', alpha=0.7)
ax.set_ylabel('Shannon Entropy')
ax.set_title('Shannon Entropy Comparison')
ax.grid(True, alpha=0.3, axis='y')

ax = plt.subplot(4, 3, 6)
sampen_vals = [chaos.sample_entropy(signal=s)['sampen']
               for _, s in signals]
ax.bar(names, sampen_vals, color='coral', alpha=0.7)
ax.set_ylabel('Sample Entropy')
ax.set_title('Sample Entropy Comparison')
ax.grid(True, alpha=0.3, axis='y')

ax = plt.subplot(4, 3, 8)
higuchi_vals = [chaos.higuchi_fd(signal=s)['hfd']
                for _, s in signals]
ax.bar(names, higuchi_vals, color='seagreen', alpha=0.7)
ax.set_ylabel('Higuchi FD')
ax.set_title('Higuchi Fractal Dimension')
ax.grid(True, alpha=0.3, axis='y')

ax = plt.subplot(4, 3, 9)
dfa_vals = [chaos.dfa(signal=s)['alpha']
            for _, s in signals]
ax.bar(names, dfa_vals, color='mediumpurple', alpha=0.7)
ax.axhline(y=0.5, color='r', linestyle='--', label='White noise')
ax.axhline(y=1.0, color='orange', linestyle='--', label='Pink noise')
ax.set_ylabel('DFA Exponent (α)')
ax.set_title('DFA Scaling Exponent')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

ax = plt.subplot(4, 3, 11)
hurst_vals = [chaos.hurst_exponent(signal=s)['H']
              for _, s in signals]
ax.bar(names, hurst_vals, color='tomato', alpha=0.7)
ax.axhline(y=0.5, color='r', linestyle='--', label='Random walk')
ax.set_ylabel('Hurst Exponent (H)')
ax.set_title('Hurst Exponent')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

ax = plt.subplot(4, 3, 12)
pe_vals = [chaos.permutation_entropy(signal=s)['pe']
           for _, s in signals]
ax.bar(names, pe_vals, color='gold', alpha=0.7)
ax.set_ylabel('Permutation Entropy')
ax.set_title('Permutation Entropy')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('chaos_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved as 'chaos_analysis.png'")

print("\n" + "=" * 70)
print("Chaos Analysis Complete!")
print("=" * 70)
print("\nKey Findings:")
print("  • Regular signals show low entropy and low fractal dimension")
print("  • Random signals show high entropy but no long-range correlations")
print("  • Correlated signals show intermediate complexity with structure")
print("  • Chaotic signals show high complexity with deterministic structure")
print("\nApplications in Biosignals:")
print("  • HRV analysis: DFA, Hurst, Sample Entropy")
print("  • EEG seizure detection: Higuchi FD, Lyapunov exponent")
print("  • Signal quality assessment: Shannon entropy")
print("  • Complexity quantification: Multiscale entropy")
print("=" * 70)
