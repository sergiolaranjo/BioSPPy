# -*- coding: utf-8 -*-
"""
biosppy.chaos
-------------

This module provides chaos theory and nonlinear dynamics analysis methods for
biological signals, including entropy measures and fractal dimension estimators.

:copyright: (c) 2015-2025 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
from six.moves import range

# 3rd party
import numpy as np
from scipy import signal as sp_signal
from scipy.spatial.distance import pdist, squareform

# local
from . import utils


def shannon_entropy(signal=None, normalize=True):
    """Calculate Shannon entropy of a signal.

    Shannon entropy measures the average information content or uncertainty
    in a signal. Higher entropy indicates more randomness/complexity.

    Parameters
    ----------
    signal : array
        Input signal.
    normalize : bool, optional
        If True, normalize entropy to [0, 1] range. Default is True.

    Returns
    -------
    entropy : float
        Shannon entropy value.

    Notes
    -----
    * Shannon entropy: H(X) = -sum(p(x) * log2(p(x)))
    * Used to quantify signal complexity and information content.
    * Normalized entropy allows comparison across signals of different lengths.

    Examples
    --------
    >>> import numpy as np
    >>> from biosppy import chaos
    >>> # Regular signal (low entropy)
    >>> regular = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000))
    >>> # Random signal (high entropy)
    >>> random = np.random.randn(1000)
    >>> print("Regular:", chaos.shannon_entropy(signal=regular)['entropy'])
    >>> print("Random:", chaos.shannon_entropy(signal=random)['entropy'])

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal).flatten()

    # compute histogram
    hist, _ = np.histogram(signal, bins='auto', density=True)
    hist = hist[hist > 0]  # remove zero bins

    # normalize to get probabilities
    prob = hist / np.sum(hist)

    # compute Shannon entropy
    entropy = -np.sum(prob * np.log2(prob))

    # normalize if requested
    if normalize and len(prob) > 1:
        max_entropy = np.log2(len(prob))
        entropy = entropy / max_entropy

    return utils.ReturnTuple((entropy,), ('entropy',))


def sample_entropy(signal=None, m=2, r=None, scale=True):
    """Calculate Sample Entropy (SampEn) of a signal.

    Sample Entropy is a modification of Approximate Entropy that measures the
    regularity and complexity of time series data. It's less biased and more
    consistent than Approximate Entropy.

    Parameters
    ----------
    signal : array
        Input signal.
    m : int, optional
        Embedding dimension (pattern length). Default is 2.
    r : float, optional
        Tolerance for matching patterns. If None, defaults to 0.2 * std(signal).
    scale : bool, optional
        If True, scale r relative to signal std. Default is True.

    Returns
    -------
    sampen : float
        Sample Entropy value.
    m : int
        Embedding dimension used.
    r : float
        Tolerance used.

    Notes
    -----
    * Lower SampEn values indicate more regularity (predictable patterns).
    * Higher SampEn values indicate more complexity (less predictable).
    * Common values: m=2, r=0.1-0.25 * std(signal).
    * Requires at least 10^m to 30^m data points for reliable estimates.

    References
    ----------
    Richman JS, Moorman JR. Physiological time-series analysis using approximate
    entropy and sample entropy. Am J Physiol Heart Circ Physiol. 2000;278(6):H2039-49.

    Examples
    --------
    >>> import numpy as np
    >>> from biosppy import chaos
    >>> # Regular signal (low entropy)
    >>> regular = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000))
    >>> # Random signal (high entropy)
    >>> random = np.random.randn(1000)
    >>> print("Regular:", chaos.sample_entropy(signal=regular)['sampen'])
    >>> print("Random:", chaos.sample_entropy(signal=random)['sampen'])

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal).flatten()
    N = len(signal)

    # set tolerance
    if r is None:
        r = 0.2 * np.std(signal, ddof=1)
    elif scale:
        r = r * np.std(signal, ddof=1)

    # template matching
    def _maxdist(xi, xj):
        """Maximum absolute distance between two vectors."""
        return np.max(np.abs(xi - xj))

    def _phi(m):
        """Count template matches."""
        patterns = np.array([signal[i:i+m] for i in range(N - m + 1)])
        count = 0
        for i in range(len(patterns) - 1):
            # compare pattern i with all subsequent patterns
            dists = np.array([_maxdist(patterns[i], patterns[j])
                            for j in range(i + 1, len(patterns))])
            count += np.sum(dists <= r)
        return count

    # compute phi for m and m+1
    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)

    # compute sample entropy
    if phi_m == 0 or phi_m1 == 0:
        sampen = np.inf
    else:
        sampen = -np.log(phi_m1 / phi_m)

    return utils.ReturnTuple((sampen, m, r), ('sampen', 'm', 'r'))


def approximate_entropy(signal=None, m=2, r=None, scale=True):
    """Calculate Approximate Entropy (ApEn) of a signal.

    Approximate Entropy quantifies the regularity and unpredictability of
    fluctuations in a time series. It's similar to Sample Entropy but includes
    self-matches.

    Parameters
    ----------
    signal : array
        Input signal.
    m : int, optional
        Embedding dimension (pattern length). Default is 2.
    r : float, optional
        Tolerance for matching patterns. If None, defaults to 0.2 * std(signal).
    scale : bool, optional
        If True, scale r relative to signal std. Default is True.

    Returns
    -------
    apen : float
        Approximate Entropy value.
    m : int
        Embedding dimension used.
    r : float
        Tolerance used.

    Notes
    -----
    * Lower ApEn values indicate more regularity.
    * Higher ApEn values indicate more complexity.
    * ApEn includes self-matches, making it slightly biased.
    * Sample Entropy (SampEn) is generally preferred over ApEn.

    References
    ----------
    Pincus SM. Approximate entropy as a measure of system complexity.
    Proc Natl Acad Sci USA. 1991;88(6):2297-301.

    Examples
    --------
    >>> import numpy as np
    >>> from biosppy import chaos
    >>> signal = np.random.randn(1000)
    >>> result = chaos.approximate_entropy(signal=signal, m=2, r=0.2)

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal).flatten()
    N = len(signal)

    # set tolerance
    if r is None:
        r = 0.2 * np.std(signal, ddof=1)
    elif scale:
        r = r * np.std(signal, ddof=1)

    def _maxdist(xi, xj):
        """Maximum absolute distance between two vectors."""
        return np.max(np.abs(xi - xj))

    def _phi(m):
        """Compute phi function."""
        patterns = np.array([signal[i:i+m] for i in range(N - m + 1)])
        phi_values = []

        for i in range(len(patterns)):
            # count matches (including self)
            matches = 0
            for j in range(len(patterns)):
                if _maxdist(patterns[i], patterns[j]) <= r:
                    matches += 1
            phi_values.append(matches / len(patterns))

        return np.mean(np.log(phi_values))

    # compute ApEn
    apen = _phi(m) - _phi(m + 1)

    return utils.ReturnTuple((apen, m, r), ('apen', 'm', 'r'))


def multiscale_entropy(signal=None, m=2, r=None, max_scale=20, method='sample'):
    """Calculate Multiscale Entropy (MSE) of a signal.

    Multiscale Entropy evaluates the complexity of a signal across multiple
    time scales. It's particularly useful for analyzing physiological signals.

    Parameters
    ----------
    signal : array
        Input signal.
    m : int, optional
        Embedding dimension. Default is 2.
    r : float, optional
        Tolerance for matching. If None, defaults to 0.2 * std(signal).
    max_scale : int, optional
        Maximum scale factor. Default is 20.
    method : str, optional
        Entropy method: 'sample' or 'approximate'. Default is 'sample'.

    Returns
    -------
    mse : array
        Multiscale entropy values for each scale.
    scales : array
        Scale factors used.

    Notes
    -----
    * MSE(scale=1) is equivalent to regular Sample/Approximate Entropy.
    * For each scale τ, signal is coarse-grained: y_j^(τ) = (1/τ) * sum(x_i)
    * Useful for distinguishing between complex and random signals.
    * Complex signals show high entropy across multiple scales.
    * Random signals show high entropy only at small scales.

    References
    ----------
    Costa M, Goldberger AL, Peng CK. Multiscale entropy analysis of complex
    physiologic time series. Phys Rev Lett. 2002;89(6):068102.

    Examples
    --------
    >>> import numpy as np
    >>> from biosppy import chaos
    >>> signal = np.random.randn(5000)
    >>> result = chaos.multiscale_entropy(signal=signal, max_scale=20)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(result['scales'], result['mse'])

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal).flatten()

    if method not in ['sample', 'approximate']:
        raise ValueError("Method must be 'sample' or 'approximate'.")

    # select entropy function
    if method == 'sample':
        entropy_fn = sample_entropy
    else:
        entropy_fn = approximate_entropy

    # compute entropy at each scale
    mse_values = []
    scales = np.arange(1, max_scale + 1)

    for scale in scales:
        # coarse-grain the signal
        if scale == 1:
            coarse_signal = signal
        else:
            n_segments = len(signal) // scale
            coarse_signal = np.mean(
                signal[:n_segments * scale].reshape(n_segments, scale),
                axis=1
            )

        # compute entropy
        try:
            result = entropy_fn(signal=coarse_signal, m=m, r=r, scale=False)
            if method == 'sample':
                entropy_val = result['sampen']
            else:
                entropy_val = result['apen']

            # handle infinite values
            if np.isinf(entropy_val):
                entropy_val = np.nan

            mse_values.append(entropy_val)
        except Exception:
            mse_values.append(np.nan)

    mse_values = np.array(mse_values)

    return utils.ReturnTuple((mse_values, scales), ('mse', 'scales'))


def permutation_entropy(signal=None, order=3, delay=1, normalize=True):
    """Calculate Permutation Entropy of a signal.

    Permutation Entropy measures the complexity of a time series by analyzing
    the order patterns in the signal. It's robust to noise and computationally
    efficient.

    Parameters
    ----------
    signal : array
        Input signal.
    order : int, optional
        Order of permutation pattern (embedding dimension). Default is 3.
    delay : int, optional
        Time delay for embedding. Default is 1.
    normalize : bool, optional
        If True, normalize to [0, 1] range. Default is True.

    Returns
    -------
    pe : float
        Permutation Entropy value.
    order : int
        Order used.

    Notes
    -----
    * PE values close to 0 indicate regular/deterministic signals.
    * PE values close to 1 indicate random/complex signals.
    * Typical order values: 3-7 (higher orders require longer signals).
    * Robust to noise and outliers.
    * Fast computation compared to other entropy measures.

    References
    ----------
    Bandt C, Pompe B. Permutation entropy: a natural complexity measure for
    time series. Phys Rev Lett. 2002;88(17):174102.

    Examples
    --------
    >>> import numpy as np
    >>> from biosppy import chaos
    >>> # Regular signal
    >>> regular = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000))
    >>> # Random signal
    >>> random = np.random.randn(1000)
    >>> print("Regular:", chaos.permutation_entropy(signal=regular)['pe'])
    >>> print("Random:", chaos.permutation_entropy(signal=random)['pe'])

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal).flatten()
    N = len(signal)

    # check if we have enough data
    if N < order * delay:
        raise ValueError("Signal too short for given order and delay.")

    # create permutation patterns
    permutations = {}
    for i in range(N - delay * (order - 1)):
        # extract values
        pattern = signal[i:i + delay * order:delay]
        # get permutation (ranking)
        perm = tuple(np.argsort(pattern))
        # count occurrences
        permutations[perm] = permutations.get(perm, 0) + 1

    # compute probabilities
    n_patterns = sum(permutations.values())
    probabilities = np.array([count / n_patterns for count in permutations.values()])

    # compute permutation entropy
    pe = -np.sum(probabilities * np.log2(probabilities))

    # normalize if requested
    if normalize:
        import math
        max_pe = np.log2(float(math.factorial(order)))
        pe = pe / max_pe

    return utils.ReturnTuple((pe, order), ('pe', 'order'))


def dfa(signal=None, min_win=4, max_win=None, n_wins=10):
    """Detrended Fluctuation Analysis (DFA) for fractal scaling.

    DFA quantifies the self-similarity and long-range correlations in a
    non-stationary time series. It's widely used for analyzing heart rate
    variability and other physiological signals.

    Parameters
    ----------
    signal : array
        Input signal.
    min_win : int, optional
        Minimum window size. Default is 4.
    max_win : int, optional
        Maximum window size. If None, uses len(signal) // 4.
    n_wins : int, optional
        Number of window sizes to test. Default is 10.

    Returns
    -------
    alpha : float
        DFA scaling exponent (Hurst-like exponent).
    fluctuations : array
        Fluctuation values F(n) for each window size.
    window_sizes : array
        Window sizes used in the analysis.

    Notes
    -----
    * α < 0.5: Anti-correlated signal
    * α = 0.5: White noise (uncorrelated)
    * 0.5 < α < 1.0: Correlated signal (long-range correlations)
    * α = 1.0: 1/f noise (pink noise)
    * α > 1.0: Non-stationary, strongly correlated
    * For HRV: healthy individuals typically show α ≈ 1.0

    References
    ----------
    Peng CK, Havlin S, Stanley HE, Goldberger AL. Quantification of scaling
    exponents and crossover phenomena in nonstationary heartbeat time series.
    Chaos. 1995;5(1):82-87.

    Examples
    --------
    >>> import numpy as np
    >>> from biosppy import chaos
    >>> # Generate correlated signal (fractional Brownian motion)
    >>> signal = np.cumsum(np.random.randn(1000))
    >>> result = chaos.dfa(signal=signal)
    >>> print("DFA exponent:", result['alpha'])

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal).flatten()
    N = len(signal)

    # set maximum window size
    if max_win is None:
        max_win = N // 4

    # create window sizes (logarithmically spaced)
    window_sizes = np.unique(
        np.logspace(np.log10(min_win), np.log10(max_win), n_wins, dtype=int)
    )

    # integrate the signal (cumulative sum of deviations from mean)
    y = np.cumsum(signal - np.mean(signal))

    # compute fluctuations for each window size
    fluctuations = []

    for win_size in window_sizes:
        # divide signal into non-overlapping segments
        n_segments = N // win_size

        # forward segments
        f_forward = np.zeros(n_segments)
        for i in range(n_segments):
            segment = y[i * win_size:(i + 1) * win_size]
            # fit polynomial (linear trend)
            x = np.arange(len(segment))
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            # compute fluctuation
            f_forward[i] = np.sqrt(np.mean((segment - trend) ** 2))

        # backward segments (to use remaining data)
        f_backward = np.zeros(n_segments)
        for i in range(n_segments):
            segment = y[N - (i + 1) * win_size:N - i * win_size]
            x = np.arange(len(segment))
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            f_backward[i] = np.sqrt(np.mean((segment - trend) ** 2))

        # average fluctuation
        fluctuation = np.sqrt(
            (np.sum(f_forward ** 2) + np.sum(f_backward ** 2)) / (2 * n_segments)
        )
        fluctuations.append(fluctuation)

    fluctuations = np.array(fluctuations)

    # compute scaling exponent alpha (slope in log-log plot)
    coeffs = np.polyfit(np.log10(window_sizes), np.log10(fluctuations), 1)
    alpha = coeffs[0]

    return utils.ReturnTuple(
        (alpha, fluctuations, window_sizes),
        ('alpha', 'fluctuations', 'window_sizes')
    )


def higuchi_fd(signal=None, k_max=10):
    """Higuchi Fractal Dimension for signal complexity.

    The Higuchi method estimates the fractal dimension of a time series
    directly in the time domain, providing a measure of signal complexity
    and self-similarity.

    Parameters
    ----------
    signal : array
        Input signal.
    k_max : int, optional
        Maximum interval time. Default is 10.

    Returns
    -------
    hfd : float
        Higuchi Fractal Dimension.
    k_values : array
        k values used.
    lengths : array
        Curve lengths for each k.

    Notes
    -----
    * HFD ranges from 1 (smooth, regular signal) to 2 (highly irregular, noise-like).
    * Values close to 1 indicate regular, predictable signals.
    * Values close to 2 indicate complex, irregular signals.
    * Commonly used for EEG analysis and seizure detection.

    References
    ----------
    Higuchi T. Approach to an irregular time series on the basis of the fractal
    theory. Physica D. 1988;31(2):277-283.

    Examples
    --------
    >>> import numpy as np
    >>> from biosppy import chaos
    >>> # Regular signal
    >>> regular = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000))
    >>> # Noisy signal
    >>> noisy = regular + 0.5 * np.random.randn(1000)
    >>> print("Regular:", chaos.higuchi_fd(signal=regular)['hfd'])
    >>> print("Noisy:", chaos.higuchi_fd(signal=noisy)['hfd'])

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal).flatten()
    N = len(signal)

    # compute curve length for each k
    k_values = np.arange(1, k_max + 1)
    lengths = np.zeros(k_max)

    for k in k_values:
        Lk = []
        for m in range(k):
            # construct sub-sequence
            Lm = 0
            indices = np.arange(m, N, k)
            if len(indices) < 2:
                continue

            for i in range(len(indices) - 1):
                Lm += np.abs(signal[indices[i + 1]] - signal[indices[i]])

            # normalize
            Lm = Lm * (N - 1) / (((len(indices) - 1) * k))
            Lk.append(Lm)

        lengths[k - 1] = np.mean(Lk) if Lk else 0

    # compute fractal dimension (slope in log-log plot)
    # Remove zero values
    valid_mask = lengths > 0
    k_valid = k_values[valid_mask]
    lengths_valid = lengths[valid_mask]

    if len(k_valid) < 2:
        raise ValueError("Not enough valid data points to compute fractal dimension.")

    coeffs = np.polyfit(np.log(k_valid), np.log(lengths_valid), 1)
    hfd = -coeffs[0]  # negative slope is the fractal dimension

    return utils.ReturnTuple((hfd, k_values, lengths), ('hfd', 'k_values', 'lengths'))


def petrosian_fd(signal=None):
    """Petrosian Fractal Dimension for signal complexity.

    The Petrosian method provides a fast estimation of the fractal dimension
    based on the number of sign changes in the signal's first derivative.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    pfd : float
        Petrosian Fractal Dimension.

    Notes
    -----
    * PFD is computationally very fast.
    * Less accurate than Higuchi FD but useful for real-time applications.
    * Values typically range from 1 to 1.5 for biological signals.

    References
    ----------
    Petrosian A. Kolmogorov complexity of finite sequences and recognition of
    different preictal EEG patterns. Proc. 8th IEEE Symp. Comput.-Based Med.
    Syst. 1995:212-217.

    Examples
    --------
    >>> import numpy as np
    >>> from biosppy import chaos
    >>> signal = np.random.randn(1000)
    >>> result = chaos.petrosian_fd(signal=signal)
    >>> print("Petrosian FD:", result['pfd'])

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal).flatten()
    N = len(signal)

    # compute first derivative
    diff = np.diff(signal)

    # count sign changes
    n_delta = np.sum(diff[:-1] * diff[1:] < 0)

    # compute Petrosian FD
    pfd = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * n_delta)))

    return utils.ReturnTuple((pfd,), ('pfd',))


def katz_fd(signal=None):
    """Katz Fractal Dimension for signal complexity.

    The Katz method estimates the fractal dimension based on the Euclidean
    distance in the signal's phase space.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    kfd : float
        Katz Fractal Dimension.

    Notes
    -----
    * KFD is fast and simple to compute.
    * Values typically range from 1 to 2.
    * Good for characterizing waveform complexity.

    References
    ----------
    Katz MJ. Fractals and the analysis of waveforms. Comput Biol Med.
    1988;18(3):145-156.

    Examples
    --------
    >>> import numpy as np
    >>> from biosppy import chaos
    >>> signal = np.random.randn(1000)
    >>> result = chaos.katz_fd(signal=signal)
    >>> print("Katz FD:", result['kfd'])

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal).flatten()
    N = len(signal)

    # compute distances
    dists = np.abs(np.diff(signal))
    L = np.sum(dists)  # total curve length

    # diameter (maximum distance from first point)
    a = np.sqrt((np.arange(N) - 0) ** 2 + (signal - signal[0]) ** 2)
    d = np.max(a)

    # Katz FD
    if d == 0:
        kfd = 0
    else:
        kfd = np.log10(N) / (np.log10(d / L) + np.log10(N))

    return utils.ReturnTuple((kfd,), ('kfd',))


def hurst_exponent(signal=None, min_win=10, max_win=None, n_wins=20):
    """Hurst Exponent (H) for long-range dependence analysis.

    The Hurst exponent characterizes the long-term memory of a time series
    and its tendency to regress to the mean or cluster in one direction.

    Parameters
    ----------
    signal : array
        Input signal.
    min_win : int, optional
        Minimum window size. Default is 10.
    max_win : int, optional
        Maximum window size. If None, uses len(signal) // 2.
    n_wins : int, optional
        Number of window sizes. Default is 20.

    Returns
    -------
    H : float
        Hurst exponent.
    rs_values : array
        Rescaled range (R/S) values.
    window_sizes : array
        Window sizes used.

    Notes
    -----
    * H = 0.5: Random walk (Brownian motion), no long-term memory
    * 0 < H < 0.5: Anti-persistent behavior (mean-reverting)
    * 0.5 < H < 1: Persistent behavior (trending)
    * H close to 1: Strong persistence, long-term positive autocorrelation
    * Used in finance, hydrology, and physiological signal analysis

    References
    ----------
    Hurst HE. Long-term storage capacity of reservoirs. Trans Am Soc Civ Eng.
    1951;116:770-799.

    Examples
    --------
    >>> import numpy as np
    >>> from biosppy import chaos
    >>> # Random walk (H ≈ 0.5)
    >>> signal = np.cumsum(np.random.randn(2000))
    >>> result = chaos.hurst_exponent(signal=signal)
    >>> print("Hurst exponent:", result['H'])

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal).flatten()
    N = len(signal)

    # set maximum window size
    if max_win is None:
        max_win = N // 2

    # create window sizes
    window_sizes = np.unique(
        np.logspace(np.log10(min_win), np.log10(max_win), n_wins, dtype=int)
    )

    # compute R/S for each window size
    rs_values = []

    for win_size in window_sizes:
        # divide into non-overlapping windows
        n_windows = N // win_size
        rs_window = []

        for i in range(n_windows):
            window = signal[i * win_size:(i + 1) * win_size]

            # mean-centered cumulative sum
            mean_window = np.mean(window)
            Y = np.cumsum(window - mean_window)

            # range
            R = np.max(Y) - np.min(Y)

            # standard deviation
            S = np.std(window, ddof=1)

            # rescaled range
            if S != 0:
                rs_window.append(R / S)

        if rs_window:
            rs_values.append(np.mean(rs_window))

    rs_values = np.array(rs_values)

    if len(rs_values) < 2:
        raise ValueError("Not enough valid R/S values to compute Hurst exponent.")

    # Hurst exponent (slope of log(R/S) vs log(n))
    valid_sizes = window_sizes[:len(rs_values)]
    coeffs = np.polyfit(np.log(valid_sizes), np.log(rs_values), 1)
    H = coeffs[0]

    return utils.ReturnTuple((H, rs_values, window_sizes[:len(rs_values)]),
                            ('H', 'rs_values', 'window_sizes'))


def lyapunov_exponent(signal=None, emb_dim=10, matrix_dim=4, min_tsep=None,
                     max_tsep=None, tau=1):
    """Estimate the largest Lyapunov exponent using Rosenstein's algorithm.

    The Lyapunov exponent quantifies the rate of separation of infinitesimally
    close trajectories, measuring the system's sensitivity to initial conditions
    (chaos).

    Parameters
    ----------
    signal : array
        Input signal.
    emb_dim : int, optional
        Embedding dimension. Default is 10.
    matrix_dim : int, optional
        Matrix dimension for delay coordinates. Default is 4.
    min_tsep : int, optional
        Minimum temporal separation. If None, uses 10.
    max_tsep : int, optional
        Maximum temporal separation. If None, uses emb_dim * 10.
    tau : int, optional
        Time delay for embedding. Default is 1.

    Returns
    -------
    lambda_max : float
        Largest Lyapunov exponent (bits/iteration).
    divergence : array
        Average logarithmic divergence over time.

    Notes
    -----
    * λ > 0: Chaotic behavior, exponential divergence
    * λ = 0: Periodic or quasi-periodic behavior
    * λ < 0: Stable fixed point, convergence
    * Positive Lyapunov exponent is a hallmark of deterministic chaos
    * Common in heart rate variability and EEG analysis

    References
    ----------
    Rosenstein MT, Collins JJ, De Luca CJ. A practical method for calculating
    largest Lyapunov exponents from small data sets. Physica D. 1993;65(1-2):117-134.

    Examples
    --------
    >>> import numpy as np
    >>> from biosppy import chaos
    >>> # Logistic map (chaotic)
    >>> x = [0.1]
    >>> r = 3.9
    >>> for _ in range(5000):
    ...     x.append(r * x[-1] * (1 - x[-1]))
    >>> result = chaos.lyapunov_exponent(signal=np.array(x))
    >>> print("Lyapunov exponent:", result['lambda_max'])

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal).flatten()
    N = len(signal)

    # set defaults
    if min_tsep is None:
        min_tsep = 10
    if max_tsep is None:
        max_tsep = emb_dim * 10

    # time-delay embedding
    M = N - (emb_dim - 1) * tau
    if M <= 0:
        raise ValueError("Signal too short for given embedding parameters.")

    embedded = np.zeros((M, emb_dim))
    for i in range(M):
        for j in range(emb_dim):
            embedded[i, j] = signal[i + j * tau]

    # find nearest neighbors
    max_iter = min(max_tsep, M - 1)
    divergence = np.zeros(max_iter)

    for i in range(M - max_iter):
        # compute distances to all other points
        distances = np.sqrt(np.sum((embedded - embedded[i]) ** 2, axis=1))

        # find nearest neighbor (excluding nearby points in time)
        valid_indices = np.where(np.abs(np.arange(M) - i) > min_tsep)[0]
        if len(valid_indices) == 0:
            continue

        nearest_idx = valid_indices[np.argmin(distances[valid_indices])]

        # track divergence
        for j in range(max_iter):
            if i + j < M and nearest_idx + j < M:
                dist = np.sqrt(np.sum((embedded[i + j] - embedded[nearest_idx + j]) ** 2))
                if dist > 0:
                    divergence[j] += np.log(dist)

    # average divergence
    divergence = divergence / (M - max_iter)

    # fit line to estimate Lyapunov exponent
    # use middle portion of the curve (avoid initial transient and saturation)
    fit_start = max_iter // 10
    fit_end = max_iter // 2

    x_fit = np.arange(fit_start, fit_end)
    y_fit = divergence[fit_start:fit_end]

    if len(x_fit) > 1:
        coeffs = np.polyfit(x_fit, y_fit, 1)
        lambda_max = coeffs[0]  # slope is the Lyapunov exponent
    else:
        lambda_max = np.nan

    return utils.ReturnTuple((lambda_max, divergence), ('lambda_max', 'divergence'))
