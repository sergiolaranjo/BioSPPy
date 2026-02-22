# -*- coding: utf-8 -*-
"""
biosppy.signals.emd
-------------------

This module provides methods for Empirical Mode Decomposition (EMD) and its
improved variants (EEMD, CEEMDAN) for signal decomposition and analysis.

The Hilbert-Huang Transform (HHT) consists of two main steps:
1. Empirical Mode Decomposition (EMD) - decomposes signal into Intrinsic Mode Functions (IMFs)
2. Hilbert Spectral Analysis - extracts instantaneous frequency and amplitude

:copyright: (c) 2015-2025 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

References
----------
.. [Huang98] Huang, N. E., Shen, Z., Long, S. R., Wu, M. C., Shih, H. H.,
             Zheng, Q., ... & Liu, H. H. (1998). The empirical mode decomposition
             and the Hilbert spectrum for nonlinear and non-stationary time series
             analysis. Proceedings of the Royal Society of London. Series A, 454(1971), 903-995.
.. [Wu09] Wu, Z., & Huang, N. E. (2009). Ensemble empirical mode decomposition:
          a noise-assisted data analysis method. Advances in adaptive data analysis, 1(01), 1-41.
.. [Torres11] Torres, M. E., Colominas, M. A., Schlotthauer, G., & Flandrin, P. (2011).
              A complete ensemble empirical mode decomposition with adaptive noise.
              IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 4144-4147.
.. [Colominas14] Colominas, M. A., Schlotthauer, G., & Torres, M. E. (2014).
                 Improved complete ensemble EMD: A suitable tool for biomedical signal processing.
                 Biomedical Signal Processing and Control, 14, 19-29.
"""

# Imports
import numpy as np
from scipy import interpolate
from .. import utils


def _find_extrema(signal):
    """Find local maxima and minima in a signal.

    Parameters
    ----------
    signal : array
        Input signal.

    Returns
    -------
    max_indices : array
        Indices of local maxima.
    min_indices : array
        Indices of local minima.
    """
    signal = np.array(signal)
    n = len(signal)

    # Find maxima
    max_indices = []
    for i in range(1, n - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            max_indices.append(i)

    # Find minima
    min_indices = []
    for i in range(1, n - 1):
        if signal[i] < signal[i - 1] and signal[i] < signal[i + 1]:
            min_indices.append(i)

    return np.array(max_indices), np.array(min_indices)


def _get_envelopes(signal, max_indices, min_indices):
    """Compute upper and lower envelopes using cubic spline interpolation.

    Parameters
    ----------
    signal : array
        Input signal.
    max_indices : array
        Indices of local maxima.
    min_indices : array
        Indices of local minima.

    Returns
    -------
    upper_env : array
        Upper envelope.
    lower_env : array
        Lower envelope.
    mean_env : array
        Mean of upper and lower envelopes.
    """
    n = len(signal)
    t = np.arange(n)

    # Handle edge cases
    if len(max_indices) < 2 or len(min_indices) < 2:
        return signal, signal, np.zeros(n)

    # Add boundary points for better extrapolation
    # Prepend
    max_indices_ext = np.concatenate([[0], max_indices, [n - 1]])
    min_indices_ext = np.concatenate([[0], min_indices, [n - 1]])

    max_values = signal[max_indices_ext]
    min_values = signal[min_indices_ext]

    # Cubic spline interpolation
    try:
        upper_spline = interpolate.CubicSpline(max_indices_ext, max_values,
                                               bc_type='natural')
        lower_spline = interpolate.CubicSpline(min_indices_ext, min_values,
                                               bc_type='natural')

        upper_env = upper_spline(t)
        lower_env = lower_spline(t)
        mean_env = (upper_env + lower_env) / 2.0

    except Exception:
        # Fallback to linear interpolation
        upper_env = np.interp(t, max_indices_ext, max_values)
        lower_env = np.interp(t, min_indices_ext, min_values)
        mean_env = (upper_env + lower_env) / 2.0

    return upper_env, lower_env, mean_env


def _is_imf(signal, tol=0.05):
    """Check if a signal satisfies the IMF criteria.

    An IMF must satisfy two conditions:
    1. Number of extrema and zero crossings must differ at most by one
    2. Mean of envelopes must be close to zero

    Parameters
    ----------
    signal : array
        Input signal.
    tol : float, optional
        Tolerance for mean envelope check.

    Returns
    -------
    is_imf : bool
        True if signal is an IMF.
    """
    max_indices, min_indices = _find_extrema(signal)
    n_extrema = len(max_indices) + len(min_indices)

    if n_extrema < 3:
        return False

    # Check zero crossings
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    n_zero = len(zero_crossings)

    # Condition 1: differ by at most 1
    if abs(n_extrema - n_zero) > 1:
        return False

    # Condition 2: mean envelope close to zero
    if len(max_indices) < 2 or len(min_indices) < 2:
        return False

    _, _, mean_env = _get_envelopes(signal, max_indices, min_indices)
    mean_val = np.abs(mean_env).mean()
    signal_std = np.std(signal)

    if signal_std == 0:
        return False

    return (mean_val / signal_std) < tol


def _sift(signal, max_iter=1000, tol=0.05):
    """Perform sifting process to extract one IMF.

    Parameters
    ----------
    signal : array
        Input signal or residue.
    max_iter : int, optional
        Maximum number of sifting iterations.
    tol : float, optional
        Tolerance for IMF criteria.

    Returns
    -------
    imf : array
        Extracted IMF.
    residue : array
        Remaining signal after IMF extraction.
    """
    h = signal.copy()

    for iteration in range(max_iter):
        max_indices, min_indices = _find_extrema(h)

        # Check if we can continue
        if len(max_indices) < 2 or len(min_indices) < 2:
            break

        # Compute mean envelope
        _, _, mean_env = _get_envelopes(h, max_indices, min_indices)

        # Update h
        h_prev = h.copy()
        h = h - mean_env

        # Check stopping criteria
        if _is_imf(h, tol):
            break

        # Check convergence
        sd = np.sum((h - h_prev) ** 2) / np.sum(h_prev ** 2 + 1e-10)
        if sd < 0.001:
            break

    imf = h
    residue = signal - imf

    return imf, residue


def emd(signal=None, max_imf=None, max_iter=1000, tol=0.05):
    """Empirical Mode Decomposition (EMD).

    Decomposes a signal into Intrinsic Mode Functions (IMFs) and a residue.

    Parameters
    ----------
    signal : array
        Input signal.
    max_imf : int, optional
        Maximum number of IMFs to extract.
        If None, extracts all possible IMFs.
    max_iter : int, optional
        Maximum number of sifting iterations per IMF.
    tol : float, optional
        Tolerance for IMF criteria.

    Returns
    -------
    imfs : array
        Extracted IMFs (n_imfs, n_samples).
    residue : array
        Final residue after all IMFs extraction.

    Notes
    -----
    - IMFs are oscillatory components with well-behaved Hilbert transforms
    - The original signal can be reconstructed as: signal = sum(imfs) + residue
    - EMD is adaptive and data-driven, suitable for non-linear and non-stationary signals

    Example
    -------
    >>> import numpy as np
    >>> from biosppy.signals import emd
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
    >>> imfs, residue = emd.emd(signal=signal)
    """
    # Check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal, dtype=float)
    n = len(signal)

    if max_imf is None:
        # Theoretical maximum: log2(N)
        max_imf = int(np.log2(n))

    imfs = []
    residue = signal.copy()

    for i in range(max_imf):
        # Check if residue has enough extrema
        max_indices, min_indices = _find_extrema(residue)
        n_extrema = len(max_indices) + len(min_indices)

        if n_extrema < 3:
            break

        # Extract IMF
        imf, residue = _sift(residue, max_iter=max_iter, tol=tol)

        # Check if IMF is valid
        if np.std(imf) < 1e-10:
            break

        imfs.append(imf)

        # Check if residue is monotonic
        max_indices, min_indices = _find_extrema(residue)
        if len(max_indices) + len(min_indices) < 3:
            break

    if len(imfs) == 0:
        imfs = np.array([signal])
    else:
        imfs = np.array(imfs)

    return utils.ReturnTuple((imfs, residue), ('imfs', 'residue'))


def eemd(signal=None, num_ensemble=100, noise_std=0.2, max_imf=None,
         max_iter=1000, tol=0.05, random_seed=None):
    """Ensemble Empirical Mode Decomposition (EEMD).

    Improves EMD by adding white noise to reduce mode mixing.
    Performs EMD on multiple noise-added copies and averages the results.

    Parameters
    ----------
    signal : array
        Input signal.
    num_ensemble : int, optional
        Number of ensemble members (noise realizations).
    noise_std : float, optional
        Standard deviation of added white noise (relative to signal std).
    max_imf : int, optional
        Maximum number of IMFs to extract.
    max_iter : int, optional
        Maximum sifting iterations per IMF.
    tol : float, optional
        Tolerance for IMF criteria.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    imfs : array
        Extracted IMFs (n_imfs, n_samples).
    residue : array
        Final residue.

    Notes
    -----
    - EEMD reduces mode mixing by ensemble averaging
    - Computational cost is N times higher than EMD (N = num_ensemble)
    - Recommended: num_ensemble >= 100, noise_std = 0.2

    References
    ----------
    .. [Wu09] Wu, Z., & Huang, N. E. (2009). Ensemble empirical mode decomposition.

    Example
    -------
    >>> from biosppy.signals import emd
    >>> import numpy as np
    >>> signal = np.sin(2*np.pi*5*np.linspace(0,1,1000))
    >>> imfs, residue = emd.eemd(signal=signal, num_ensemble=100)
    """
    # Check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal, dtype=float)
    n = len(signal)
    signal_std = np.std(signal)

    if random_seed is not None:
        np.random.seed(random_seed)

    # Determine number of IMFs from first decomposition
    first_imfs, first_residue = emd(signal, max_imf=max_imf, max_iter=max_iter, tol=tol)
    n_imfs = len(first_imfs)

    if max_imf is None or max_imf > n_imfs:
        max_imf = n_imfs

    # Initialize ensemble storage
    imfs_ensemble = np.zeros((num_ensemble, max_imf, n))
    residue_ensemble = np.zeros((num_ensemble, n))

    # Ensemble loop
    for ens in range(num_ensemble):
        # Add white noise
        noise = np.random.randn(n) * noise_std * signal_std
        noisy_signal = signal + noise

        # Decompose
        imfs_temp, residue_temp = emd(noisy_signal, max_imf=max_imf,
                                       max_iter=max_iter, tol=tol)

        # Store (pad if necessary)
        n_imfs_temp = len(imfs_temp)
        for i in range(min(n_imfs_temp, max_imf)):
            imfs_ensemble[ens, i, :] = imfs_temp[i]

        residue_ensemble[ens, :] = residue_temp

    # Average over ensemble
    imfs = np.mean(imfs_ensemble, axis=0)
    residue = np.mean(residue_ensemble, axis=0)

    return utils.ReturnTuple((imfs, residue), ('imfs', 'residue'))


def _generate_adaptive_noise(modes, scale, random_state=None):
    """Generate adaptive noise for CEEMDAN.

    Parameters
    ----------
    modes : array
        EMD modes of white noise.
    scale : float
        Noise scaling factor.
    random_state : RandomState, optional
        Random number generator state.

    Returns
    -------
    noise : array
        Generated adaptive noise.
    """
    if random_state is None:
        random_state = np.random.RandomState()

    n_modes, n_samples = modes.shape
    noise = np.zeros(n_samples)

    for i in range(n_modes):
        noise += modes[i] * random_state.randn()

    return noise * scale


def ceemdan(signal=None, num_ensemble=100, noise_std=0.2, max_imf=None,
            max_iter=1000, tol=0.05, random_seed=None):
    """Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN).

    Most recent and improved version of EMD that provides better spectral separation
    and reduced reconstruction error compared to EEMD.

    Parameters
    ----------
    signal : array
        Input signal.
    num_ensemble : int, optional
        Number of ensemble members.
    noise_std : float, optional
        Standard deviation of added noise (relative to signal std).
    max_imf : int, optional
        Maximum number of IMFs to extract.
    max_iter : int, optional
        Maximum sifting iterations per IMF.
    tol : float, optional
        Tolerance for IMF criteria.
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    imfs : array
        Extracted IMFs (n_imfs, n_samples).
    residue : array
        Final residue.

    Notes
    -----
    - CEEMDAN provides better mode alignment and less residual noise than EEMD
    - Uses adaptive noise at each decomposition stage
    - Nearly perfect reconstruction: signal ≈ sum(imfs) + residue
    - Recommended for biomedical signal analysis (HRV, EEG, EMG)

    References
    ----------
    .. [Torres11] Torres et al. (2011). A complete ensemble EMD with adaptive noise.
    .. [Colominas14] Colominas et al. (2014). Improved complete ensemble EMD.

    Example
    -------
    >>> from biosppy.signals import emd
    >>> import numpy as np
    >>> # Generate test signal
    >>> t = np.linspace(0, 1, 1000)
    >>> signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*15*t)
    >>> # Decompose using CEEMDAN
    >>> imfs, residue = emd.ceemdan(signal=signal, num_ensemble=100, noise_std=0.2)
    >>> # Reconstruct
    >>> reconstructed = np.sum(imfs, axis=0) + residue
    """
    # Check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal, dtype=float)
    n = len(signal)
    signal_std = np.std(signal)

    if random_seed is not None:
        np.random.seed(random_seed)

    # Pre-compute noise realizations
    noise_ensemble = np.random.randn(num_ensemble, n) * noise_std * signal_std

    # Pre-compute first EMD mode of each noise
    E1 = np.zeros((num_ensemble, n))
    for ens in range(num_ensemble):
        imfs_noise, _ = emd(noise_ensemble[ens], max_imf=1, max_iter=max_iter, tol=tol)
        if len(imfs_noise) > 0:
            E1[ens] = imfs_noise[0]

    # Determine max number of IMFs
    if max_imf is None:
        max_imf = int(np.log2(n))

    # Initialize
    imfs = []
    residue = signal.copy()

    # Extract first IMF
    modes_sum = np.zeros(n)
    for ens in range(num_ensemble):
        temp_signal = signal + noise_ensemble[ens]
        imf_temp, _ = emd(temp_signal, max_imf=1, max_iter=max_iter, tol=tol)
        if len(imf_temp) > 0:
            modes_sum += imf_temp[0]

    imf1 = modes_sum / num_ensemble
    imfs.append(imf1)
    residue = signal - imf1

    # Extract subsequent IMFs
    for k in range(1, max_imf):
        # Check stopping criterion
        max_indices, min_indices = _find_extrema(residue)
        if len(max_indices) + len(min_indices) < 3:
            break

        # Compute E_k (kth mode of noise)
        Ek = np.zeros((num_ensemble, n))
        for ens in range(num_ensemble):
            imfs_noise, _ = emd(noise_ensemble[ens], max_imf=k+1,
                               max_iter=max_iter, tol=tol)
            if len(imfs_noise) > k:
                Ek[ens] = imfs_noise[k]

        # Extract kth IMF with adaptive noise
        modes_sum = np.zeros(n)
        for ens in range(num_ensemble):
            temp_signal = residue + noise_std * signal_std * Ek[ens]
            imf_temp, _ = emd(temp_signal, max_imf=1, max_iter=max_iter, tol=tol)
            if len(imf_temp) > 0:
                modes_sum += imf_temp[0]

        imf_k = modes_sum / num_ensemble

        # Check if IMF is valid
        if np.std(imf_k) < 1e-10:
            break

        imfs.append(imf_k)
        residue = residue - imf_k

    if len(imfs) == 0:
        imfs = np.array([signal])
    else:
        imfs = np.array(imfs)

    return utils.ReturnTuple((imfs, residue), ('imfs', 'residue'))


def hilbert_spectrum(imfs=None, sampling_rate=1000.0):
    """Compute Hilbert spectrum from IMFs.

    Applies Hilbert transform to each IMF to extract instantaneous
    amplitude, frequency, and phase.

    Parameters
    ----------
    imfs : array
        Intrinsic Mode Functions (n_imfs, n_samples).
    sampling_rate : float, optional
        Sampling frequency (Hz).

    Returns
    -------
    inst_amplitude : array
        Instantaneous amplitude for each IMF (n_imfs, n_samples).
    inst_frequency : array
        Instantaneous frequency for each IMF (n_imfs, n_samples).
    inst_phase : array
        Instantaneous phase for each IMF (n_imfs, n_samples).

    Notes
    -----
    - Instantaneous frequency is computed as derivative of unwrapped phase
    - Negative frequencies are set to zero
    - Edge effects may occur at signal boundaries

    Example
    -------
    >>> from biosppy.signals import emd
    >>> import numpy as np
    >>> signal = np.sin(2*np.pi*5*np.linspace(0,1,1000))
    >>> imfs, _ = emd.ceemdan(signal=signal)
    >>> amp, freq, phase = emd.hilbert_spectrum(imfs=imfs, sampling_rate=1000)
    """
    # Check inputs
    if imfs is None:
        raise TypeError("Please specify IMFs.")

    imfs = np.array(imfs)

    if imfs.ndim == 1:
        imfs = imfs.reshape(1, -1)

    n_imfs, n_samples = imfs.shape
    dt = 1.0 / sampling_rate

    # Initialize output arrays
    inst_amplitude = np.zeros((n_imfs, n_samples))
    inst_frequency = np.zeros((n_imfs, n_samples))
    inst_phase = np.zeros((n_imfs, n_samples))

    # Process each IMF
    for i in range(n_imfs):
        # Hilbert transform
        analytic = np.fft.ifft(np.fft.fft(imfs[i]) *
                              np.concatenate([[1], 2*np.ones(n_samples//2-1),
                                            [1], np.zeros(n_samples//2-1 if n_samples % 2 == 0 else n_samples//2)]))

        # Instantaneous amplitude
        inst_amplitude[i] = np.abs(analytic)

        # Instantaneous phase
        inst_phase[i] = np.angle(analytic)

        # Instantaneous frequency (derivative of unwrapped phase)
        phase_unwrapped = np.unwrap(inst_phase[i])
        inst_frequency[i, 1:] = np.diff(phase_unwrapped) / (2 * np.pi * dt)
        inst_frequency[i, 0] = inst_frequency[i, 1]

        # Remove negative frequencies
        inst_frequency[i] = np.maximum(inst_frequency[i], 0)

    return utils.ReturnTuple((inst_amplitude, inst_frequency, inst_phase),
                           ('inst_amplitude', 'inst_frequency', 'inst_phase'))


def marginal_spectrum(inst_amplitude=None, inst_frequency=None,
                      freq_bins=None, sampling_rate=1000.0):
    """Compute marginal Hilbert spectrum (frequency-amplitude distribution).

    Integrates instantaneous amplitude over time for each frequency.

    Parameters
    ----------
    inst_amplitude : array
        Instantaneous amplitude (n_imfs, n_samples).
    inst_frequency : array
        Instantaneous frequency (n_imfs, n_samples).
    freq_bins : array, optional
        Frequency bins for histogram.
        If None, uses 100 bins from 0 to Nyquist frequency.
    sampling_rate : float, optional
        Sampling frequency (Hz).

    Returns
    -------
    frequencies : array
        Frequency bins (Hz).
    amplitudes : array
        Marginal amplitude spectrum.

    Notes
    -----
    - Marginal spectrum shows total energy contribution at each frequency
    - Similar to Fourier power spectrum but for non-stationary signals

    Example
    -------
    >>> from biosppy.signals import emd
    >>> import numpy as np
    >>> signal = np.sin(2*np.pi*5*np.linspace(0,1,1000))
    >>> imfs, _ = emd.ceemdan(signal=signal)
    >>> amp, freq, _ = emd.hilbert_spectrum(imfs=imfs)
    >>> f, h = emd.marginal_spectrum(inst_amplitude=amp, inst_frequency=freq)
    """
    # Check inputs
    if inst_amplitude is None or inst_frequency is None:
        raise TypeError("Please specify instantaneous amplitude and frequency.")

    inst_amplitude = np.array(inst_amplitude)
    inst_frequency = np.array(inst_frequency)

    if inst_amplitude.ndim == 1:
        inst_amplitude = inst_amplitude.reshape(1, -1)
        inst_frequency = inst_frequency.reshape(1, -1)

    # Define frequency bins
    if freq_bins is None:
        nyquist = sampling_rate / 2.0
        freq_bins = np.linspace(0, nyquist, 100)

    n_bins = len(freq_bins)
    amplitudes = np.zeros(n_bins - 1)

    # Compute histogram
    for i in range(inst_amplitude.shape[0]):
        hist, _ = np.histogram(inst_frequency[i], bins=freq_bins,
                              weights=inst_amplitude[i])
        amplitudes += hist

    # Frequency bin centers
    frequencies = (freq_bins[:-1] + freq_bins[1:]) / 2.0

    return utils.ReturnTuple((frequencies, amplitudes),
                           ('frequencies', 'amplitudes'))
