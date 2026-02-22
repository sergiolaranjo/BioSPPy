# -*- coding: utf-8 -*-
"""
biosppy.features.wavelet_coherence
-----------------------------------

This module provides methods to compute wavelet coherence between two signals,
including phase analysis and temporal delay estimation.

The wavelet coherence measures the cross-correlation between two time series
as a function of frequency and time, similar to the MATLAB implementation.

:copyright: (c) 2015-2023 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# 3rd party
import numpy as np
import pywt
from scipy import signal as sp_signal

# local
from .. import utils


def wavelet_coherence(signal1=None, signal2=None, sampling_rate=1000.,
                     wavelet='morl', scales=None,
                     compute_phase=True, compute_delay=True):
    """Compute wavelet coherence between two signals.

    Wavelet coherence is a measure of cross-correlation between two signals
    as a function of frequency and time. It is analogous to the coherence
    between two signals in the frequency domain, but localized in time.

    Parameters
    ----------
    signal1 : array
        First input signal.
    signal2 : array
        Second input signal (must have same length as signal1).
    sampling_rate : float, optional
        Sampling rate of the signals in Hz. Default is 1000.
    wavelet : str, optional
        Wavelet name to use. Default is 'morl' (Morlet wavelet).
        Common options: 'morl', 'cmor', 'cgau', 'mexh', 'gaus'.
    scales : array, optional
        Scales to use for the wavelet transform. If None, automatically
        computed based on signal length.
    compute_phase : bool, optional
        Whether to compute phase difference. Default is True.
    compute_delay : bool, optional
        Whether to compute temporal delay from phase. Default is True.
        Requires compute_phase=True.

    Returns
    -------
    coherence : array
        Wavelet coherence matrix (scales x time).
    frequencies : array
        Corresponding frequencies for each scale in Hz.
    phase : array, optional
        Phase difference matrix (scales x time) in radians.
        Only returned if compute_phase=True.
    delay : array, optional
        Temporal delay matrix (scales x time) in seconds.
        Only returned if compute_delay=True.
    cross_spectrum : array
        Cross-wavelet spectrum (complex matrix, scales x time).

    Notes
    -----
    The wavelet coherence is computed as:

    .. math::
        WTC = \\frac{|S(W_1 \\cdot W_2^*)|^2}{S(|W_1|^2) \\cdot S(|W_2|^2)}

    where W_1 and W_2 are the continuous wavelet transforms of signal1 and signal2,
    W_2^* is the complex conjugate of W_2, and S is a smoothing operator in time
    and scale.

    The phase difference is computed as:

    .. math::
        \\phi = \\text{angle}(S(W_1 \\cdot W_2^*))

    The temporal delay is computed as:

    .. math::
        \\tau = \\frac{\\phi}{2\\pi f}

    where f is the frequency.

    References
    ----------
    .. [1] Torrence, C. and Compo, G.P., 1998. A practical guide to wavelet analysis.
           Bulletin of the American Meteorological society, 79(1), pp.61-78.
    .. [2] Grinsted, A., Moore, J.C. and Jevrejeva, S., 2004. Application of the cross
           wavelet transform and wavelet coherence to geophysical time series.
           Nonlinear processes in geophysics, 11(5/6), pp.561-566.

    Examples
    --------
    >>> import numpy as np
    >>> from biosppy.features import wavelet_coherence as wc
    >>> # Generate two test signals with some correlation
    >>> t = np.linspace(0, 10, 1000)
    >>> signal1 = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
    >>> signal2 = np.sin(2 * np.pi * 5 * t + 0.5) + 0.3 * np.sin(2 * np.pi * 10 * t)
    >>> # Compute wavelet coherence
    >>> result = wc.wavelet_coherence(signal1, signal2, sampling_rate=100)
    >>> coherence, frequencies, phase, delay, cross_spectrum = result

    """

    # check inputs
    if signal1 is None or signal2 is None:
        raise TypeError("Please specify both input signals.")

    # ensure numpy
    signal1 = np.array(signal1)
    signal2 = np.array(signal2)

    # check signals have same length
    if len(signal1) != len(signal2):
        raise ValueError("Both signals must have the same length.")

    n = len(signal1)

    # compute scales if not provided
    if scales is None:
        # Use logarithmically spaced scales
        # This covers a good frequency range
        max_scale = n // 2
        min_scale = 2
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 64)

    # compute continuous wavelet transform for both signals
    cwt1, frequencies = _continuous_wavelet_transform(signal1, scales, wavelet, sampling_rate)
    cwt2, _ = _continuous_wavelet_transform(signal2, scales, wavelet, sampling_rate)

    # compute cross-wavelet spectrum
    cross_spectrum = cwt1 * np.conj(cwt2)

    # compute power spectra
    power1 = np.abs(cwt1) ** 2
    power2 = np.abs(cwt2) ** 2

    # smooth the spectra (essential for coherence calculation)
    smooth_cross = _smooth_wavelet_spectrum(cross_spectrum, scales, wavelet)
    smooth_power1 = _smooth_wavelet_spectrum(power1, scales, wavelet)
    smooth_power2 = _smooth_wavelet_spectrum(power2, scales, wavelet)

    # compute wavelet coherence
    coherence = np.abs(smooth_cross) ** 2 / (smooth_power1 * smooth_power2)

    # ensure real-valued output
    coherence = np.real(coherence)

    # clip to [0, 1] to avoid numerical errors
    coherence = np.clip(coherence, 0, 1)

    # prepare output
    args = [coherence, frequencies]
    names = ['coherence', 'frequencies']

    # compute phase if requested
    if compute_phase:
        phase = np.angle(smooth_cross)
        args.append(phase)
        names.append('phase')

        # compute temporal delay if requested
        if compute_delay:
            # delay = phase / (2 * pi * frequency)
            # reshape frequencies for broadcasting
            freq_matrix = frequencies[:, np.newaxis]
            # avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                delay = phase / (2 * np.pi * freq_matrix)
                # set delay to 0 where frequency is 0
                delay = np.where(freq_matrix > 0, delay, 0)
            args.append(delay)
            names.append('delay')

    # always include cross spectrum
    args.append(cross_spectrum)
    names.append('cross_spectrum')

    return utils.ReturnTuple(tuple(args), tuple(names))


def cross_wavelet_spectrum(signal1=None, signal2=None, sampling_rate=1000.,
                           wavelet='morl', scales=None):
    """Compute cross-wavelet spectrum between two signals.

    The cross-wavelet spectrum reveals areas in time-frequency space where
    the two signals have common power.

    Parameters
    ----------
    signal1 : array
        First input signal.
    signal2 : array
        Second input signal (must have same length as signal1).
    sampling_rate : float, optional
        Sampling rate of the signals in Hz. Default is 1000.
    wavelet : str, optional
        Wavelet name to use. Default is 'morl' (Morlet wavelet).
    scales : array, optional
        Scales to use for the wavelet transform. If None, automatically computed.

    Returns
    -------
    cross_spectrum : array
        Cross-wavelet spectrum (complex matrix, scales x time).
    frequencies : array
        Corresponding frequencies for each scale in Hz.
    power : array
        Magnitude of cross-wavelet spectrum.
    phase : array
        Phase of cross-wavelet spectrum in radians.

    """

    # check inputs
    if signal1 is None or signal2 is None:
        raise TypeError("Please specify both input signals.")

    # ensure numpy
    signal1 = np.array(signal1)
    signal2 = np.array(signal2)

    # check signals have same length
    if len(signal1) != len(signal2):
        raise ValueError("Both signals must have the same length.")

    n = len(signal1)

    # compute scales if not provided
    if scales is None:
        max_scale = n // 2
        min_scale = 2
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 64)

    # compute continuous wavelet transform for both signals
    cwt1, frequencies = _continuous_wavelet_transform(signal1, scales, wavelet, sampling_rate)
    cwt2, _ = _continuous_wavelet_transform(signal2, scales, wavelet, sampling_rate)

    # compute cross-wavelet spectrum
    cross_spectrum = cwt1 * np.conj(cwt2)

    # compute power and phase
    power = np.abs(cross_spectrum)
    phase = np.angle(cross_spectrum)

    # output
    args = (cross_spectrum, frequencies, power, phase)
    names = ('cross_spectrum', 'frequencies', 'power', 'phase')

    return utils.ReturnTuple(args, names)


def _continuous_wavelet_transform(signal, scales, wavelet, sampling_rate):
    """Compute continuous wavelet transform.

    Parameters
    ----------
    signal : array
        Input signal.
    scales : array
        Scales for the wavelet transform.
    wavelet : str
        Wavelet name.
    sampling_rate : float
        Sampling rate in Hz.

    Returns
    -------
    cwt : array
        Continuous wavelet transform coefficients (scales x time).
    frequencies : array
        Corresponding frequencies for each scale in Hz.

    """

    # compute CWT
    cwt, frequencies = pywt.cwt(signal, scales, wavelet, sampling_period=1.0/sampling_rate)

    return cwt, frequencies


def _smooth_wavelet_spectrum(spectrum, scales, wavelet='morl'):
    """Smooth wavelet spectrum in time and scale.

    This smoothing is essential for computing wavelet coherence.
    It reduces noise and improves the statistical significance of the coherence.

    Parameters
    ----------
    spectrum : array
        Wavelet spectrum (scales x time).
    scales : array
        Scales used in the wavelet transform.
    wavelet : str
        Wavelet name used.

    Returns
    -------
    smoothed : array
        Smoothed spectrum.

    """

    n_scales, n_times = spectrum.shape
    smoothed = np.zeros_like(spectrum, dtype=complex)

    # smoothing parameters
    # These are based on Torrence & Compo (1998)
    for i, scale in enumerate(scales):
        # time smoothing: convolve with a normalized window
        # window width proportional to scale
        time_window_size = int(np.ceil(scale / 2))
        time_window_size = max(3, time_window_size)  # minimum size
        if time_window_size % 2 == 0:  # make it odd
            time_window_size += 1

        time_window = np.ones(time_window_size) / time_window_size

        # smooth in time (apply to real and imaginary parts separately)
        real_part = np.convolve(np.real(spectrum[i, :]), time_window, mode='same')
        imag_part = np.convolve(np.imag(spectrum[i, :]), time_window, mode='same')
        smoothed[i, :] = real_part + 1j * imag_part

    # scale smoothing: smooth across neighboring scales
    # use a simple moving average
    scale_window_size = 3  # smooth across 3 adjacent scales
    if n_scales >= scale_window_size:
        for j in range(n_times):
            smoothed[:, j] = np.convolve(smoothed[:, j],
                                        np.ones(scale_window_size) / scale_window_size,
                                        mode='same')

    return smoothed


def compute_temporal_delay(phase, frequencies, unwrap=True):
    """Compute temporal delay from phase difference.

    The temporal delay represents the time shift between the two signals
    at each frequency and time point.

    Parameters
    ----------
    phase : array
        Phase difference matrix (scales x time) in radians.
    frequencies : array
        Frequencies corresponding to each scale in Hz.
    unwrap : bool, optional
        Whether to unwrap the phase before computing delay. Default is True.
        Phase unwrapping removes discontinuities in the phase.

    Returns
    -------
    delay : array
        Temporal delay matrix (scales x time) in seconds.
        Positive values indicate signal1 leads signal2.
        Negative values indicate signal2 leads signal1.

    """

    phase = np.array(phase)
    frequencies = np.array(frequencies)

    # unwrap phase if requested
    if unwrap:
        phase = np.unwrap(phase, axis=1)

    # compute delay: tau = phi / (2 * pi * f)
    # reshape frequencies for broadcasting
    freq_matrix = frequencies[:, np.newaxis]

    # avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        delay = phase / (2 * np.pi * freq_matrix)
        # set delay to 0 where frequency is 0
        delay = np.where(freq_matrix > 0, delay, 0)

    return delay


def significance_test(coherence, n_samples, alpha=0.05):
    """Test statistical significance of wavelet coherence.

    Compute the significance level for wavelet coherence using
    a chi-squared distribution approximation.

    Parameters
    ----------
    coherence : array
        Wavelet coherence values (scales x time).
    n_samples : int
        Number of independent samples (approximately the signal length
        divided by the decorrelation scale).
    alpha : float, optional
        Significance level. Default is 0.05 (95% confidence).

    Returns
    -------
    is_significant : array
        Boolean matrix indicating where coherence is significant.
    threshold : float
        Coherence threshold for significance.

    Notes
    -----
    This is a simplified significance test. For more accurate results,
    consider using Monte Carlo simulations with surrogate data.

    References
    ----------
    .. [1] Torrence, C. and Compo, G.P., 1998. A practical guide to wavelet analysis.

    """

    from scipy.stats import chi2

    # degrees of freedom (approximate)
    dof = 2

    # significance threshold
    threshold = 1 - chi2.ppf(1 - alpha, dof) / n_samples

    # test significance
    is_significant = coherence > threshold

    return is_significant, threshold


def plot_wavelet_coherence(coherence, frequencies, time_axis=None,
                          phase=None, phase_step=10,
                          cmap='viridis', figsize=(12, 6)):
    """Plot wavelet coherence with optional phase arrows.

    Parameters
    ----------
    coherence : array
        Wavelet coherence matrix (frequencies x time).
    frequencies : array
        Frequency values for y-axis.
    time_axis : array, optional
        Time values for x-axis. If None, uses sample indices.
    phase : array, optional
        Phase difference matrix for plotting arrows.
    phase_step : int, optional
        Step size for plotting phase arrows (to avoid overcrowding).
        Default is 10.
    cmap : str, optional
        Colormap name. Default is 'viridis'.
    figsize : tuple, optional
        Figure size. Default is (12, 6).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.

    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib is required for plotting. Install it with: pip install matplotlib")

    n_freqs, n_times = coherence.shape

    if time_axis is None:
        time_axis = np.arange(n_times)

    # create figure
    fig, ax = plt.subplots(figsize=figsize)

    # plot coherence as image
    extent = [time_axis[0], time_axis[-1], frequencies[0], frequencies[-1]]
    im = ax.imshow(coherence, aspect='auto', origin='lower',
                   extent=extent, cmap=cmap, vmin=0, vmax=1)

    # add phase arrows if provided
    if phase is not None:
        # subsample for clarity
        freq_indices = np.arange(0, n_freqs, phase_step)
        time_indices = np.arange(0, n_times, phase_step)

        for i in freq_indices:
            for j in time_indices:
                if coherence[i, j] > 0.5:  # only show arrows where coherence is high
                    # arrow direction based on phase
                    dx = np.sin(phase[i, j])
                    dy = np.cos(phase[i, j])

                    t = time_axis[j]
                    f = frequencies[i]

                    # scale arrow size
                    arrow_scale = 0.5 * (time_axis[-1] - time_axis[0]) / n_times * phase_step

                    ax.arrow(t, f, dx * arrow_scale, dy * arrow_scale,
                            head_width=0.1, head_length=0.1,
                            fc='white', ec='white', alpha=0.7, linewidth=0.5)

    # labels and colorbar
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Wavelet Coherence')

    cbar = plt.colorbar(im, ax=ax, label='Coherence')

    plt.tight_layout()

    return fig, ax
