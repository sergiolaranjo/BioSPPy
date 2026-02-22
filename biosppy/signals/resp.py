# -*- coding: utf-8 -*-
"""
biosppy.signals.resp
--------------------

This module provides methods to process Respiration (Resp) signals.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function

# 3rd party
import numpy as np
import scipy.signal as ss
from scipy.interpolate import interp1d

# local
from . import tools as st
from .. import plotting, utils


def resp(signal=None, sampling_rate=1000., units=None, path=None, show=True):
    """Process a raw Respiration signal and extract relevant signal features
    using default parameters.

    Parameters
    ----------
    signal : array
        Raw Respiration signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    units : str, optional
        The units of the input signal. If specified, the plot will have the
        y-axis labeled with the corresponding units.
    path : str, optional
        If provided, the plot will be saved to the specified file.
    show : bool, optional
        If True, show a summary plot.

    Returns
    -------
    ts : array
        Signal time axis reference (seconds).
    filtered : array
        Filtered Respiration signal.
    zeros : array
        Indices of Respiration zero crossings.
    resp_rate_ts : array
        Respiration rate time axis reference (seconds).
    resp_rate : array
        Instantaneous respiration rate (Hz).

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # ensure numpy
    signal = np.array(signal)

    sampling_rate = float(sampling_rate)

    # filter signal
    filtered, _, _ = st.filter_signal(signal=signal,
                                      ftype='butter',
                                      band='bandpass',
                                      order=2,
                                      frequency=[0.1, 0.35],
                                      sampling_rate=sampling_rate)

    # compute zero crossings
    zeros, = st.zero_cross(signal=filtered, detrend=True)
    beats = zeros[::2]

    if len(beats) < 2:
        rate_idx = []
        rate = []
    else:
        # compute respiration rate
        rate_idx = beats[1:]
        rate = sampling_rate * (1. / np.diff(beats))

        # physiological limits
        indx = np.nonzero(rate <= 0.35)
        rate_idx = rate_idx[indx]
        rate = rate[indx]

        # smooth with moving average
        size = 3
        rate, _ = st.smoother(signal=rate,
                              kernel='boxcar',
                              size=size,
                              mirror=True)

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)
    ts_rate = ts[rate_idx]

    # plot
    if show:
        plotting.plot_resp(ts=ts,
                           raw=signal,
                           filtered=filtered,
                           zeros=zeros,
                           resp_rate_ts=ts_rate,
                           resp_rate=rate,
                           units=units,
                           path=path,
                           show=True)

    # output
    args = (ts, filtered, zeros, ts_rate, rate)
    names = ('ts', 'filtered', 'zeros', 'resp_rate_ts', 'resp_rate')

    return utils.ReturnTuple(args, names)


def find_extrema(signal=None, sampling_rate=1000.):
    """Detect respiratory peaks (inspiration maxima) and troughs (expiration
    minima) from a filtered respiratory signal.

    Parameters
    ----------
    signal : array
        Filtered respiratory signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    peaks : array
        Indices of inspiration peaks.
    troughs : array
        Indices of expiration troughs.

    Notes
    -----
    * Uses scipy.signal.find_peaks with a minimum distance constraint
      based on a physiological minimum breath duration (~1 second).
    * Peaks correspond to maximum inspiration; troughs to maximum expiration.
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal, dtype=float)
    sampling_rate = float(sampling_rate)

    # minimum breath duration ~ 1 second (60 breaths/min max)
    min_distance = int(sampling_rate * 1.0)
    if min_distance < 1:
        min_distance = 1

    # find peaks (inspiration maxima)
    peaks, _ = ss.find_peaks(signal, distance=min_distance)

    # find troughs (expiration minima) by inverting signal
    troughs, _ = ss.find_peaks(-signal, distance=min_distance)

    # output
    args = (peaks, troughs)
    names = ('peaks', 'troughs')

    return utils.ReturnTuple(args, names)


def breath_metrics(signal=None, sampling_rate=1000., peaks=None,
                   troughs=None):
    """Compute breath-by-breath respiratory metrics from a filtered
    respiratory signal.

    Parameters
    ----------
    signal : array
        Filtered respiratory signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    peaks : array, optional
        Indices of inspiration peaks. If None, computed automatically.
    troughs : array, optional
        Indices of expiration troughs. If None, computed automatically.

    Returns
    -------
    resp_rate : float
        Mean respiratory rate (breaths per minute).
    resp_rate_std : float
        Standard deviation of respiratory rate (breaths per minute).
    ti_mean : float
        Mean inspiratory time (seconds).
    te_mean : float
        Mean expiratory time (seconds).
    ti_te_ratio : float
        Mean ratio of inspiratory to expiratory time.
    duty_cycle : float
        Mean duty cycle Ti / Ttot.
    breath_intervals : array
        Breath-to-breath intervals (seconds).
    breath_amplitudes : array
        Peak-to-trough amplitudes for each breath.
    resp_rate_cv : float
        Coefficient of variation of respiratory rate.

    References
    ----------
    Bruce EN. Temporal variations in the pattern of breathing.
    J Appl Physiol. 1996;80(4):1079-1087.
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal, dtype=float)
    sampling_rate = float(sampling_rate)

    # detect extrema if not provided
    if peaks is None or troughs is None:
        extrema = find_extrema(signal=signal, sampling_rate=sampling_rate)
        peaks = extrema['peaks']
        troughs = extrema['troughs']

    if len(peaks) < 2 or len(troughs) < 2:
        raise ValueError("Not enough peaks/troughs for breath analysis.")

    # compute breath intervals from consecutive peaks
    breath_intervals = np.diff(peaks) / sampling_rate  # seconds

    # respiratory rate (breaths per minute)
    resp_rates = 60.0 / breath_intervals
    resp_rate = np.mean(resp_rates)
    resp_rate_std = np.std(resp_rates)
    resp_rate_cv = resp_rate_std / resp_rate if resp_rate > 0 else 0.0

    # compute inspiratory and expiratory times
    # For each trough-to-peak: inspiratory time (Ti)
    # For each peak-to-trough: expiratory time (Te)
    ti_values = []
    te_values = []
    breath_amplitudes = []

    for i in range(len(peaks) - 1):
        peak_idx = peaks[i]

        # find preceding trough (start of inspiration)
        prev_troughs = troughs[troughs < peak_idx]
        if len(prev_troughs) == 0:
            continue
        trough_before = prev_troughs[-1]

        # find following trough (end of expiration)
        next_troughs = troughs[troughs > peak_idx]
        if len(next_troughs) == 0:
            continue
        trough_after = next_troughs[0]

        # inspiratory time: trough to peak
        ti = (peak_idx - trough_before) / sampling_rate
        # expiratory time: peak to next trough
        te = (trough_after - peak_idx) / sampling_rate

        if ti > 0 and te > 0:
            ti_values.append(ti)
            te_values.append(te)
            # breath amplitude
            amplitude = signal[peak_idx] - signal[trough_before]
            breath_amplitudes.append(amplitude)

    ti_values = np.array(ti_values)
    te_values = np.array(te_values)
    breath_amplitudes = np.array(breath_amplitudes)

    ti_mean = np.mean(ti_values) if len(ti_values) > 0 else np.nan
    te_mean = np.mean(te_values) if len(te_values) > 0 else np.nan
    ti_te_ratio = ti_mean / te_mean if te_mean > 0 else np.nan
    ttot = ti_mean + te_mean
    duty_cycle = ti_mean / ttot if ttot > 0 else np.nan

    # output
    args = (resp_rate, resp_rate_std, ti_mean, te_mean, ti_te_ratio,
            duty_cycle, breath_intervals, breath_amplitudes, resp_rate_cv)
    names = ('resp_rate', 'resp_rate_std', 'ti_mean', 'te_mean',
             'ti_te_ratio', 'duty_cycle', 'breath_intervals',
             'breath_amplitudes', 'resp_rate_cv')

    return utils.ReturnTuple(args, names)


def respiratory_sinus_arrhythmia(rri=None, resp_signal=None,
                                 sampling_rate_resp=1000.,
                                 sampling_rate_rri=None,
                                 rpeaks=None, method='peak_valley'):
    """Compute Respiratory Sinus Arrhythmia (RSA) from RR intervals and
    a respiratory signal.

    RSA is the heart rate modulation by respiration: heart rate increases
    during inspiration and decreases during expiration. It is a marker
    of vagal (parasympathetic) cardiac control.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    resp_signal : array
        Filtered respiratory signal.
    sampling_rate_resp : int, float, optional
        Sampling frequency of respiratory signal (Hz).
    sampling_rate_rri : int, float, optional
        Sampling frequency of original ECG (Hz). Required if rpeaks given.
    rpeaks : array, optional
        R-peak indices. Used to align RRI with respiratory phase.
    method : str, optional
        RSA computation method: 'peak_valley' or 'porges_bohrer'.
        Default: 'peak_valley'.

    Returns
    -------
    rsa : float
        RSA magnitude (ms). Peak-valley: mean difference between max and
        min RRI within each breath. Porges-Bohrer: amplitude of filtered
        HRV at respiratory frequency.
    rsa_values : array
        Per-breath RSA values (peak_valley method only).

    References
    ----------
    Grossman P, Taylor EW. Toward understanding respiratory sinus arrhythmia:
    relations to cardiac vagal tone, evolution and biobehavioral functions.
    Biol Psychol. 2007;74(2):263-285.

    Lewis GF, Furman SA, McCool MF, Porges SW. Statistical strategies to
    quantify respiratory sinus arrhythmia. Biol Psychol. 2012;91(1):21-28.
    """

    # check inputs
    if rri is None:
        raise TypeError("Please specify RR intervals.")
    if resp_signal is None:
        raise TypeError("Please specify a respiratory signal.")

    rri = np.array(rri, dtype=float)
    resp_signal = np.array(resp_signal, dtype=float)
    sampling_rate_resp = float(sampling_rate_resp)

    if method == 'peak_valley':
        # detect respiratory peaks and troughs
        extrema = find_extrema(signal=resp_signal,
                               sampling_rate=sampling_rate_resp)
        resp_peaks = extrema['peaks']
        resp_troughs = extrema['troughs']

        if len(resp_peaks) < 2 or len(resp_troughs) < 2:
            raise ValueError("Not enough respiratory cycles for RSA.")

        # create a time axis for RRI
        rri_times = np.cumsum(rri) / 1000.0  # seconds
        rri_times -= rri_times[0]

        # respiratory time axis
        resp_times = np.arange(len(resp_signal)) / sampling_rate_resp

        # for each respiratory cycle (trough to trough), find max and min RRI
        rsa_values = []

        for i in range(len(resp_troughs) - 1):
            t_start = resp_times[resp_troughs[i]]
            t_end = resp_times[resp_troughs[i + 1]]

            # find RRI within this breath cycle
            mask = (rri_times >= t_start) & (rri_times < t_end)
            rri_in_breath = rri[mask]

            if len(rri_in_breath) >= 2:
                rsa_val = np.max(rri_in_breath) - np.min(rri_in_breath)
                rsa_values.append(rsa_val)

        rsa_values = np.array(rsa_values)
        rsa = np.mean(rsa_values) if len(rsa_values) > 0 else np.nan

    elif method == 'porges_bohrer':
        # Porges-Bohrer method: bandpass RRI at respiratory frequency
        # Estimate respiratory frequency from the signal
        extrema = find_extrema(signal=resp_signal,
                               sampling_rate=sampling_rate_resp)
        resp_peaks = extrema['peaks']

        if len(resp_peaks) < 3:
            raise ValueError("Not enough respiratory peaks for RSA.")

        # mean respiratory frequency
        resp_intervals = np.diff(resp_peaks) / sampling_rate_resp
        resp_freq = 1.0 / np.mean(resp_intervals)

        # bandpass RRI around respiratory frequency (+-30%)
        freq_low = resp_freq * 0.7
        freq_high = resp_freq * 1.3

        # resample RRI to uniform sampling
        rri_times = np.cumsum(rri) / 1000.0
        rri_times -= rri_times[0]
        frs = 4.0  # 4 Hz resampling
        t_uniform = np.arange(rri_times[0], rri_times[-1], 1.0 / frs)

        if len(t_uniform) < 10:
            raise ValueError("RRI too short for Porges-Bohrer RSA.")

        rri_interp = interp1d(rri_times, rri, kind='cubic',
                              fill_value='extrapolate')
        rri_uniform = rri_interp(t_uniform)

        # bandpass filter
        nyq = frs / 2.0
        if freq_low / nyq < 0.01:
            freq_low = 0.01 * nyq
        if freq_high / nyq >= 1.0:
            freq_high = 0.99 * nyq

        b, a = ss.butter(2, [freq_low / nyq, freq_high / nyq],
                         btype='bandpass')
        rri_filtered = ss.filtfilt(b, a, rri_uniform)

        # RSA = variance of the bandpass-filtered RRI
        rsa = np.log(np.var(rri_filtered))  # ln(ms^2) as per Porges
        rsa_values = rri_filtered

    else:
        raise ValueError(f"Unknown RSA method: '{method}'. "
                         "Use 'peak_valley' or 'porges_bohrer'.")

    # output
    args = (rsa, rsa_values)
    names = ('rsa', 'rsa_values')

    return utils.ReturnTuple(args, names)


def ecg_derived_respiration(ecg_signal=None, rpeaks=None,
                            sampling_rate=1000., method='amplitude'):
    """Derive a respiratory signal from ECG using R-wave modulation.

    The respiratory cycle modulates the ECG signal in several ways: changes
    in thoracic impedance affect R-wave amplitude, and respiratory sinus
    arrhythmia modulates the RR intervals.

    Parameters
    ----------
    ecg_signal : array
        Raw or filtered ECG signal.
    rpeaks : array
        R-peak location indices.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    method : str, optional
        EDR method: 'amplitude' (R-wave amplitude modulation) or
        'interval' (RR interval modulation). Default: 'amplitude'.

    Returns
    -------
    edr : array
        ECG-derived respiratory signal, resampled to the original
        sampling rate.
    edr_times : array
        Time axis for EDR signal (seconds).
    resp_rate : float
        Estimated respiratory rate (breaths per minute).

    References
    ----------
    Moody GB, Mark RG, Zoccola A, Mantero S. Derivation of respiratory
    signals from multi-lead ECGs. Computers in Cardiology. 1985;12:113-116.
    """

    # check inputs
    if rpeaks is None:
        raise TypeError("Please specify R-peak locations.")
    if ecg_signal is None and method == 'amplitude':
        raise TypeError("Please specify an ECG signal for amplitude method.")

    rpeaks = np.array(rpeaks, dtype=int)
    sampling_rate = float(sampling_rate)

    if len(rpeaks) < 4:
        raise ValueError("Need at least 4 R-peaks for EDR.")

    if method == 'amplitude':
        ecg_signal = np.array(ecg_signal, dtype=float)

        # extract R-wave amplitudes
        r_amplitudes = ecg_signal[rpeaks]

        # time of each R-peak
        r_times = rpeaks / sampling_rate

        # interpolate to uniform sampling at 4 Hz (sufficient for respiration)
        frs = 4.0
        t_uniform = np.arange(r_times[0], r_times[-1], 1.0 / frs)
        edr_interp = interp1d(r_times, r_amplitudes, kind='cubic',
                              fill_value='extrapolate')
        edr = edr_interp(t_uniform)

        # bandpass filter to respiratory range (0.1-0.5 Hz)
        nyq = frs / 2.0
        low = 0.1 / nyq
        high = min(0.5 / nyq, 0.99)
        b, a = ss.butter(2, [low, high], btype='bandpass')
        edr = ss.filtfilt(b, a, edr)

        edr_times = t_uniform

    elif method == 'interval':
        # RR interval modulation
        rri = np.diff(rpeaks) / sampling_rate * 1000.0  # ms
        r_times = rpeaks[1:] / sampling_rate

        # interpolate to uniform sampling at 4 Hz
        frs = 4.0
        t_uniform = np.arange(r_times[0], r_times[-1], 1.0 / frs)
        rri_interp = interp1d(r_times, rri, kind='cubic',
                              fill_value='extrapolate')
        edr = rri_interp(t_uniform)

        # bandpass filter to respiratory range
        nyq = frs / 2.0
        low = 0.1 / nyq
        high = min(0.5 / nyq, 0.99)
        b, a = ss.butter(2, [low, high], btype='bandpass')
        edr = ss.filtfilt(b, a, edr)

        edr_times = t_uniform

    else:
        raise ValueError(f"Unknown EDR method: '{method}'. "
                         "Use 'amplitude' or 'interval'.")

    # estimate respiratory rate from EDR
    extrema = find_extrema(signal=edr,
                           sampling_rate=1.0 / (edr_times[1] - edr_times[0]))
    edr_peaks = extrema['peaks']
    if len(edr_peaks) >= 2:
        peak_intervals = np.diff(edr_peaks) * (edr_times[1] - edr_times[0])
        resp_rate = 60.0 / np.mean(peak_intervals)
    else:
        resp_rate = np.nan

    # output
    args = (edr, edr_times, resp_rate)
    names = ('edr', 'edr_times', 'resp_rate')

    return utils.ReturnTuple(args, names)


def ppg_derived_respiration(ppg_signal=None, peaks=None, onsets=None,
                            sampling_rate=1000., method='riav'):
    """Derive a respiratory signal from PPG using respiratory-induced
    modulations.

    Parameters
    ----------
    ppg_signal : array
        Filtered PPG signal.
    peaks : array
        PPG systolic peak indices.
    onsets : array, optional
        PPG onset indices (for RIAV method).
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    method : str, optional
        Derivation method:
        - 'riiv': Respiratory Induced Intensity Variation (baseline)
        - 'riav': Respiratory Induced Amplitude Variation
        - 'rifv': Respiratory Induced Frequency Variation
        Default: 'riav'.

    Returns
    -------
    pdr : array
        PPG-derived respiratory signal.
    pdr_times : array
        Time axis for PDR signal (seconds).
    resp_rate : float
        Estimated respiratory rate (breaths per minute).

    References
    ----------
    Charlton PH, Bonnici T, Tarassenko L, et al. An assessment of algorithms
    to estimate respiratory rate from the electrocardiogram and
    photoplethysmogram. Physiol Meas. 2016;37(4):610-626.
    """

    # check inputs
    if ppg_signal is None:
        raise TypeError("Please specify a PPG signal.")
    if peaks is None:
        raise TypeError("Please specify PPG peak locations.")

    ppg_signal = np.array(ppg_signal, dtype=float)
    peaks = np.array(peaks, dtype=int)
    sampling_rate = float(sampling_rate)

    if len(peaks) < 4:
        raise ValueError("Need at least 4 PPG peaks for PDR.")

    peak_times = peaks / sampling_rate

    if method == 'riiv':
        # Respiratory Induced Intensity Variation
        # Baseline modulation of PPG signal at peak locations
        peak_values = ppg_signal[peaks]

        # interpolate to uniform sampling
        frs = 4.0
        t_uniform = np.arange(peak_times[0], peak_times[-1], 1.0 / frs)
        pdr_interp = interp1d(peak_times, peak_values, kind='cubic',
                              fill_value='extrapolate')
        pdr = pdr_interp(t_uniform)

    elif method == 'riav':
        # Respiratory Induced Amplitude Variation
        if onsets is None:
            raise TypeError("Onsets are required for RIAV method.")
        onsets = np.array(onsets, dtype=int)

        # compute pulse amplitudes (peak - onset)
        amplitudes = []
        valid_times = []
        for peak_idx in peaks:
            prev_onsets = onsets[onsets < peak_idx]
            if len(prev_onsets) == 0:
                continue
            onset_idx = prev_onsets[-1]
            amp = ppg_signal[peak_idx] - ppg_signal[onset_idx]
            amplitudes.append(amp)
            valid_times.append(peak_idx / sampling_rate)

        amplitudes = np.array(amplitudes)
        valid_times = np.array(valid_times)

        if len(amplitudes) < 4:
            raise ValueError("Not enough valid pulses for RIAV.")

        # interpolate
        frs = 4.0
        t_uniform = np.arange(valid_times[0], valid_times[-1], 1.0 / frs)
        pdr_interp = interp1d(valid_times, amplitudes, kind='cubic',
                              fill_value='extrapolate')
        pdr = pdr_interp(t_uniform)

    elif method == 'rifv':
        # Respiratory Induced Frequency Variation
        # pulse rate modulation
        pulse_intervals = np.diff(peaks) / sampling_rate  # seconds
        pulse_times = peak_times[1:]

        if len(pulse_intervals) < 4:
            raise ValueError("Not enough pulses for RIFV.")

        # interpolate
        frs = 4.0
        t_uniform = np.arange(pulse_times[0], pulse_times[-1], 1.0 / frs)
        pdr_interp = interp1d(pulse_times, pulse_intervals, kind='cubic',
                              fill_value='extrapolate')
        pdr = pdr_interp(t_uniform)

    else:
        raise ValueError(f"Unknown PDR method: '{method}'. "
                         "Use 'riiv', 'riav', or 'rifv'.")

    pdr_times = t_uniform

    # bandpass filter to respiratory range (0.1-0.5 Hz)
    frs = 1.0 / (pdr_times[1] - pdr_times[0])
    nyq = frs / 2.0
    low = 0.1 / nyq
    high = min(0.5 / nyq, 0.99)

    if low < high and len(pdr) > 12:  # need enough samples for filter
        b, a = ss.butter(2, [low, high], btype='bandpass')
        pdr = ss.filtfilt(b, a, pdr)

    # estimate respiratory rate
    edr_sr = 1.0 / (pdr_times[1] - pdr_times[0])
    extrema = find_extrema(signal=pdr, sampling_rate=edr_sr)
    edr_peaks = extrema['peaks']
    if len(edr_peaks) >= 2:
        peak_intervals = np.diff(edr_peaks) / edr_sr
        resp_rate = 60.0 / np.mean(peak_intervals)
    else:
        resp_rate = np.nan

    # output
    args = (pdr, pdr_times, resp_rate)
    names = ('pdr', 'pdr_times', 'resp_rate')

    return utils.ReturnTuple(args, names)
