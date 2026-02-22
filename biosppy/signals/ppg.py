# -*- coding: utf-8 -*-
"""
biosppy.signals.ppg
-------------------

This module provides methods to process Photoplethysmogram (PPG) signals.

:copyright: (c) 2015-2018 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
from six.moves import range

# 3rd party
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

# local
from . import tools as st
from .. import plotting, utils


def ppg(signal=None, sampling_rate=1000., units=None, path=None, show=True):
    """Process a raw PPG signal and extract relevant signal features using
    default parameters.

    Parameters
    ----------
    signal : array
        Raw PPG signal.
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
        Filtered PPG signal.
    peaks : array
        Indices of PPG pulse peaks.
    templates_ts : array
        Templates time axis reference (seconds).
    templates : array
        Extracted heartbeat templates.
    heart_rate_ts : array
        Heart rate time axis reference (seconds).
    heart_rate : array
        Instantaneous heart rate (bpm).

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
                                      order=4,
                                      frequency=[1, 8],
                                      sampling_rate=sampling_rate)

    # find peaks
    peaks, _ = find_onsets_elgendi2013(signal=filtered,
                                       sampling_rate=sampling_rate)

    # extract templates
    onsets, peaks, segments_loc = ppg_segmentation(filtered, sampling_rate,
                                                   peaks)
    templates_ts, templates = _extract_templates(filtered, sampling_rate,
                                                 onsets, peaks, segments_loc)

    # compute heart rate
    hr_idx, hr = st.get_heart_rate(beats=onsets,
                                   sampling_rate=sampling_rate,
                                   smooth=True,
                                   size=3)

    # get time vectors
    length = len(signal)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)
    ts_hr = ts[hr_idx]

    # plot
    if show:
        plotting.plot_ppg(ts=ts,
                          raw=signal,
                          filtered=filtered,
                          peaks=peaks,
                          templates_ts=templates_ts,
                          templates=templates,
                          heart_rate_ts=ts_hr,
                          heart_rate=hr,
                          units=units,
                          path=path,
                          show=True)

    # output
    args = (ts, filtered, peaks, templates_ts, templates,
            ts_hr, hr)
    names = ('ts', 'filtered', 'peaks', 'templates_ts', 'templates',
             'heart_rate_ts', 'heart_rate')

    return utils.ReturnTuple(args, names)


def find_onsets_elgendi2013(signal=None, sampling_rate=1000., peakwindow=0.111, beatwindow=0.667, beatoffset=0.02,
                            mindelay=0.3):
    """
    Determines onsets of PPG pulses.

    Parameters
    ----------
    signal : array
        Input filtered PPG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    peakwindow : float
        Parameter W1 on referenced article
        Optimized at 0.111
    beatwindow : float
        Parameter W2 on referenced article
        Optimized at 0.667
    beatoffset : float
        Parameter beta on referenced article
        Optimized at 0.2
    mindelay : float
        Minimum delay between peaks.
        Avoids false positives

    Returns
    ----------
    onsets : array
        Indices of PPG pulse onsets.
    params : dict
        Input parameters of the function


    References
    ----------
    - Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D (2013) Systolic Peak Detection in
    Acceleration Photoplethysmograms Measured from Emergency Responders in Tropical Conditions.
    PLoS ONE 8(10): e76585. doi:10.1371/journal.pone.0076585.
    
    Notes
    ---------------------
    Optimal ranges for signal filtering (from Elgendi et al. 2013):
    "Optimization of the beat detector’s spectral window for the lower frequency resulted in a 
    value within 0.5– 1 Hz with the higher frequency within 7–15 Hz"
    
    All the number references below between curly brackets {...} by the code refer to the line numbers of
    code in "Table 2 Algorithm IV: DETECTOR (PPG signal, F1, F2, W1, W2, b)" from Elgendi et al. 2013 for a
    better comparison of the algorithm
    
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    # Create copy of signal (not to modify the original object)
    signal_copy = np.copy(signal)

    # Truncate to zero and square
    # {3, 4}
    signal_copy[signal_copy < 0] = 0
    squared_signal = signal_copy ** 2

    # Calculate peak detection threshold
    # {5}
    ma_peak_kernel = int(np.rint(peakwindow * sampling_rate))
    ma_peak, _ = st.smoother(squared_signal, kernel="boxcar", size=ma_peak_kernel)

    # {6}
    ma_beat_kernel = int(np.rint(beatwindow * sampling_rate))
    ma_beat, _ = st.smoother(squared_signal, kernel="boxcar", size=ma_beat_kernel)

    # Calculate threshold value
    # {7, 8, 9}
    thr1 = ma_beat + beatoffset * np.mean(squared_signal)

    # Identify start and end of PPG waves.
    # {10-16}
    waves = ma_peak > thr1
    beg_waves = np.where(np.logical_and(np.logical_not(waves[0:-1]), waves[1:]))[0]
    end_waves = np.where(np.logical_and(waves[0:-1], np.logical_not(waves[1:])))[0]
    # Throw out wave-ends that precede first wave-start.
    end_waves = end_waves[end_waves > beg_waves[0]]

    # Identify systolic peaks within waves (ignore waves that are too short).
    num_waves = min(beg_waves.size, end_waves.size)
    # {18}
    min_len = int(np.rint(peakwindow * sampling_rate))
    min_delay = int(np.rint(mindelay * sampling_rate))
    onsets = [0]

    # {19}
    for i in range(num_waves):

        beg = beg_waves[i]
        end = end_waves[i]
        len_wave = end - beg

        # {20, 22, 23}
        if len_wave < min_len:
            continue

        # Find local maxima and their prominence within wave span.
        # {21}
        data = signal_copy[beg:end]
        locmax, props = ss.find_peaks(data, prominence=(None, None))

        # If more than one peak
        if locmax.size > 0:
            # Identify most prominent local maximum.
            peak = beg + locmax[np.argmax(props["prominences"])]
            # Enforce minimum delay between onsets.
            if peak - onsets[-1] > min_delay:
                onsets.append(peak)

    onsets.pop(0)
    onsets = np.array(onsets, dtype='int')

    # output
    params = {'signal': signal, 'sampling_rate': sampling_rate, 'peakwindow': peakwindow, 'beatwindow': beatwindow, 'beatoffset': beatoffset, 'mindelay': mindelay}

    args = (onsets, params)
    names = ('onsets', 'params')

    return utils.ReturnTuple(args, names)


def find_onsets_kavsaoglu2016(
    signal=None,
    sampling_rate=1000.0,
    alpha=0.2,
    k=4,
    init_bpm=90,
    min_delay=0.6,
    max_BPM=150,
):
    """
    Determines onsets of PPG pulses.

    Parameters
    ----------
    signal : array
        Input filtered PPG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    alpha : float, optional
        Low-pass filter factor.
        Avoids abrupt changes of BPM.
    k : int, float, optional
        Number of segments by pulse.
        Width of each segment = Period of pulse according to current BPM / k
    init_bpm : int, float, optional
        Initial BPM.
        Higher value results in a smaller segment width.
    min_delay : float
        Minimum delay between peaks as percentage of current BPM pulse period.
        Avoids false positives
    max_bpm : int, float, optional
        Maximum BPM.
        Maximum value accepted as valid BPM.

    Returns
    ----------
    onsets : array
        Indices of PPG pulse onsets.
    window_marks : array
        Indices of segments window boundaries.
    params : dict
        Input parameters of the function


    References
    ----------
    - Kavsaoğlu, Ahmet & Polat, Kemal & Bozkurt, Mehmet. (2016). An innovative peak detection algorithm for
    photoplethysmography signals: An adaptive segmentation method. TURKISH JOURNAL OF ELECTRICAL ENGINEERING
    & COMPUTER SCIENCES. 24. 1782-1796. 10.3906/elk-1310-177.

    Notes
    ---------------------
    This algorithm is an adaption of the one described on Kavsaoğlu et al. (2016).
    This version takes into account a minimum delay between peaks and builds upon the adaptive segmentation
    by using a low-pass filter for BPM changes. This way, even if the algorithm wrongly detects a peak, the
    BPM value will stay relatively constant so the next pulse can be correctly segmented.

    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if alpha <= 0 or alpha > 1:
        raise TypeError("The value of alpha must be in the range: ]0, 1].")

    if k <= 0:
        raise TypeError("The number of divisions by pulse should be greater than 0.")

    if init_bpm <= 0:
        raise TypeError("Provide a valid BPM value for initial estimation.")

    if min_delay < 0 or min_delay > 1:
        raise TypeError(
            "The minimum delay percentage between peaks must be between 0 and 1"
        )

    if max_BPM >= 248:
        raise TypeError("The maximum BPM must assure the person is alive")

    # current bpm
    bpm = init_bpm

    # current segment window width
    window = int(sampling_rate * (60 / bpm) / k)

    # onsets array
    onsets = []

    # window marks array - stores the boundaries of each segment
    window_marks = []

    # buffer for peak indices
    idx_buffer = [-1, -1, -1]

    # buffer to store the previous 3 values for onset detection
    min_buffer = [0, 0, 0]

    # signal pointer
    i = 0
    while i + window < len(signal):
        # remove oldest values
        idx_buffer.pop(0)
        min_buffer.pop(0)

        # add the index of the minimum value of the current segment to buffer
        idx_buffer.append(int(i + np.argmin(signal[i : i + window])))

        # add the minimum value of the current segment to buffer
        min_buffer.append(signal[idx_buffer[-1]])

        if (
            # the buffer has to be filled with valid values
            idx_buffer[0] > -1
            # the center value of the buffer must be smaller than its neighbours
            and (min_buffer[1] < min_buffer[0] and min_buffer[1] <= min_buffer[2])
            # if an onset was previously detected, guarantee that the new onset respects the minimum delay, minimum BPM and maximum BPM
            and (
                len(onsets) == 0
                or (
                    (idx_buffer[1] - onsets[-1]) / sampling_rate >= min_delay * 60 / bpm
                    and (idx_buffer[1] - onsets[-1]) / sampling_rate > 60 / max_BPM
                )
            )
        ):
            # store the onset
            onsets.append(idx_buffer[1])

            # if more than one onset was detected, update the bpm and the segment width
            if len(onsets) > 1:
                # calculate new bpm from the latest two onsets
                new_bpm = int(60 * sampling_rate / (onsets[-1] - onsets[-2]))

                # update the bpm value
                bpm = alpha * new_bpm + (1 - alpha) * bpm

                # update the segment window width
                window = int(sampling_rate * (60 / bpm) / k)

        # update the signal pointer
        i += window

        # store window segment boundaries index
        window_marks.append(i)

    onsets = np.array(onsets, dtype="int")
    window_marks = np.array(window_marks, dtype="int")

    # output
    params = {
        "signal": signal,
        "sampling_rate": sampling_rate,
        "alpha": alpha,
        "k": k,
        "init_bpm": init_bpm,
        "min_delay": min_delay,
        "max_bpm": max_BPM,
    }

    args = (onsets, window_marks, params)
    names = ("onsets", "window_marks", "params")

    return utils.ReturnTuple(args, names)


def ppg_segmentation(signal=None,
                     sampling_rate=1000.,
                     peaks=None,
                     selection=False,
                     peak_threshold=None):
    """Segments a filtered PPG signal. Segmentation filtering is achieved by
    taking into account segments selected by peak height and pulse morphology.

    Parameters
    ----------
    signal : array
        Filtered PPG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    peaks : array
        List of PPG systolic peaks.
    selection : bool, optional
        If True, performs selection with peak height and pulse morphology.
    peak_threshold : int, float, optional
        If `selection` is True, selects peaks with height greater than defined
        threshold.

    Returns
    -------
    onsets : array
        Indices of PPG pulse onsets (i.e., start of beats) of the selected
        segments.
    peaks : array
        List of PPG systolic peaks of the selected segments.
    segments_loc : array
        Start and end indices for each selected pulse segment.

    """

    # check inputs
    if signal is None or peaks is None:
        raise TypeError("Please check inputs.")

    # ensure input format
    signal = np.array(signal)
    sampling_rate = float(sampling_rate)

    # find onsets
    onsets = []
    minima = (np.diff(np.sign(np.diff(signal))) > 0).nonzero()[0]
    for ind in peaks:
        candidates = minima[minima < ind]
        if len(candidates) == 0:
            continue
        onsets.append(candidates.max())
    onsets = np.array(onsets, dtype='int')

    # raise error if onset detection failed
    if len(onsets) == 0:
        raise TypeError("No onsets detected.")

    # assign start and end of each segment
    segments_loc = np.vstack((onsets[:-1], onsets[1:])).T

    # segment selection by morphology
    if selection:
        segments_sel = []
        for ind in range(segments_loc.shape[0]):
            # search segments with at least 4 max+min (standard waveform)
            segment = signal[segments_loc[ind, 0]: segments_loc[ind, 1]]
            if len(np.where(np.diff(np.sign(np.diff(segment))))[0]) >= 4:
                segments_sel.append(ind)

        segments_loc = segments_loc[segments_sel]
        onsets = onsets[segments_sel]
        peaks = peaks[segments_sel]

    # segment selection by height
    if peak_threshold is not None:
        segments_sel = []
        for ind in range(segments_loc.shape[0]):
            # search segments with peak higher than threshold
            segment = signal[segments_loc[ind, 0]: segments_loc[ind, 1]]
            if max(segment) > peak_threshold:
                segments_sel.append(ind)

        segments_loc = segments_loc[segments_sel]
        onsets = onsets[segments_sel]
        peaks = peaks[segments_sel]

    # output
    args = (onsets, peaks, segments_loc)
    names = ('onsets', 'peaks', 'segments_loc')

    return utils.ReturnTuple(args, names)


def _extract_templates(signal=None,
                      sampling_rate=1000.,
                      onsets=None,
                      peaks=None,
                      segments_loc=None):
    """Extracts the templates from the PPG signal, which are aligned with their
    systolic peaks. To achieve this, the segments are padded with NaNs. Should
    be used in combination with signals.ppg.ppg_segmentation.

    Parameters
    ----------
    signal : array
        Filtered PPG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    onsets : array
        List of onsets (i.e., start of beats) of the PPG waves.
    peaks : array
        List of PPG systolic peaks.
    segments_loc : array
        Start and end indices for each selected pulse segment.

    Returns
    -------
    templates_ts : array
        Time axis common to all templates.
    templates : array
        List of templates aligned with the systolic peaks.

    """
    # initialize output
    templates = []

    # find the longest onset-peak duration
    shifts = peaks - onsets
    max_shift = np.max(peaks - onsets)

    # left padding
    max_len = 0
    for i in range(len(segments_loc)):
        segment = signal[segments_loc[i, 0]: segments_loc[i, 1]]
        segment = np.pad(segment, max_shift - shifts[i], mode='constant',
                         constant_values=(np.nan,))
        templates.append(segment)

        # find the largest segment
        if len(segment) > max_len:
            max_len = len(segment)

    # right padding
    for index, segment in enumerate(templates):
        templates[index] = np.pad(segment, (0, max_len - len(segment)),
                                  mode='constant', constant_values=(np.nan,))

    templates = np.asarray(templates).T

    # time vector
    templates_ts = np.arange(-max_shift, max_len - max_shift) / sampling_rate

    # output
    args = (templates_ts, templates)
    names = ('templates_ts', 'templates')

    return utils.ReturnTuple(args, names)


def find_systolic_peaks(signal=None, sampling_rate=1000., min_delay=0.3):
    """Detect systolic peaks in a filtered PPG signal.

    Parameters
    ----------
    signal : array
        Filtered PPG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    min_delay : float, optional
        Minimum delay between peaks in seconds. Default: 0.3.

    Returns
    -------
    peaks : array
        Indices of systolic peaks.
    """

    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal, dtype=float)
    sampling_rate = float(sampling_rate)

    min_distance = int(min_delay * sampling_rate)
    if min_distance < 1:
        min_distance = 1

    peaks, _ = ss.find_peaks(signal, distance=min_distance,
                             height=np.mean(signal))

    args = (peaks,)
    names = ('peaks',)
    return utils.ReturnTuple(args, names)


def find_dicrotic_notch(signal=None, peaks=None, onsets=None,
                        sampling_rate=1000.):
    """Detect dicrotic notch locations in PPG pulse waveforms.

    The dicrotic notch is the small dip following the systolic peak,
    marking the closure of the aortic valve. It separates the systolic
    and diastolic phases of the pulse wave.

    Parameters
    ----------
    signal : array
        Filtered PPG signal.
    peaks : array
        Indices of systolic peaks.
    onsets : array
        Indices of pulse onsets.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).

    Returns
    -------
    dicrotic_notches : array
        Indices of dicrotic notch locations.
    dicrotic_peaks : array
        Indices of diastolic (dicrotic) peaks following the notch.

    Notes
    -----
    * The dicrotic notch is detected as the local minimum between the
      systolic peak and the next onset.
    * The dicrotic peak is the local maximum following the notch.
    * If no clear notch is found, -1 is used for that pulse.

    References
    ----------
    Elgendi M. On the analysis of fingertip photoplethysmogram signals.
    Curr Cardiol Rev. 2012;8(1):14-25.
    """

    if signal is None or peaks is None or onsets is None:
        raise TypeError("Please specify signal, peaks, and onsets.")

    signal = np.array(signal, dtype=float)
    peaks = np.array(peaks, dtype=int)
    onsets = np.array(onsets, dtype=int)

    dicrotic_notches = []
    dicrotic_peaks = []

    for i in range(len(peaks)):
        peak_idx = peaks[i]

        # find the next onset after this peak
        next_onsets = onsets[onsets > peak_idx]
        if len(next_onsets) == 0:
            dicrotic_notches.append(-1)
            dicrotic_peaks.append(-1)
            continue

        next_onset = next_onsets[0]

        # search for dicrotic notch between peak and next onset
        segment = signal[peak_idx:next_onset]
        beat_dur = next_onset - peak_idx
        if len(segment) < 3 or beat_dur < 3:
            dicrotic_notches.append(-1)
            dicrotic_peaks.append(-1)
            continue

        # constrain search to [0.15, 0.7] of peak-to-onset duration
        # to avoid noise near the peak or onset
        search_start = int(0.15 * beat_dur)
        search_end = int(0.7 * beat_dur)
        if search_start >= search_end:
            search_start = 0
            search_end = len(segment)

        # find local minima in the constrained segment
        local_mins, _ = ss.find_peaks(-segment)

        # filter to those within the search window
        valid_mins = local_mins[(local_mins >= search_start) &
                                (local_mins <= search_end)]

        if len(valid_mins) == 0:
            # fallback: try all local mins, or use second derivative
            if len(local_mins) > 0:
                valid_mins = local_mins
            else:
                # use second derivative zero-crossing as fallback
                d2 = np.gradient(np.gradient(segment))
                zc = []
                for j in range(search_start,
                               min(search_end, len(d2) - 1)):
                    if d2[j] >= 0 and d2[j + 1] < 0:
                        zc.append(j)
                if len(zc) > 0:
                    valid_mins = np.array([zc[0]])
                else:
                    dicrotic_notches.append(-1)
                    dicrotic_peaks.append(-1)
                    continue

        if len(valid_mins) > 0:
            # select the deepest minimum in the valid range
            notch_rel = valid_mins[np.argmin(segment[valid_mins])]
            notch_idx = peak_idx + notch_rel
            dicrotic_notches.append(notch_idx)

            # find dicrotic peak: local max after the notch
            after_notch = signal[notch_idx:next_onset]
            if len(after_notch) > 1:
                local_maxs, _ = ss.find_peaks(after_notch)
                if len(local_maxs) > 0:
                    dpeak_idx = notch_idx + local_maxs[0]
                    dicrotic_peaks.append(dpeak_idx)
                else:
                    dicrotic_peaks.append(-1)
            else:
                dicrotic_peaks.append(-1)
        else:
            dicrotic_notches.append(-1)
            dicrotic_peaks.append(-1)

    dicrotic_notches = np.array(dicrotic_notches, dtype=int)
    dicrotic_peaks = np.array(dicrotic_peaks, dtype=int)

    args = (dicrotic_notches, dicrotic_peaks)
    names = ('dicrotic_notches', 'dicrotic_peaks')

    return utils.ReturnTuple(args, names)


def sdppg(signal=None, sampling_rate=1000., peaks=None, onsets=None):
    """Compute the Second Derivative of the PPG (SDPPG) and extract
    a, b, c, d, e wave features.

    The SDPPG (also called Acceleration Plethysmogram, APG) is widely used
    for vascular aging assessment. The a, b, c, d, e waves are characteristic
    peaks/troughs of the SDPPG waveform.

    Parameters
    ----------
    signal : array
        Filtered PPG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    peaks : array, optional
        Indices of systolic peaks. If None, detected automatically.
    onsets : array, optional
        Indices of pulse onsets. If None, detected automatically.

    Returns
    -------
    sdppg_signal : array
        Second derivative of the PPG signal.
    a_waves : array
        Amplitudes of 'a' waves (initial positive peak, corresponds to
        early systolic positive wave).
    b_waves : array
        Amplitudes of 'b' waves (early systolic negative wave).
    c_waves : array
        Amplitudes of 'c' waves (late systolic re-increasing wave).
    d_waves : array
        Amplitudes of 'd' waves (late systolic decreasing wave).
    e_waves : array
        Amplitudes of 'e' waves (early diastolic positive wave).
    aging_index : array
        Vascular aging index per pulse: (b - c - d - e) / a.
    aging_index_mean : float
        Mean vascular aging index.

    References
    ----------
    Takazawa K, Tanaka N, Fujita M, et al. Assessment of vasoactive agents
    and vascular aging by the second derivative of the photoplethysmogram
    waveform. Hypertension. 1998;32(2):365-370.

    Notes
    -----
    * The aging index increases with age and arterial stiffness.
    * Typical aging index ranges: -0.8 to 0.2 for healthy subjects.
    * Higher values indicate stiffer arteries.
    * The 'a' wave is always positive, 'b' is negative, c/d/e vary.
    """

    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal, dtype=float)
    sampling_rate = float(sampling_rate)

    # compute second derivative
    # smooth slightly first to reduce noise amplification
    dt = 1.0 / sampling_rate
    first_deriv = np.gradient(signal, dt)
    sdppg_signal = np.gradient(first_deriv, dt)

    # detect peaks and onsets if not provided
    if peaks is None or onsets is None:
        det = find_onsets_elgendi2013(signal=signal,
                                      sampling_rate=sampling_rate)
        if onsets is None:
            onsets = det['onsets']
        if peaks is None:
            # find peaks between onsets
            peaks_list = []
            for i in range(len(onsets) - 1):
                segment = signal[onsets[i]:onsets[i + 1]]
                if len(segment) > 0:
                    peaks_list.append(onsets[i] + np.argmax(segment))
            peaks = np.array(peaks_list, dtype=int)

    # extract a, b, c, d, e waves for each pulse
    a_waves = []
    b_waves = []
    c_waves = []
    d_waves = []
    e_waves = []

    for i in range(len(onsets) - 1):
        onset = onsets[i]
        next_onset = onsets[i + 1]

        # find the peak within this pulse
        pulse_peaks = peaks[(peaks >= onset) & (peaks < next_onset)]
        if len(pulse_peaks) == 0:
            continue

        peak = pulse_peaks[0]

        # extract SDPPG for this pulse (onset to next onset)
        pulse_sdppg = sdppg_signal[onset:next_onset]
        if len(pulse_sdppg) < 5:
            continue

        # find the 'a' wave: first positive peak of SDPPG in early systole
        # (within onset to systolic peak)
        systolic_sdppg = sdppg_signal[onset:peak]
        if len(systolic_sdppg) < 2:
            continue

        # 'a' wave: max positive peak in early systole
        a_idx = np.argmax(systolic_sdppg)
        a_val = systolic_sdppg[a_idx]

        # find subsequent extrema after the 'a' wave
        remaining_sdppg = sdppg_signal[onset + a_idx:next_onset]
        if len(remaining_sdppg) < 4:
            continue

        # find local extrema
        local_maxs, _ = ss.find_peaks(remaining_sdppg)
        local_mins, _ = ss.find_peaks(-remaining_sdppg)

        # 'b' wave: first minimum after 'a' (negative)
        if len(local_mins) > 0:
            b_val = remaining_sdppg[local_mins[0]]
        else:
            b_val = np.min(remaining_sdppg)

        # 'c' wave: next maximum after 'b'
        c_val = 0.0
        d_val = 0.0
        e_val = 0.0
        c_maxs = local_maxs[local_maxs > local_mins[0]] if len(local_mins) > 0 else local_maxs
        if len(c_maxs) > 0:
            c_val = remaining_sdppg[c_maxs[0]]

            # 'd' wave: next minimum after 'c'
            d_mins = local_mins[local_mins > c_maxs[0]] if len(local_mins) > 0 else np.array([])
            if len(d_mins) > 0:
                d_val = remaining_sdppg[d_mins[0]]

                # 'e' wave: next maximum after 'd'
                e_maxs = local_maxs[local_maxs > d_mins[0]]
                if len(e_maxs) > 0:
                    e_val = remaining_sdppg[e_maxs[0]]

        a_waves.append(a_val)
        b_waves.append(b_val)
        c_waves.append(c_val)
        d_waves.append(d_val)
        e_waves.append(e_val)

    a_waves = np.array(a_waves)
    b_waves = np.array(b_waves)
    c_waves = np.array(c_waves)
    d_waves = np.array(d_waves)
    e_waves = np.array(e_waves)

    # compute aging index: (b - c - d - e) / a
    valid = a_waves != 0
    aging_index = np.full(len(a_waves), np.nan)
    aging_index[valid] = (b_waves[valid] - c_waves[valid] -
                          d_waves[valid] - e_waves[valid]) / a_waves[valid]
    aging_index_mean = np.nanmean(aging_index) if len(aging_index) > 0 else np.nan

    args = (sdppg_signal, a_waves, b_waves, c_waves, d_waves, e_waves,
            aging_index, aging_index_mean)
    names = ('sdppg_signal', 'a_waves', 'b_waves', 'c_waves', 'd_waves',
             'e_waves', 'aging_index', 'aging_index_mean')

    return utils.ReturnTuple(args, names)


def pulse_wave_analysis(signal=None, sampling_rate=1000., peaks=None,
                        onsets=None, height=None):
    """Compute pulse wave analysis (PWA) features from a PPG signal.

    Parameters
    ----------
    signal : array
        Filtered PPG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    peaks : array, optional
        Indices of systolic peaks. If None, detected automatically.
    onsets : array, optional
        Indices of pulse onsets. If None, detected automatically.
    height : float, optional
        Subject height in meters. Required for Stiffness Index computation.

    Returns
    -------
    augmentation_index : float
        Mean Augmentation Index (AIx). Ratio of augmentation pressure to
        pulse pressure expressed as percentage.
    stiffness_index : float
        Mean Stiffness Index (SI) in m/s. Subject height divided by
        peak-to-dicrotic-notch time. Requires height parameter.
    reflection_index : float
        Mean Reflection Index (RI). Ratio of dicrotic peak amplitude to
        systolic peak amplitude.
    crest_time_mean : float
        Mean crest time (seconds). Time from onset to systolic peak.
    delta_t_mean : float
        Mean delta T (seconds). Time from systolic peak to dicrotic notch.
    pulse_width_mean : float
        Mean pulse width (seconds). Onset to next onset.
    pulse_area : array
        Area under each pulse waveform (arbitrary units).

    References
    ----------
    Millasseau SC, Kelly RP, Ritter JM, Chowienczyk PJ. Determination of
    age-related increases in large artery stiffness by digital pulse contour
    analysis. Clin Sci. 2002;103(4):371-377.

    Kelly R, Hayward C, Avolio A, O'Rourke M. Noninvasive determination of
    age-related changes in the human arterial pulse. Circulation.
    1989;80(6):1652-1659.
    """

    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal, dtype=float)
    sampling_rate = float(sampling_rate)

    # detect landmarks if not provided
    if peaks is None or onsets is None:
        det = find_onsets_elgendi2013(signal=signal,
                                      sampling_rate=sampling_rate)
        onsets = det['onsets']
        # find peaks between onsets
        peaks_list = []
        for i in range(len(onsets) - 1):
            segment = signal[onsets[i]:onsets[i + 1]]
            if len(segment) > 0:
                peaks_list.append(onsets[i] + np.argmax(segment))
        peaks = np.array(peaks_list, dtype=int)

    # detect dicrotic notch
    dn_result = find_dicrotic_notch(signal=signal, peaks=peaks,
                                     onsets=onsets,
                                     sampling_rate=sampling_rate)
    dicrotic_notches = dn_result['dicrotic_notches']
    dicrotic_peaks_arr = dn_result['dicrotic_peaks']

    # compute features per pulse
    crest_times = []
    delta_ts = []
    pulse_widths = []
    reflection_indices = []
    augmentation_indices = []
    stiffness_indices = []
    pulse_areas = []

    for i in range(min(len(peaks), len(onsets) - 1)):
        peak_idx = peaks[i]
        onset_idx = onsets[i]
        next_onset_idx = onsets[i + 1] if i + 1 < len(onsets) else None

        # crest time: onset to peak
        ct = (peak_idx - onset_idx) / sampling_rate
        crest_times.append(ct)

        # pulse width
        if next_onset_idx is not None:
            pw = (next_onset_idx - onset_idx) / sampling_rate
            pulse_widths.append(pw)

            # pulse area
            pulse = signal[onset_idx:next_onset_idx]
            _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
            area = _trapz(pulse - signal[onset_idx]) / sampling_rate
            pulse_areas.append(area)

        # delta T and related features
        if i < len(dicrotic_notches) and dicrotic_notches[i] > 0:
            dn_idx = dicrotic_notches[i]
            dt_val = (dn_idx - peak_idx) / sampling_rate
            delta_ts.append(dt_val)

            # stiffness index: SI = height / (t_diastolic_peak - t_systolic_peak)
            if i < len(dicrotic_peaks_arr) and dicrotic_peaks_arr[i] > 0:
                dp_idx = dicrotic_peaks_arr[i]
                dt_dp = (dp_idx - peak_idx) / sampling_rate
                if height is not None and dt_dp > 0:
                    si = height / dt_dp
                    stiffness_indices.append(si)

                # reflection index: RI = diastolic_amp / systolic_amp
                systolic_amp = signal[peak_idx] - signal[onset_idx]
                diastolic_amp = signal[dp_idx] - signal[onset_idx]
                if systolic_amp > 0:
                    ri = diastolic_amp / systolic_amp
                    reflection_indices.append(ri)

            # augmentation index: AIx = (P2 - P1) / PP * 100
            # P1 = amplitude at inflection point on rising phase
            # P2 = systolic peak amplitude
            # Find inflection point via second derivative zero-crossing
            systolic_amp = signal[peak_idx] - signal[onset_idx]
            if systolic_amp > 0:
                rising = signal[onset_idx:peak_idx]
                if len(rising) > 4:
                    d2_rising = np.gradient(np.gradient(rising))
                    # find last zero-crossing (positive-to-negative)
                    zc_list = []
                    for j in range(len(d2_rising) - 1):
                        if d2_rising[j] >= 0 and d2_rising[j + 1] < 0:
                            zc_list.append(j)
                    if len(zc_list) > 0:
                        ip_local = zc_list[-1]
                        p1 = signal[onset_idx + ip_local] - signal[onset_idx]
                        p2 = systolic_amp
                        pp = systolic_amp
                        aix = (p2 - p1) / pp * 100.0
                    else:
                        aix = 0.0  # no inflection point found
                else:
                    aix = 0.0
                augmentation_indices.append(aix)

    # compute means
    crest_time_mean = np.mean(crest_times) if crest_times else np.nan
    delta_t_mean = np.mean(delta_ts) if delta_ts else np.nan
    pulse_width_mean = np.mean(pulse_widths) if pulse_widths else np.nan
    augmentation_index = np.mean(augmentation_indices) if augmentation_indices else np.nan
    stiffness_index = np.mean(stiffness_indices) if stiffness_indices else np.nan
    reflection_index = np.mean(reflection_indices) if reflection_indices else np.nan
    pulse_area = np.array(pulse_areas) if pulse_areas else np.array([])

    args = (augmentation_index, stiffness_index, reflection_index,
            crest_time_mean, delta_t_mean, pulse_width_mean, pulse_area)
    names = ('augmentation_index', 'stiffness_index', 'reflection_index',
             'crest_time_mean', 'delta_t_mean', 'pulse_width_mean',
             'pulse_area')

    return utils.ReturnTuple(args, names)


def ppg_signal_quality(signal=None, sampling_rate=1000., peaks=None):
    """Assess PPG signal quality using multiple criteria.

    Parameters
    ----------
    signal : array
        Filtered PPG signal.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    peaks : array, optional
        Indices of systolic peaks. If None, detected automatically.

    Returns
    -------
    sqi_perfusion : float
        Perfusion index: AC/DC ratio (%). Higher values indicate
        better perfusion and signal quality.
    sqi_skewness : float
        Skewness-based SQI. Clean PPG is right-skewed (positive skewness).
    sqi_kurtosis : float
        Kurtosis-based SQI. Clean PPG has specific kurtosis range.
    sqi_template : float
        Template-matching SQI. Average correlation of each pulse with
        the mean pulse template. Range: 0 (poor) to 1 (excellent).
    sqi_overall : float
        Overall SQI score (0-1) combining all metrics.

    References
    ----------
    Elgendi M. Optimal signal quality index for photoplethysmogram signals.
    Bioengineering. 2016;3(4):21.

    Li Q, Clifford GD. Dynamic time warping and machine learning for signal
    quality assessment of pulsatile signals. Physiol Meas.
    2012;33(9):1491-1501.
    """

    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal, dtype=float)
    sampling_rate = float(sampling_rate)

    # perfusion index: AC/DC ratio
    dc = np.mean(signal)
    ac = (np.max(signal) - np.min(signal)) / 2.0
    sqi_perfusion = (ac / abs(dc) * 100.0) if abs(dc) > 0 else 0.0

    # skewness
    from scipy.stats import skew, kurtosis as sp_kurtosis
    sqi_skewness = skew(signal)

    # kurtosis
    sqi_kurtosis = sp_kurtosis(signal)

    # template matching SQI
    sqi_template = 0.0
    if peaks is not None and len(peaks) >= 3:
        templates = []
        for i in range(len(peaks) - 1):
            start = peaks[i]
            end = peaks[i + 1]
            pulse = signal[start:end]
            if len(pulse) >= 5:
                target_len = 100
                x_old = np.linspace(0, 1, len(pulse))
                x_new = np.linspace(0, 1, target_len)
                pulse_resampled = np.interp(x_new, x_old, pulse)
                templates.append(pulse_resampled)

        if len(templates) >= 3:
            templates_arr = np.array(templates)
            mean_template = np.mean(templates_arr, axis=0)

            correlations = []
            for tmpl in templates_arr:
                corr = np.corrcoef(tmpl, mean_template)[0, 1]
                correlations.append(corr)

            sqi_template = np.mean(correlations)

    # overall SQI score
    scores = []
    scores.append(min(sqi_perfusion / 5.0, 1.0))
    scores.append(1.0 if sqi_skewness > 0 else max(0, 0.5 + sqi_skewness))
    if sqi_template > 0:
        scores.append(max(0, sqi_template))
    sqi_overall = np.mean(scores)

    args = (sqi_perfusion, sqi_skewness, sqi_kurtosis, sqi_template,
            sqi_overall)
    names = ('sqi_perfusion', 'sqi_skewness', 'sqi_kurtosis',
             'sqi_template', 'sqi_overall')

    return utils.ReturnTuple(args, names)
