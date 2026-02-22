# -*- coding: utf-8 -*-
"""
biosppy.signals.baroreflex
--------------------------

This module provides baroreflex sensitivity analysis functionality,
combining ECG and arterial blood pressure signals.
This module provides methods to analyze baroreflex sensitivity (BRS) using
the sequential method by Di Rienzo et al.

:copyright: (c) 2015-2025 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

"""

# Imports
# standard library
from collections import OrderedDict

# third-party
import numpy as np
from scipy import signal, stats, interpolate

# local
from . import tools
from .. import utils


def baroreflex_sensitivity(rri=None, sbp=None, rpeaks=None, systolic_peaks=None,
                           sampling_rate=1000.0, method='sequence',
                           min_sequences=3, show=False, **kwargs):
    """Compute baroreflex sensitivity (BRS) from RR intervals and systolic blood pressure.

    Baroreflex sensitivity represents the relationship between changes in
    arterial blood pressure and heart rate (RR intervals). It is an important
    marker of autonomic cardiovascular control.

    Parameters
    ----------
    rri : array, optional
        RR intervals in milliseconds.
    sbp : array, optional
        Systolic blood pressure values in mmHg.
    rpeaks : array, optional
        R-peak locations (samples). If provided, RRI will be computed.
    systolic_peaks : array, optional
        Systolic peak locations (samples). If provided, SBP will be extracted.
    sampling_rate : float, optional
        Sampling frequency (Hz); default is 1000.0 Hz.
    method : str, optional
        Method for BRS computation: 'sequence' (default), 'spectral', 'alpha', or 'all'.
        - 'sequence': Sequence method (Bertinieri et al., 1988)
        - 'spectral': Spectral (transfer function) method
        - 'alpha': Alpha coefficient method
        - 'all': Compute all methods
    min_sequences : int, optional
        Minimum number of sequences for sequence method; default is 3.
    show : bool, optional
        If True, show plots; default is False.
    **kwargs : dict, optional
        Additional keyword arguments:
        - sequence_length : int
            Minimum sequence length (default: 3)
        - sequence_threshold : float
            Threshold for BP and RRI changes (default: 1.0)
        - sequence_delay : int
            Maximum delay between BP and RRI changes in beats (default: 0-3)

    Returns
    -------
    brs : dict
        Dictionary containing baroreflex sensitivity results:
        - 'brs_sequence': Sequence method BRS (ms/mmHg)
        - 'brs_spectral_lf': Spectral method BRS in LF band (ms/mmHg)
        - 'brs_spectral_hf': Spectral method BRS in HF band (ms/mmHg)
        - 'brs_alpha_lf': Alpha coefficient in LF band (ms/mmHg)
        - 'brs_alpha_hf': Alpha coefficient in HF band (ms/mmHg)
        - 'n_sequences_up': Number of up sequences
        - 'n_sequences_down': Number of down sequences
        - 'coherence_lf': Coherence in LF band
        - 'coherence_hf': Coherence in HF band
        - 'rri': RR intervals used (ms)
        - 'sbp': SBP values used (mmHg)

    Notes
    -----
    * BRS values are typically in the range of 5-30 ms/mmHg in healthy adults.
    * Lower BRS values indicate reduced baroreflex function.
    * The sequence method identifies spontaneous sequences of increasing or
      decreasing BP and RRI.

    References
    ----------
    * Bertinieri et al. (1988). "Evaluation of baroreceptor reflex by blood
      pressure monitoring in unanesthetized cats." American Journal of Physiology.
    * Laude et al. (2004). "Comparison of various techniques used to estimate
      spontaneous baroreflex sensitivity." American Journal of Physiology.
    * Parati et al. (2000). "Point:Counterpoint: Cardiovascular variability is/is
      not an index of autonomic control of circulation." Journal of Applied Physiology.

    Examples
    --------
    >>> # Using pre-computed RRI and SBP
    >>> brs_results = baroreflex_sensitivity(rri=rri_array, sbp=sbp_array)
    >>> print(f"BRS (sequence method): {brs_results['brs_sequence']:.2f} ms/mmHg")

    >>> # Using R-peaks and systolic peaks
    >>> brs_results = baroreflex_sensitivity(
    ...     rpeaks=ecg_rpeaks,
    ...     systolic_peaks=abp_systolic_peaks,
    ...     sampling_rate=1000.0,
    ...     method='all'
    ... )
    """
    # Check input
    if rri is None and rpeaks is None:
        raise TypeError("Either 'rri' or 'rpeaks' must be provided.")

    if sbp is None and systolic_peaks is None:
        raise TypeError("Either 'sbp' or 'systolic_peaks' must be provided.")

    # Compute RRI from R-peaks if needed
    if rri is None:
        rri = np.diff(rpeaks) / sampling_rate * 1000.0  # Convert to ms

    # Ensure arrays
    rri = np.array(rri, dtype='float64')
    if sbp is not None:
        sbp = np.array(sbp, dtype='float64')

    # Initialize output
    out = OrderedDict()
    out['rri'] = rri
    out['sbp'] = sbp

    # Compute based on method
    if method == 'sequence' or method == 'all':
        seq_results = _sequence_method(
            rri, sbp, min_sequences=min_sequences, **kwargs
        )
        out.update(seq_results)

    if method == 'spectral' or method == 'all':
        spec_results = _spectral_method(rri, sbp, sampling_rate=sampling_rate)
        out.update(spec_results)

    if method == 'alpha' or method == 'all':
        alpha_results = _alpha_method(rri, sbp, sampling_rate=sampling_rate)
        out.update(alpha_results)

    if method not in ['sequence', 'spectral', 'alpha', 'all']:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Valid options: 'sequence', 'spectral', 'alpha', 'all'."
        )

    # Plotting
    if show:
        _plot_baroreflex(out, method=method)

    return utils.ReturnTuple(tuple(out.values()), tuple(out.keys()))


def _sequence_method(rri, sbp, sequence_length=3, sequence_threshold=1.0,
                     sequence_delay=1, min_sequences=3):
    """Compute BRS using sequence method.

    Parameters
    ----------
    rri : array
        RR intervals (ms).
    sbp : array
        Systolic blood pressure (mmHg).
    sequence_length : int
        Minimum sequence length.
    sequence_threshold : float
        Threshold for changes (mmHg and ms).
    sequence_delay : int
        Maximum delay between BP and RRI changes.
    min_sequences : int
        Minimum number of sequences required.

    Returns
    -------
    results : dict
        Sequence method results.
    """
    # Ensure same length
    min_len = min(len(rri), len(sbp))
    rri = rri[:min_len]
    sbp = sbp[:min_len]

    # Find sequences
    up_sequences = []
    down_sequences = []

    for delay in range(sequence_delay + 1):
        # Shift RRI by delay
        if delay > 0:
            rri_shifted = rri[delay:]
            sbp_shifted = sbp[:-delay]
        else:
            rri_shifted = rri
            sbp_shifted = sbp

        # Find increasing sequences
        up_seq = _find_sequences(
            sbp_shifted, rri_shifted,
            direction='up',
            min_length=sequence_length,
            threshold=sequence_threshold
        )
        up_sequences.extend(up_seq)

        # Find decreasing sequences
        down_seq = _find_sequences(
            sbp_shifted, rri_shifted,
            direction='down',
            min_length=sequence_length,
            threshold=sequence_threshold
        )
        down_sequences.extend(down_seq)

    # Compute BRS from sequences
    brs_values = []

    for seq in up_sequences + down_sequences:
        sbp_seq = seq['sbp']
        rri_seq = seq['rri']

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(sbp_seq, rri_seq)

        # Only include if correlation is significant
        if r_value**2 > 0.85:  # R² > 0.85
            brs_values.append(slope)

    # Compute mean BRS
    if len(brs_values) >= min_sequences:
        brs_sequence = np.mean(brs_values)
    else:
        brs_sequence = np.nan

    results = {
        'brs_sequence': brs_sequence,
        'n_sequences_up': len(up_sequences),
        'n_sequences_down': len(down_sequences),
        'brs_values': np.array(brs_values)
    }

    return results


def _find_sequences(sbp, rri, direction='up', min_length=3, threshold=1.0):
    """Find sequences of simultaneous changes in SBP and RRI.

    Parameters
    ----------
    sbp : array
        Systolic blood pressure values.
    rri : array
        RR intervals.
    direction : str
        'up' for increasing or 'down' for decreasing.
    min_length : int
        Minimum sequence length.
    threshold : float
        Minimum change threshold.

    Returns
    -------
    sequences : list
        List of sequences (dicts with 'sbp' and 'rri' arrays).

    References
    ----------
    .. [DiRienzo85] M. Di Rienzo, G. Parati, P. Castiglioni, R. Tordi, G. Mancia,
       and A. Pedotti, "Baroreflex effectiveness index: an additional measure of
       baroreflex control of heart rate in daily life," American Journal of
       Physiology-Regulatory, Integrative and Comparative Physiology, vol. 280,
       no. 3, pp. R744-R751, 2001.

    .. [Parati88] G. Parati, M. Di Rienzo, and G. Mancia, "How to measure
       baroreflex sensitivity: from the cardiovascular laboratory to daily life,"
       Journal of hypertension, vol. 18, no. 1, pp. 7-19, 2000.
    """
    sequences = []
    i = 0

    while i < len(sbp) - min_length + 1:
        # Check if we can start a sequence
        seq_sbp = [sbp[i]]
        seq_rri = [rri[i]]

        j = i + 1
        while j < len(sbp):
            sbp_change = sbp[j] - sbp[j-1]
            rri_change = rri[j] - rri[j-1]

            # Check if both change in the same direction
            if direction == 'up':
                if sbp_change >= threshold and rri_change >= threshold:
                    seq_sbp.append(sbp[j])
                    seq_rri.append(rri[j])
                    j += 1
                else:
                    break
            elif direction == 'down':
                if sbp_change <= -threshold and rri_change <= -threshold:
                    seq_sbp.append(sbp[j])
                    seq_rri.append(rri[j])
                    j += 1
                else:
                    break

        # Check if sequence is long enough
        if len(seq_sbp) >= min_length:
            sequences.append({
                'sbp': np.array(seq_sbp),
                'rri': np.array(seq_rri),
                'start': i,
                'end': j
            })

        i = j if j > i + 1 else i + 1

    return sequences


def sequential_method(sbp=None, rri=None, threshold_sbp=1.0, threshold_rri=5.0,
                     min_sequence_length=3, max_lag=0, correlation_threshold=0.85):
    """Compute baroreflex sensitivity (BRS) using the sequential method
    by Di Rienzo et al.

    This method identifies sequences of consecutive beats where both systolic
    blood pressure (SBP) and RR intervals increase or both decrease. For each
    sequence, it computes the slope of the linear relationship between SBP and
    RRI changes. The baroreflex sensitivity is estimated as the average of
    these slopes.

    Parameters
    ----------
    sbp : array
        Systolic blood pressure values (mmHg) for each cardiac cycle.
    rri : array
        RR intervals (ms) for each cardiac cycle.
    threshold_sbp : float, optional
        Minimum change in SBP to consider a significant variation (mmHg).
        Default: 1.0 mmHg.
    threshold_rri : float, optional
        Minimum change in RRI to consider a significant variation (ms).
        Default: 5.0 ms.
    min_sequence_length : int, optional
        Minimum number of consecutive beats to consider a valid sequence.
        Default: 3.
    max_lag : int, optional
        Maximum lag (in beats) to test between SBP changes and RRI responses.
        If max_lag > 0, the function will test lags from 0 to max_lag and
        return results for all lags. Default: 0 (no lag).
    correlation_threshold : float, optional
        Minimum correlation coefficient (r) for a sequence to be considered
        valid. Default: 0.85.

    Returns
    -------
    brs : float or array
        Baroreflex sensitivity (ms/mmHg). If max_lag = 0, returns a single
        value. If max_lag > 0, returns an array with BRS values for each lag
        from 0 to max_lag.
    n_sequences_up : int or array
        Number of sequences with increasing SBP and RRI (up-up sequences).
    n_sequences_down : int or array
        Number of sequences with decreasing SBP and RRI (down-down sequences).
    n_sequences_total : int or array
        Total number of valid sequences.
    sequences_info : list or array of lists
        Detailed information about each valid sequence for each lag.
        Each element contains: (start_index, end_index, slope, correlation,
        sequence_type) where sequence_type is 'up' or 'down'.

    Notes
    -----
    The sequential method assumes that:
    - SBP and RRI arrays have the same length and are aligned (beat-to-beat).
    - Both signals are preprocessed and artifacts are removed.
    - The relationship between SBP and RRI is approximately linear for small
      changes.

    The method is particularly useful for short-term recordings (5-10 minutes)
    and provides a non-invasive estimate of baroreflex sensitivity.

    Examples
    --------
    >>> # Single lag analysis (lag = 0)
    >>> brs, n_up, n_down, n_total, seq_info = sequential_method(
    ...     sbp=sbp_values,
    ...     rri=rri_values,
    ...     min_sequence_length=3
    ... )

    >>> # Multiple lag analysis (lag = 0, 1, 2)
    >>> results = sequential_method(
    ...     sbp=sbp_values,
    ...     rri=rri_values,
    ...     max_lag=2,
    ...     min_sequence_length=3
    ... )
    >>> brs_values = results['brs']  # Array with BRS for each lag
    >>> print(f"BRS at lag 0: {brs_values[0]:.2f} ms/mmHg")
    >>> print(f"BRS at lag 1: {brs_values[1]:.2f} ms/mmHg")
    """

    # Check inputs
    if sbp is None:
        raise TypeError("Please specify systolic blood pressure (SBP) values.")
    if rri is None:
        raise TypeError("Please specify RR interval (RRI) values.")

    # Ensure numpy arrays
    sbp = np.array(sbp, dtype=float)
    rri = np.array(rri, dtype=float)

    # Check array lengths
    if len(sbp) != len(rri):
        raise ValueError("SBP and RRI arrays must have the same length.")

    if len(sbp) < min_sequence_length + 1:
        raise ValueError(f"Input signals must have at least {min_sequence_length + 1} samples.")

    # If max_lag = 0, run single lag analysis
    if max_lag == 0:
        brs, n_up, n_down, n_total, seq_info = _sequential_single_lag(
            sbp, rri, 0, threshold_sbp, threshold_rri,
            min_sequence_length, correlation_threshold
        )

        # Output as ReturnTuple
        args = (brs, n_up, n_down, n_total, seq_info)
        names = ('brs', 'n_sequences_up', 'n_sequences_down',
                'n_sequences_total', 'sequences_info')

        return utils.ReturnTuple(args, names)

    # Multiple lag analysis
    else:
        brs_all = []
        n_up_all = []
        n_down_all = []
        n_total_all = []
        seq_info_all = []

        for lag in range(max_lag + 1):
            brs, n_up, n_down, n_total, seq_info = _sequential_single_lag(
                sbp, rri, lag, threshold_sbp, threshold_rri,
                min_sequence_length, correlation_threshold
            )

            brs_all.append(brs)
            n_up_all.append(n_up)
            n_down_all.append(n_down)
            n_total_all.append(n_total)
            seq_info_all.append(seq_info)

        # Convert to numpy arrays
        brs_all = np.array(brs_all)
        n_up_all = np.array(n_up_all)
        n_down_all = np.array(n_down_all)
        n_total_all = np.array(n_total_all)

        # Output as ReturnTuple
        args = (brs_all, n_up_all, n_down_all, n_total_all, seq_info_all)
        names = ('brs', 'n_sequences_up', 'n_sequences_down',
                'n_sequences_total', 'sequences_info')

        return utils.ReturnTuple(args, names)


def _sequential_single_lag(sbp, rri, lag, threshold_sbp, threshold_rri,
                           min_sequence_length, correlation_threshold):
    """Helper function to compute BRS for a single lag value.

    Parameters
    ----------
    sbp : array
        Systolic blood pressure values (mmHg).
    rri : array
        RR intervals (ms).
    lag : int
        Lag between SBP and RRI (in beats).
    threshold_sbp : float
        Minimum SBP change threshold (mmHg).
    threshold_rri : float
        Minimum RRI change threshold (ms).
    min_sequence_length : int
        Minimum sequence length.
    correlation_threshold : float
        Minimum correlation coefficient.

    Returns
    -------
    brs : float
        Baroreflex sensitivity (ms/mmHg).
    n_sequences_up : int
        Number of up-up sequences.
    n_sequences_down : int
        Number of down-down sequences.
    n_sequences_total : int
        Total number of sequences.
    sequences_info : list
        Information about each valid sequence.
    """

    # Adjust arrays for lag
    if lag > 0:
        sbp_lagged = sbp[:-lag]
        rri_lagged = rri[lag:]
    else:
        sbp_lagged = sbp
        rri_lagged = rri

    # Compute differences
    delta_sbp = np.diff(sbp_lagged)
    delta_rri = np.diff(rri_lagged)

    # Identify direction of changes (up: +1, down: -1, no change: 0)
    sbp_direction = np.sign(delta_sbp)
    rri_direction = np.sign(delta_rri)

    # Apply thresholds
    sbp_direction[np.abs(delta_sbp) < threshold_sbp] = 0
    rri_direction[np.abs(delta_rri) < threshold_rri] = 0

    # Find sequences where both SBP and RRI move in the same direction
    # +1: both increasing, -1: both decreasing, 0: not matching
    concordance = sbp_direction * rri_direction
    concordance[concordance < 0] = 0  # Remove discordant changes

    # Identify continuous sequences
    sequences_info = []
    slopes = []

    i = 0
    while i < len(concordance):
        if concordance[i] != 0:
            # Start of a potential sequence
            seq_type = 'up' if concordance[i] == 1 and sbp_direction[i] > 0 else 'down'
            start_idx = i

            # Extend sequence while concordance continues
            while i < len(concordance) and concordance[i] == concordance[start_idx]:
                i += 1

            end_idx = i
            seq_length = end_idx - start_idx + 1  # +1 because we need n+1 points for n intervals

            # Check if sequence is long enough
            if seq_length >= min_sequence_length:
                # Extract SBP and RRI values for this sequence
                # Need to add 1 to indices to get the actual values (not deltas)
                sbp_seq = sbp_lagged[start_idx:end_idx + 1]
                rri_seq = rri_lagged[start_idx:end_idx + 1]

                # Compute linear regression
                if len(sbp_seq) > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(sbp_seq, rri_seq)

                    # Check correlation threshold
                    if np.abs(r_value) >= correlation_threshold:
                        slopes.append(slope)
                        sequences_info.append({
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'length': seq_length,
                            'slope': slope,
                            'correlation': r_value,
                            'p_value': p_value,
                            'type': seq_type
                        })
        else:
            i += 1

    # Compute statistics
    if len(slopes) > 0:
        brs = np.mean(slopes)
        n_sequences_up = sum(1 for seq in sequences_info if seq['type'] == 'up')
        n_sequences_down = sum(1 for seq in sequences_info if seq['type'] == 'down')
        n_sequences_total = len(sequences_info)
    else:
        brs = np.nan
        n_sequences_up = 0
        n_sequences_down = 0
        n_sequences_total = 0

    return brs, n_sequences_up, n_sequences_down, n_sequences_total, sequences_info


def baroreflex_effectiveness_index(sbp=None, rri=None, threshold_sbp=1.0,
                                   threshold_rri=5.0):
    """Compute the Baroreflex Effectiveness Index (BEI).

    The BEI quantifies the percentage of SBP ramps that are followed by
    a baroreflex-mediated change in RRI. It provides a measure of how
    effectively the baroreflex responds to blood pressure changes.

    Parameters
    ----------
    sbp : array
        Systolic blood pressure values (mmHg) for each cardiac cycle.
    rri : array
        RR intervals (ms) for each cardiac cycle.
    threshold_sbp : float, optional
        Minimum change in SBP to consider a significant variation (mmHg).
        Default: 1.0 mmHg.
    threshold_rri : float, optional
        Minimum change in RRI to consider a significant variation (ms).
        Default: 5.0 ms.

    Returns
    -------
    bei : float
        Baroreflex Effectiveness Index (percentage, 0-100).
    n_sbp_ramps : int
        Total number of SBP ramps detected.
    n_effective_ramps : int
        Number of SBP ramps followed by an RRI response.

    Notes
    -----
    The BEI is calculated as:
    BEI = (number of SBP ramps followed by RRI changes / total SBP ramps) x 100

    A higher BEI indicates more effective baroreflex control. Normal values
    are typically in the range of 30-60%.

    Examples
    --------
    >>> bei, n_total, n_effective = baroreflex_effectiveness_index(
    ...     sbp=sbp_values,
    ...     rri=rri_values
    ... )
    >>> print(f"Baroreflex Effectiveness Index: {bei:.1f}%")
    """

    # Check inputs
    if sbp is None:
        raise TypeError("Please specify systolic blood pressure (SBP) values.")
    if rri is None:
        raise TypeError("Please specify RR interval (RRI) values.")

    # Ensure numpy arrays
    sbp = np.array(sbp, dtype=float)
    rri = np.array(rri, dtype=float)

    # Check array lengths
    if len(sbp) != len(rri):
        raise ValueError("SBP and RRI arrays must have the same length.")

    # Compute differences
    delta_sbp = np.diff(sbp)
    delta_rri = np.diff(rri)

    # Identify SBP ramps (significant changes)
    sbp_ramps = np.abs(delta_sbp) >= threshold_sbp

    # Identify RRI responses (significant changes)
    rri_responses = np.abs(delta_rri) >= threshold_rri

    # Check if SBP ramps are followed by RRI responses in the same direction
    effective_ramps = sbp_ramps & rri_responses & (np.sign(delta_sbp) == np.sign(delta_rri))

    # Compute BEI
    n_sbp_ramps = np.sum(sbp_ramps)
    n_effective_ramps = np.sum(effective_ramps)

    if n_sbp_ramps > 0:
        bei = (n_effective_ramps / n_sbp_ramps) * 100.0
    else:
        bei = np.nan

    # Output
    args = (bei, n_sbp_ramps, n_effective_ramps)
    names = ('bei', 'n_sbp_ramps', 'n_effective_ramps')

    return utils.ReturnTuple(args, names)


def _spectral_method(rri, sbp, sampling_rate=4.0):
    """Compute BRS using spectral (transfer function) method.

    Parameters
    ----------
    rri : array
        RR intervals (ms).
    sbp : array
        Systolic blood pressure (mmHg).
    sampling_rate : float
        Resampling rate for spectral analysis (Hz).

    Returns
    -------
    results : dict
        Spectral method results.
    """
    # Ensure same length
    min_len = min(len(rri), len(sbp))
    rri = rri[:min_len]
    sbp = sbp[:min_len]

    # Create uniform time vector
    time_orig = np.arange(len(rri))
    time_uniform = np.arange(0, len(rri), 1.0/sampling_rate)

    # Interpolate to uniform sampling
    f_rri = interpolate.interp1d(time_orig, rri, kind='cubic', fill_value='extrapolate')
    f_sbp = interpolate.interp1d(time_orig, sbp, kind='cubic', fill_value='extrapolate')

    rri_uniform = f_rri(time_uniform)
    sbp_uniform = f_sbp(time_uniform)

    # Compute cross-spectral density
    freqs, Pxy = signal.csd(sbp_uniform, rri_uniform, fs=sampling_rate, nperseg=256)
    freqs, Pxx = signal.welch(sbp_uniform, fs=sampling_rate, nperseg=256)
    freqs, Pyy = signal.welch(rri_uniform, fs=sampling_rate, nperseg=256)

    # Compute transfer function (gain)
    # H(f) = Pxy(f) / Pxx(f)
    H = Pxy / Pxx
    gain = np.abs(H)

    # Compute coherence
    coherence = np.abs(Pxy)**2 / (Pxx * Pyy)

    # Define frequency bands
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)

    # Extract BRS in LF band
    lf_mask = (freqs >= lf_band[0]) & (freqs <= lf_band[1])
    if np.any(lf_mask):
        # Use coherence-weighted average
        coherence_lf = np.mean(coherence[lf_mask])
        if coherence_lf > 0.5:  # Only if coherence is high enough
            brs_spectral_lf = np.mean(gain[lf_mask])
        else:
            brs_spectral_lf = np.nan
    else:
        brs_spectral_lf = np.nan
        coherence_lf = np.nan

    # Extract BRS in HF band
    hf_mask = (freqs >= hf_band[0]) & (freqs <= hf_band[1])
    if np.any(hf_mask):
        coherence_hf = np.mean(coherence[hf_mask])
        if coherence_hf > 0.5:
            brs_spectral_hf = np.mean(gain[hf_mask])
        else:
            brs_spectral_hf = np.nan
    else:
        brs_spectral_hf = np.nan
        coherence_hf = np.nan

    results = {
        'brs_spectral_lf': brs_spectral_lf,
        'brs_spectral_hf': brs_spectral_hf,
        'coherence_lf': coherence_lf,
        'coherence_hf': coherence_hf,
        'transfer_function_freqs': freqs,
        'transfer_function_gain': gain,
        'coherence': coherence
    }

    return results


def _alpha_method(rri, sbp, sampling_rate=4.0):
    """Compute BRS using alpha coefficient method.

    Parameters
    ----------
    rri : array
        RR intervals (ms).
    sbp : array
        Systolic blood pressure (mmHg).
    sampling_rate : float
        Resampling rate for spectral analysis (Hz).

    Returns
    -------
    results : dict
        Alpha method results.
    """
    # Ensure same length
    min_len = min(len(rri), len(sbp))
    rri = rri[:min_len]
    sbp = sbp[:min_len]

    # Create uniform time vector
    time_orig = np.arange(len(rri))
    time_uniform = np.arange(0, len(rri), 1.0/sampling_rate)

    # Interpolate to uniform sampling
    f_rri = interpolate.interp1d(time_orig, rri, kind='cubic', fill_value='extrapolate')
    f_sbp = interpolate.interp1d(time_orig, sbp, kind='cubic', fill_value='extrapolate')

    rri_uniform = f_rri(time_uniform)
    sbp_uniform = f_sbp(time_uniform)

    # Compute power spectral densities
    freqs, Pxx = signal.welch(sbp_uniform, fs=sampling_rate, nperseg=256)
    freqs, Pyy = signal.welch(rri_uniform, fs=sampling_rate, nperseg=256)

    # Define frequency bands
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)

    # Compute alpha coefficient in LF band
    lf_mask = (freqs >= lf_band[0]) & (freqs <= lf_band[1])
    if np.any(lf_mask):
        power_sbp_lf = np.sum(Pxx[lf_mask])
        power_rri_lf = np.sum(Pyy[lf_mask])
        if power_sbp_lf > 0:
            brs_alpha_lf = np.sqrt(power_rri_lf / power_sbp_lf)
        else:
            brs_alpha_lf = np.nan
    else:
        brs_alpha_lf = np.nan

    # Compute alpha coefficient in HF band
    hf_mask = (freqs >= hf_band[0]) & (freqs <= hf_band[1])
    if np.any(hf_mask):
        power_sbp_hf = np.sum(Pxx[hf_mask])
        power_rri_hf = np.sum(Pyy[hf_mask])
        if power_sbp_hf > 0:
            brs_alpha_hf = np.sqrt(power_rri_hf / power_sbp_hf)
        else:
            brs_alpha_hf = np.nan
    else:
        brs_alpha_hf = np.nan

    results = {
        'brs_alpha_lf': brs_alpha_lf,
        'brs_alpha_hf': brs_alpha_hf
    }

    return results


def _plot_baroreflex(brs_results, method='all'):
    """Plot baroreflex analysis results.

    Parameters
    ----------
    brs_results : dict or ReturnTuple
        Baroreflex sensitivity results.
    method : str
        Method used for computation.
    """
    import matplotlib.pyplot as plt

    # Convert ReturnTuple to dict if needed
    if isinstance(brs_results, utils.ReturnTuple):
        brs_dict = dict(zip(brs_results.keys(), brs_results.values()))
    else:
        brs_dict = brs_results

    # Create figure
    fig = plt.figure(figsize=(12, 8))

    # Plot 1: RRI vs SBP scatter
    ax1 = plt.subplot(2, 2, 1)
    rri = brs_dict.get('rri')
    sbp = brs_dict.get('sbp')

    if rri is not None and sbp is not None:
        ax1.scatter(sbp, rri, alpha=0.5)
        ax1.set_xlabel('SBP (mmHg)')
        ax1.set_ylabel('RRI (ms)')
        ax1.set_title('RRI vs SBP')
        ax1.grid(True)

    # Plot 2: BRS values
    ax2 = plt.subplot(2, 2, 2)
    brs_labels = []
    brs_values_plot = []

    if 'brs_sequence' in brs_dict and not np.isnan(brs_dict['brs_sequence']):
        brs_labels.append('Sequence')
        brs_values_plot.append(brs_dict['brs_sequence'])

    if 'brs_spectral_lf' in brs_dict and not np.isnan(brs_dict['brs_spectral_lf']):
        brs_labels.append('Spectral LF')
        brs_values_plot.append(brs_dict['brs_spectral_lf'])

    if 'brs_spectral_hf' in brs_dict and not np.isnan(brs_dict['brs_spectral_hf']):
        brs_labels.append('Spectral HF')
        brs_values_plot.append(brs_dict['brs_spectral_hf'])

    if 'brs_alpha_lf' in brs_dict and not np.isnan(brs_dict['brs_alpha_lf']):
        brs_labels.append('Alpha LF')
        brs_values_plot.append(brs_dict['brs_alpha_lf'])

    if 'brs_alpha_hf' in brs_dict and not np.isnan(brs_dict['brs_alpha_hf']):
        brs_labels.append('Alpha HF')
        brs_values_plot.append(brs_dict['brs_alpha_hf'])

    if brs_values_plot:
        ax2.bar(range(len(brs_values_plot)), brs_values_plot)
        ax2.set_xticks(range(len(brs_labels)))
        ax2.set_xticklabels(brs_labels, rotation=45)
        ax2.set_ylabel('BRS (ms/mmHg)')
        ax2.set_title('Baroreflex Sensitivity')
        ax2.grid(True)

    # Plot 3: Transfer function (if available)
    ax3 = plt.subplot(2, 2, 3)
    if 'transfer_function_freqs' in brs_dict and 'transfer_function_gain' in brs_dict:
        freqs = brs_dict['transfer_function_freqs']
        gain = brs_dict['transfer_function_gain']
        ax3.plot(freqs, gain)
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Gain (ms/mmHg)')
        ax3.set_title('Transfer Function')
        ax3.set_xlim([0, 0.5])
        ax3.grid(True)

    # Plot 4: Coherence (if available)
    ax4 = plt.subplot(2, 2, 4)
    if 'coherence' in brs_dict and 'transfer_function_freqs' in brs_dict:
        freqs = brs_dict['transfer_function_freqs']
        coherence = brs_dict['coherence']
        ax4.plot(freqs, coherence)
        ax4.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Coherence')
        ax4.set_title('Coherence')
        ax4.set_xlim([0, 0.5])
        ax4.set_ylim([0, 1])
        ax4.legend()
        ax4.grid(True)

    plt.tight_layout()
    plt.show()


def analyze_multichannel_baroreflex(mc_signal, ecg_channel='ECG',
                                    abp_channel='ABP', method='all', **kwargs):
    """Analyze baroreflex from a MultiChannelSignal object.

    Parameters
    ----------
    mc_signal : MultiChannelSignal
        Multi-channel signal object with processed ECG and ABP channels.
    ecg_channel : str, optional
        Name of ECG channel; default is 'ECG'.
    abp_channel : str, optional
        Name of ABP channel; default is 'ABP'.
    method : str, optional
        BRS computation method; default is 'all'.
    **kwargs : dict, optional
        Additional keyword arguments for baroreflex_sensitivity().

    Returns
    -------
    brs_results : ReturnTuple
        Baroreflex sensitivity results.

    Examples
    --------
    >>> from biosppy.signals import multichannel
    >>> # Create and process multi-channel signal
    >>> mc = multichannel.multichannel(
    ...     signals={'ECG': ecg_data, 'ABP': abp_data},
    ...     sampling_rate=1000.0,
    ...     channel_types={'ECG': 'ecg', 'ABP': 'abp'}
    ... )
    >>> # Analyze baroreflex
    >>> brs = analyze_multichannel_baroreflex(mc)
    >>> print(f"BRS: {brs['brs_sequence']:.2f} ms/mmHg")
    """
    # Get processed results
    ecg_results = mc_signal.get_processed(ecg_channel)
    abp_results = mc_signal.get_processed(abp_channel)

    # Extract R-peaks and systolic peaks
    rpeaks = ecg_results['rpeaks']

    # For ABP, use onsets to find systolic peaks
    # Systolic peaks are between consecutive onsets
    abp_signal = abp_results['filtered']
    onsets = abp_results['onsets']

    # Find systolic peaks (maximum between consecutive onsets)
    systolic_peaks = []
    systolic_values = []

    for i in range(len(onsets) - 1):
        start = onsets[i]
        end = onsets[i + 1]
        segment = abp_signal[start:end]

        if len(segment) > 0:
            peak_idx = np.argmax(segment) + start
            systolic_peaks.append(peak_idx)
            systolic_values.append(abp_signal[peak_idx])

    systolic_peaks = np.array(systolic_peaks)
    systolic_values = np.array(systolic_values)

    # Compute baroreflex sensitivity
    brs_results = baroreflex_sensitivity(
        rpeaks=rpeaks,
        systolic_peaks=systolic_peaks,
        sbp=systolic_values,
        sampling_rate=mc_signal.sampling_rate,
        method=method,
        **kwargs
    )

    return brs_results
