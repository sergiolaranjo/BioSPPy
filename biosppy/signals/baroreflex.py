# -*- coding: utf-8 -*-
"""
biosppy.signals.baroreflex
--------------------------

This module provides methods to analyze baroreflex sensitivity (BRS) using
the sequential method by Di Rienzo et al.

:copyright: (c) 2015-2025 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

References
----------
.. [DiRienzo85] M. Di Rienzo, G. Parati, P. Castiglioni, R. Tordi, G. Mancia,
   and A. Pedotti, "Baroreflex effectiveness index: an additional measure of
   baroreflex control of heart rate in daily life," American Journal of
   Physiology-Regulatory, Integrative and Comparative Physiology, vol. 280,
   no. 3, pp. R744–R751, 2001.

.. [Parati88] G. Parati, M. Di Rienzo, and G. Mancia, "How to measure
   baroreflex sensitivity: from the cardiovascular laboratory to daily life,"
   Journal of hypertension, vol. 18, no. 1, pp. 7–19, 2000.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function

# 3rd party
import numpy as np
from scipy import stats

# local
from .. import utils


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
    BEI = (number of SBP ramps followed by RRI changes / total SBP ramps) × 100

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
