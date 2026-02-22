# -*- coding: utf-8 -*-
"""
biosppy.signals.hrv
-------------------

This module provides computation and visualization of Heart-Rate Variability
metrics.


:copyright: (c) 2015-2023 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

"""

# Imports
# compat
from __future__ import absolute_import, division, print_function

# 3rd party
import numpy as np
import warnings
import pywt
from scipy.interpolate import interp1d
from scipy.signal import welch
import matplotlib.pyplot as plt

# local
from .. import utils
from .. import plotting
from . import tools as st

# Global variables
FBANDS = {'ulf': [0, 0.003],
          'vlf': [0.003, 0.04],
          'lf': [0.04, 0.15],
          'hf': [0.15, 0.4],
          'vhf': [0.4, 0.5]
          }

NOT_FEATURES = ['rri', 'rri_trend', 'outliers_method', 'rri_det', 'hr', 'bins',
                'q_hist', 'fbands', 'frequencies', 'powers', 'freq_method',
                'wavelet_name', 'decomposition_level', 'dwt_freq_bands']


def hrv(rpeaks=None, sampling_rate=1000., rri=None, parameters='auto',
        outliers='interpolate', detrend_rri=True, features_only=True,
        show=True, show_individual=False, **kwargs):
    """Extracts the RR-interval sequence from a list of R-peak indexes and
    extracts HRV features.

    Parameters
    ----------
    rpeaks : array
        R-peak index locations.
    sampling_rate : int, float, optional
        Sampling frequency (Hz). Default: 1000.0 Hz.
    rri : array, optional
        RR-intervals (ms). Providing this parameter overrides the computation of
        RR-intervals from rpeaks.
    parameters : str, optional
        If 'auto' computes the recommended HRV features. If 'time' computes
        only time-domain features. If 'frequency' computes only
        frequency-domain features. If 'non-linear' computes only non-linear
        features. If 'wavelet' computes only wavelet-based features.
        If 'all' computes all available HRV features. Default: 'auto'.
    outliers : str, optional
        Determines the method to handle outliers. If 'interpolate', replaces
        the outlier RR-intervals
        with cubic spline interpolation based on a local threshold. If 'filter',
        the RR-interval sequence
        is cut at the outliers. If None, no correction is performed. Default:
        'interpolate'.
    detrend_rri : bool, optional
        Whether to detrend the RRI sequence with the default method smoothness
        priors. Default: True.
    features_only : bool, optional
        Whether to return only the hrv features. Default: True.
    show : bool, optional
        Whether to show the HRV summary plot. Default: True.
    show_individual : bool, optional
        Whether to show the individual HRV plots. Default: False.
    kwargs : dict, optional
        fbands : dictionary of frequency bands (Hz) to use.
        wavelet : str, wavelet name for DWT analysis (default: 'db12').
        wavelet_level : int, decomposition level for DWT (default: 6).

    Returns
    -------
    rri : array
        RR-intervals (ms).
    rri_det : array
        Detrended RR-interval sequence (ms), if detrending was applied.
    hrv_features : dict
        The set of HRV features extracted from the RRI data. The number of
        features depends on the chosen parameters.
    """

    # check inputs
    if rpeaks is None and rri is None:
        raise TypeError("Please specify an R-Peak or RRI list or array.")

    parameters_list = ['auto', 'time', 'frequency', 'non-linear', 'wavelet', 'all']
    if parameters not in parameters_list:
        raise ValueError(f"'{parameters}' is not an available input. Enter one"
                         f"from: {parameters_list}.")

    # ensure input format
    sampling_rate = float(sampling_rate)

    # initialize outputs
    out = utils.ReturnTuple((), ())
    hrv_td, hrv_fd, hrv_nl = None, None, None

    # compute RRIs
    if rri is None:
        rpeaks = np.array(rpeaks, dtype=float)
        rri = compute_rri(rpeaks=rpeaks, sampling_rate=sampling_rate,
                          filter_rri=False)

    # compute duration
    duration = np.sum(rri) / 1000.  # seconds

    # handle outliers
    if outliers is None:
        pass
    elif outliers == 'interpolate':
        rri = rri_correction(rri)
    elif outliers == 'filter':
        rri = rri_filter(rri)

    # add rri to output
    out = out.append([rri, str(outliers)], ['rri', 'outliers_method'])

    # detrend rri sequence
    if detrend_rri:
        rri_det, rri_trend = detrend_window(rri)
        # add to output
        out = out.append([rri_det, rri_trend], ['rri_det', 'rri_trend'])
    else:
        rri_det = None
        rri_trend = None

    # plot
    if show_individual:
        plotting.plot_rri(rri, rri_trend, show=show_individual)

    # extract features
    if parameters == 'all':
        duration = np.inf

    # compute time-domain features
    if parameters in ['time', 'auto', 'all']:
        try:
            hrv_td = hrv_timedomain(rri=rri,
                                    duration=duration,
                                    detrend_rri=detrend_rri,
                                    show=show_individual,
                                    rri_detrended=rri_det)
            out = out.join(hrv_td)

        except ValueError as e:
            print('WARNING: Time-domain features not computed. Check input.')
            print(e)

            pass

    # compute frequency-domain features
    if parameters in ['frequency', 'auto', 'all']:
        try:
            hrv_fd = hrv_frequencydomain(rri=rri,
                                         duration=duration,
                                         detrend_rri=detrend_rri,
                                         show=show_individual,
                                         fbands=kwargs.get('fbands', None))
            out = out.join(hrv_fd)
        except ValueError as e:
            print('WARNING: Frequency-domain features not computed. Check input.')
            print(e)
            pass

    # compute non-linear features
    if parameters in ['non-linear', 'auto', 'all']:
        try:
            hrv_nl = hrv_nonlinear(rri=rri,
                                   duration=duration,
                                   detrend_rri=detrend_rri,
                                   show=show_individual)
            out = out.join(hrv_nl)

        except ValueError as e:
            print('WARNING: Non-linear features not computed. Check input.')
            print(e)
            pass

    # compute wavelet-based features
    if parameters in ['wavelet', 'all']:
        try:
            wavelet_name = kwargs.get('wavelet', 'db12')
            wavelet_level = kwargs.get('wavelet_level', 6)

            hrv_wv = hrv_wavelet(rri=rri,
                                 duration=duration,
                                 wavelet=wavelet_name,
                                 level=wavelet_level,
                                 detrend_rri=detrend_rri,
                                 show=show_individual)
            out = out.join(hrv_wv)

        except ValueError as e:
            print('WARNING: Wavelet features not computed. Check input.')
            print(e)
            pass

    # plot summary
    if show:
        if hrv_td is not None and hrv_fd is not None and hrv_nl is not None:
            plotting.plot_hrv(rri=rri,
                              rri_trend=rri_trend,
                              td_out=hrv_td,
                              nl_out=hrv_nl,
                              fd_out=hrv_fd,
                              show=True,
                              )
        else:
            warning = "Not all features were computed. To show the summary " \
                      "plot all features must be computed. Set " \
                      "'show_individual' to True to show the individual " \
                      "plots, or use parameters='all' to compute all features."
            warnings.warn(warning)

    # clean output if features_only
    if features_only:
        for key in NOT_FEATURES:
            try:
                out = out.delete(key)
            except ValueError:
                pass

    return out


def compute_rri(rpeaks, sampling_rate=1000., filter_rri=True, show=False):
    """Computes RR intervals in milliseconds from a list of R-peak indexes.

    Parameters
    ----------
    rpeaks : list, array
        R-peak index locations.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    filter_rri : bool, optional
        Whether to filter the RR-interval sequence. Default: True.
    show : bool, optional
        Plots the RR-interval sequence. Default: False.

    Returns
    -------
    rri : array
        RR-intervals (ms).
    """

    # ensure input format
    rpeaks = np.array(rpeaks)

    # difference of R-peaks converted to ms
    rri = (1000. * np.diff(rpeaks)) / sampling_rate

    # filter rri sequence
    if filter_rri:
        rri = rri_filter(rri)

    # check if rri is within physiological parameters
    if rri.min() < 400 or rri.min() > 1400:
        warnings.warn("RR-intervals appear to be out of normal parameters."
                      "Check input values.")

    if show:
        plotting.plot_rri(rri)

    return rri


def rri_filter(rri=None, threshold=1200):
    """Filters an RRI sequence based on a maximum threshold in milliseconds.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    threshold : int, float, optional
        Maximum rri value to accept (ms).

    Returns
    -------
    rri_filt : array
        Filtered RR-intervals (ms).
    """

    # ensure input format
    rri = np.array(rri, dtype=float)

    # filter rri values
    rri_filt = rri[np.where(rri < threshold)]

    return rri_filt


def rri_correction(rri=None, threshold=250):
    """Corrects artifacts in an RRI sequence based on a local average threshold.
    Artifacts are replaced with cubic spline interpolation.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    threshold : int, float, optional
        Local average threshold (ms). Default: 250.

    Returns
    -------
    rri : array
        Corrected RR-intervals (ms).
    """

    # check inputs
    if rri is None:
        raise ValueError("Please specify an RRI list or array.")

    # ensure input format
    rri = np.array(rri, dtype=float)

    # compute local average
    rri_filt, _ = st.smoother(signal=rri, kernel='median', size=3)

    # find artifacts
    artifacts = np.abs(rri - rri_filt) > threshold
    
    # before interpolating, check if the artifacts are at the beginning or end of the sequence
    if len(np.argwhere(artifacts)) >0:
        if min(np.where(artifacts)[0]) < min(np.where(~artifacts)[0]):
            rri = rri[min(np.where(~artifacts)[0]):]
            artifacts = artifacts[min(np.where(~artifacts)[0]):]
        
        if max(np.where(artifacts)[0]) > max(np.where(~artifacts)[0]):
            rri = rri[:max(np.where(~artifacts)[0])]
            artifacts = artifacts[:max(np.where(~artifacts)[0])]

    # replace artifacts with cubic spline interpolation
    rri[artifacts] = interp1d(np.where(~artifacts)[0], rri[~artifacts],
                              kind='cubic')(np.where(artifacts)[0])

    return rri


def hrv_timedomain(rri, duration=None, detrend_rri=True, show=False, **kwargs):
    """Computes the time domain HRV features from a sequence of RR intervals
    in milliseconds.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    duration : int, optional
        Duration of the signal (s).
    detrend_rri : bool, optional
        Whether to detrend the input signal.
    show : bool, optional
        Controls the plotting calls. Default: False.

    Returns
    -------
    hr : array
        Instantaneous heart rate (bpm).
    hr_min : float
        Minimum heart rate (bpm).
    hr_max : float
        Maximum heart rate (bpm).
    hr_minmax :  float
        Difference between the highest and the lowest heart rates (bpm).
    hr_mean : float
        Mean heart rate (bpm).
    hr_median : float
        Median heart rate (bpm).
    rr_min : float
        Minimum value of RR intervals (ms).
    rr_max : float
        Maximum value of RR intervals (ms).
    rr_minmax :  float
        Difference between the highest and the lowest values of RR intervals (ms).
    rr_mean : float
        Mean value of RR intervals (ms).
    rr_median : float
        Median value of RR intervals (ms).
    rmssd : float
        RMSSD - Root mean square of successive RR interval differences (ms).
    nn50 : int
        NN50 - Number of successive RR intervals that differ by more than 50ms.
    pnn50 : float
        pNN50 - Percentage of successive RR intervals that differ by more than
        50ms.
    sdnn: float
       SDNN - Standard deviation of RR intervals (ms).
    hti : float
        HTI - HRV triangular index - Integral of the density of the RR interval
        histogram divided by its height.
    tinn : float
        TINN - Baseline width of RR interval histogram (ms).
    """

    # check inputs
    if rri is None:
        raise ValueError("Please specify an RRI list or array.")

    # ensure numpy
    rri = np.array(rri, dtype=float)

    # detrend
    if detrend_rri:
        if 'rri_detrended' in kwargs:
            rri_det = kwargs['rri_detrended']
        else:
            rri_det = detrend_window(rri)['rri_det']
        print('Time domain: the rri sequence was detrended.')
    else:
        rri_det = rri

    if duration is None:
        duration = np.sum(rri) / 1000.  # seconds

    if duration < 10:
        raise ValueError("Signal duration must be greater than 10 seconds to "
                         "compute time-domain features.")

    # initialize outputs
    out = utils.ReturnTuple((), ())

    # compute the difference between RRIs
    rri_diff = np.diff(rri_det)

    if duration >= 10:
        # compute heart rate features
        hr = 60 / (rri / 1000)  # bpm
        hr_min = hr.min()
        hr_max = hr.max()
        hr_minmax = hr.max() - hr.min()
        hr_mean = hr.mean()
        hr_median = np.median(hr)

        out = out.append([hr, hr_min, hr_max, hr_minmax, hr_mean, hr_median],
                         ['hr', 'hr_min', 'hr_max', 'hr_minmax', 'hr_mean',
                          'hr_median'])

        # compute RRI features
        rr_min = rri.min()
        rr_max = rri.max()
        rr_minmax = rri.max() - rri.min()
        rr_mean = rri.mean()
        rr_median = np.median(rri)
        rmssd = (rri_diff ** 2).mean() ** 0.5

        out = out.append([rr_min, rr_max, rr_minmax, rr_mean, rr_median, rmssd],
                         ['rr_min', 'rr_max', 'rr_minmax', 'rr_mean',
                          'rr_median', 'rmssd'])

    if duration >= 20:
        # compute NN50 and pNN50
        th50 = 50
        nntot = len(rri_diff)
        nn50 = len(np.argwhere(abs(rri_diff) > th50))
        pnn50 = 100 * (nn50 / nntot)

        out = out.append([nn50, pnn50], ['nn50', 'pnn50'])

    if duration >= 60:
        # compute SDNN
        sdnn = rri_det.std()

        out = out.append(sdnn, 'sdnn')

    if duration >= 90:
        # compute geometrical features (histogram)
        out_geom = compute_geometrical(rri=rri, show=show)

        out = out.join(out_geom)

    return out


def hrv_frequencydomain(rri=None, duration=None, freq_method='FFT',
                        fbands=None, detrend_rri=True, show=False, **kwargs):
    """Computes the frequency domain HRV features from a sequence of RR
    intervals.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    duration : int, optional
        Duration of the signal (s).
    freq_method : str, optional
        Method for spectral estimation. If 'FFT' uses Welch's method.
    fbands : dict, optional
        Dictionary specifying the desired HRV frequency bands.
    detrend_rri : bool, optional
        Whether to detrend the input signal. Default: True.
    show : bool, optional
        Whether to show the power spectrum plot. Default: False.
    kwargs : dict, optional
        frs : resampling frequency for the RRI sequence (Hz).
        nperseg : Length of each segment in Welch periodogram.
        nfft : Length of the FFT used in Welch function.

    Returns
    -------
    {fbands}_peak : float
        Peak frequency for each frequency band (Hz).
    {fbands}_pwr : float
        Absolute power for each frequency band (ms^2).
    {fbands}_rpwr : float
        Relative power for each frequency band (nu).
    lf_hf : float
        Ratio of LF-to-HF power.
    lf_nu : float
        Ratio of LF to LF+HF power (nu).
    hf_nu :  float
        Ratio of HF to LF+HF power (nu).
    total_pwr : float
        Total power.
    """

    # check inputs
    if rri is None:
        raise TypeError("Please specify an RRI list or array.")

    freq_methods = ['FFT']
    if freq_method not in freq_methods:
        raise ValueError(f"'{freq_method}' is not an available input. Choose"
                         f"one from: {freq_methods}.")

    if fbands is None:
        fbands = FBANDS

    # ensure numpy
    rri = np.array(rri, dtype=float)

    # ensure minimal duration
    if duration is None:
        duration = np.sum(rri) / 1000.  # seconds

    if duration < 20:
        raise ValueError("Signal duration must be greater than 20 seconds to "
                         "compute frequency-domain features.")

    # initialize outputs
    out = utils.ReturnTuple((), ())
    out = out.append(fbands, 'fbands')

    # resampling with cubic interpolation for equidistant samples
    frs = kwargs['frs'] if 'frs' in kwargs else 4
    t = np.cumsum(rri)
    t -= t[0]
    rri_inter = interp1d(t, rri, 'cubic')
    t_inter = np.arange(t[0], t[-1], 1000. / frs)
    rri_inter = rri_inter(t_inter)

    # detrend
    if detrend_rri:
        rri_inter = detrend_window(rri_inter)['rri_det']
        print('Frequency domain: the rri sequence was detrended.')

    if duration >= 20:

        # compute frequencies and powers
        if freq_method == 'FFT':
            nperseg = kwargs['nperseg'] if 'nperseg' in kwargs else int(len(rri_inter)/4.5)
            nfft = kwargs['nfft'] if 'nfft' in kwargs else (256 if nperseg < 256 else 2**np.ceil(np.log(nperseg)/np.log(2)))

            frequencies, powers = welch(rri_inter, fs=frs, scaling='density',
                                        nperseg=nperseg, nfft=nfft)

            # add to output
            out = out.append([frequencies, powers, freq_method],
                             ['frequencies', 'powers', 'freq_method'])

        # compute frequency bands
        fb_out = compute_fbands(frequencies=frequencies, powers=powers, show=False)

        out = out.join(fb_out)

        # compute LF/HF ratio
        lf_hf = fb_out['lf_pwr'] / fb_out['hf_pwr']

        out = out.append(lf_hf, 'lf_hf')

        # compute LF and HF power in normal units
        lf_nu = fb_out['lf_pwr'] / (fb_out['lf_pwr'] + fb_out['hf_pwr'])
        hf_nu = 1 - lf_nu

        out = out.append([lf_nu, hf_nu], ['lf_nu', 'hf_nu'])

        # plot
        if show:
            legends = {'LF/HF': lf_hf}
            for key in out.keys():
                if key.endswith('_rpwr'):
                    legends[key] = out[key]

            plotting.plot_hrv_fbands(frequencies, powers, fbands, freq_method,
                                     legends, show=show)

    return out


def hrv_nonlinear(rri=None, duration=None, detrend_rri=True, show=False):
    """Computes the non-linear HRV features from a sequence of RR intervals.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    duration : int, optional
        Duration of the signal (s).
    detrend_rri : bool, optional
        Whether to detrend the input signal. Default: True.
    show : bool, optional
        Controls the plotting calls. Default: False.

    Returns
    -------
    s : float
        S - Area of the ellipse of the Poincaré plot (ms^2).
    sd1 : float
        SD1 - Poincaré plot standard deviation perpendicular to the identity
        line (ms).
    sd2 : float
        SD2 - Poincaré plot standard deviation along the identity line (ms).
    sd12 : float
        SD1/SD2 - SD1 to SD2 ratio.
    sd21 : float
        SD2/SD1 - SD2 to SD1 ratio.
    sampen : float
        Sample entropy.
    appen : float
        Approximate entropy.
    """

    # check inputs
    if rri is None:
        raise TypeError("Please specify an RRI list or array.")

    # ensure numpy
    rri = np.array(rri, dtype=float)

    # check duration
    if duration is None:
        duration = np.sum(rri) / 1000.  # seconds

    if duration < 90:
        raise ValueError("Signal duration must be greater than 90 seconds to "
                         "compute non-linear features.")

    # detrend
    if detrend_rri:
        rri = detrend_window(rri)['rri_det']
        print('Non-linear domain: the rri sequence was detrended.')

    # initialize outputs
    out = utils.ReturnTuple((), ())

    if duration >= 90:
        # compute SD1, SD2, SD1/SD2 and S
        cp = compute_poincare(rri=rri, show=show)
        out = out.join(cp)

        # compute sample entropy
        sampen = sample_entropy(rri)
        out = out.append(sampen, 'sampen')

    if len(rri) >= 800 or duration == np.inf:
        # compute approximate entropy
        appen = approximate_entropy(rri)
        out = out.append(appen, 'appen')

    return out


def hrv_wavelet(rri=None, duration=None, wavelet='db12', level=6,
                detrend_rri=True, show=False):
    """Computes HRV features using Discrete Wavelet Transform (DWT) decomposition.

    This function performs multi-resolution wavelet decomposition of the RRI
    sequence using Daubechies 12 wavelet (or other specified wavelet) and
    computes energy-based features for each decomposition level.

    The wavelet decomposition provides a time-frequency representation where
    different levels correspond approximately to different HRV frequency bands:
    - Level 1 (highest freq): VHF-like components
    - Level 2-3: HF-like components
    - Level 4-5: LF-like components
    - Level 6+: VLF-like components

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    duration : int, optional
        Duration of the signal (s).
    wavelet : str, optional
        Wavelet family to use. Default: 'db12' (Daubechies 12).
        Other options: 'db4', 'db6', 'sym5', 'coif3', etc.
    level : int, optional
        Number of decomposition levels. Default: 6.
    detrend_rri : bool, optional
        Whether to detrend the input signal. Default: True.
    show : bool, optional
        Controls the plotting calls. Default: False.

    Returns
    -------
    wavelet_name : str
        Name of the wavelet used.
    decomposition_level : int
        Number of decomposition levels.
    dwt_energy_total : float
        Total energy of the wavelet decomposition.
    dwt_energy_rel_a{level} : float
        Relative energy of approximation coefficients at highest level.
    dwt_energy_rel_d{i} : float
        Relative energy of detail coefficients at level i (i=1 to level).
    dwt_energy_abs_a{level} : float
        Absolute energy of approximation coefficients.
    dwt_energy_abs_d{i} : float
        Absolute energy of detail coefficients at level i.
    dwt_entropy : float
        Shannon wavelet entropy based on relative energies.
    dwt_std_a{level} : float
        Standard deviation of approximation coefficients.
    dwt_std_d{i} : float
        Standard deviation of detail coefficients at level i.

    Notes
    -----
    The Daubechies 12 (db12) wavelet provides good time-frequency localization
    and is commonly used in HRV analysis. The decomposition level should be
    chosen based on the sampling rate and desired frequency resolution.

    For an RRI sequence with mean RR~1000ms (HR~60bpm), typical sampling is
    ~1Hz, and 6 levels of decomposition provide adequate frequency bands.

    References
    ----------
    .. [1] Thurner, S., Feurstein, M. C., & Teich, M. C. (1998).
           Multiresolution wavelet analysis of heartbeat intervals discriminates
           healthy patients from those with cardiac pathology.
           Physical Review Letters, 80(7), 1544.
    .. [2] Acharya, U. R., et al. (2015). Heart rate variability:
           a review. Medical & biological engineering & computing, 44(12), 1031-1051.

    """

    # check inputs
    if rri is None:
        raise TypeError("Please specify an RRI list or array.")

    # ensure numpy
    rri = np.array(rri, dtype=float)

    # check duration
    if duration is None:
        duration = np.sum(rri) / 1000.  # seconds

    if duration < 60:
        raise ValueError("Signal duration must be greater than 60 seconds to "
                         "compute wavelet-based features.")

    # check if wavelet is valid
    if wavelet not in pywt.wavelist():
        raise ValueError(f"'{wavelet}' is not a valid wavelet. "
                         f"Available wavelets: {pywt.wavelist()}")

    # check decomposition level
    max_level = pywt.dwt_max_level(len(rri), wavelet)
    if level > max_level:
        warnings.warn(f"Decomposition level {level} is too high for signal "
                      f"length {len(rri)}. Using maximum level {max_level}.")
        level = max_level

    # detrend
    if detrend_rri:
        rri = detrend_window(rri)['rri_det']
        print('Wavelet domain: the rri sequence was detrended.')

    # initialize outputs
    out = utils.ReturnTuple((), ())

    # add wavelet info to output
    out = out.append([wavelet, level], ['wavelet_name', 'decomposition_level'])

    # perform wavelet decomposition
    coeffs = pywt.wavedec(rri, wavelet, level=level)

    # coeffs[0] = approximation coefficients (lowest frequency)
    # coeffs[1:] = detail coefficients from level n to 1 (high to low freq)

    # compute energy for each coefficient set
    energies = []
    for coeff in coeffs:
        energy = np.sum(coeff ** 2)
        energies.append(energy)

    total_energy = np.sum(energies)

    # add total energy to output
    out = out.append(total_energy, 'dwt_energy_total')

    # compute relative energies (normalized by total energy)
    rel_energies = [e / total_energy for e in energies]

    # add approximation energy (coeffs[0])
    out = out.append([energies[0], rel_energies[0]],
                     [f'dwt_energy_abs_a{level}', f'dwt_energy_rel_a{level}'])

    # add approximation std
    out = out.append(np.std(coeffs[0]), f'dwt_std_a{level}')

    # add detail energies (coeffs[1] is level n, coeffs[2] is level n-1, etc.)
    for i in range(1, len(coeffs)):
        detail_level = level - i + 1
        out = out.append([energies[i], rel_energies[i]],
                         [f'dwt_energy_abs_d{detail_level}',
                          f'dwt_energy_rel_d{detail_level}'])

        # add detail std
        out = out.append(np.std(coeffs[i]), f'dwt_std_d{detail_level}')

    # compute wavelet entropy (Shannon entropy based on energy distribution)
    # Filter out zero energies to avoid log(0)
    rel_energies_nonzero = [e for e in rel_energies if e > 0]
    if len(rel_energies_nonzero) > 0:
        wavelet_entropy = -np.sum([e * np.log(e) for e in rel_energies_nonzero])
    else:
        wavelet_entropy = 0.0

    out = out.append(wavelet_entropy, 'dwt_entropy')

    # compute frequency band approximations based on decomposition levels
    # This is an approximation based on dyadic decomposition
    # Assuming mean RR interval is around 1000ms, effective sampling ~1Hz
    mean_rr = np.mean(rri) / 1000.0  # convert to seconds
    fs_effective = 1.0 / mean_rr  # effective sampling frequency

    # map detail levels to frequency bands
    freq_bands = {}
    for i in range(1, level + 1):
        f_low = fs_effective / (2 ** (i + 1))
        f_high = fs_effective / (2 ** i)
        freq_bands[f'dwt_fband_d{i}'] = [f_low, f_high]

    # approximation band
    freq_bands[f'dwt_fband_a{level}'] = [0, fs_effective / (2 ** (level + 1))]

    # add frequency bands to output (for reference)
    out = out.append(freq_bands, 'dwt_freq_bands')

    # plot (if requested)
    if show:
        import matplotlib.pyplot as plt

        # create subplots for coefficients
        n_plots = len(coeffs)
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2*n_plots))

        if n_plots == 1:
            axes = [axes]

        # plot approximation
        axes[0].plot(coeffs[0])
        axes[0].set_title(f'Approximation (a{level}) - Energy: {energies[0]:.2f}')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True)

        # plot details
        for i in range(1, len(coeffs)):
            detail_level = level - i + 1
            axes[i].plot(coeffs[i])
            axes[i].set_title(f'Detail (d{detail_level}) - Energy: {energies[i]:.2f}')
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True)

        axes[-1].set_xlabel('Sample')
        plt.tight_layout()
        plt.show()

        # plot energy distribution
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        labels = [f'a{level}'] + [f'd{level-i+1}' for i in range(1, len(coeffs))]
        ax2.bar(labels, rel_energies)
        ax2.set_title('Relative Energy Distribution Across Wavelet Coefficients')
        ax2.set_xlabel('Coefficient Level')
        ax2.set_ylabel('Relative Energy')
        ax2.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

    return out


def compute_fbands(frequencies, powers, fbands=None, method_name=None,
                   show=False):
    """Computes frequency domain features for the specified frequency bands.

    Parameters
    ----------
    frequencies : array
        Frequency axis.
    powers : array
        Power spectrum values for the frequency axis.
    fbands : dict, optional
        Dictionary containing the limits of the frequency bands.
    method_name : str, optional
        Method that was used to compute the power spectrum. Default: None.
    show : bool, optional
        Whether to show the power spectrum plot. Default: False.

    Returns
    -------
    {fbands}_peak : float
        Peak frequency of the frequency band (Hz).
    {fbands}_pwr : float
        Absolute power of the frequency band (ms^2).
    {fbands}_rpwr : float
        Relative power of the frequency band (nu).
    """

    # initialize outputs
    out = utils.ReturnTuple((), ())

    df = frequencies[1] - frequencies[0]  # frequency resolution
    total_pwr = np.sum(powers) * df

    if fbands is None:
        fbands = FBANDS

    # compute power, peak and relative power for each frequency band
    for fband in fbands.keys():
        band = np.argwhere((frequencies >= fbands[fband][0]) & (frequencies <= fbands[fband][-1])).reshape(-1)

        # check if it's possible to compute the frequency band
        if len(band) == 0:
            continue

        pwr = np.sum(powers[band]) * df
        peak = frequencies[band][np.argmax(powers[band])]
        rpwr = pwr / total_pwr

        out = out.append([pwr, peak, rpwr], [fband + '_pwr', fband + '_peak',
                                             fband + '_rpwr'])

    # plot
    if show:
        # legends
        freq_legends = {}
        for key in out.keys():
            if key.endswith('_rpwr'):
                freq_legends[key] = out[key]

        plotting.plot_hrv_fbands(frequencies=frequencies,
                                 powers=powers,
                                 fbands=fbands,
                                 method_name=method_name,
                                 legends=freq_legends,
                                 show=show)

    return out


def compute_poincare(rri, show=False):
    """Compute the Poincaré features from a sequence of RR intervals.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    show : bool, optional
        If True, show the Poincaré plot.

    Returns
    -------
    s : float
        S - Area of the ellipse of the Poincaré plot (ms^2).
    sd1 : float
        SD1 - Poincaré plot standard deviation perpendicular to the identity
        line (ms).
    sd2 : float
        SD2 - Poincaré plot standard deviation along the identity line (ms).
    sd12 : float
        SD1/SD2 - SD1 to SD2 ratio.
    sd21 : float
        SD2/SD1 - SD2 to SD1 ratio.
    """

    # initialize outputs
    out = utils.ReturnTuple((), ())

    x = rri[:-1]
    y = rri[1:]

    # compute SD1, SD2 and S
    x1 = (x - y) / np.sqrt(2)
    x2 = (x + y) / np.sqrt(2)
    sd1 = x1.std()
    sd2 = x2.std()
    s = np.pi * sd1 * sd2

    # compute sd1/sd2 and sd2/sd1 ratio
    sd12 = sd1 / sd2
    sd21 = sd2 / sd1

    # output
    out = out.append([s, sd1, sd2, sd12, sd21], ['s', 'sd1', 'sd2', 'sd12',
                                                 'sd21'])

    if show:
        legends = {'SD1/SD2': sd12, 'SD2/SD1': sd21}
        plotting.plot_poincare(rri=rri,
                               s=s,
                               sd1=sd1,
                               sd2=sd2,
                               legends=legends,
                               show=show)

    return out


def compute_geometrical(rri, binsize=1/128, show=False):
    """Computes the geometrical features from a sequence of RR intervals.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    binsize : float, optional
        Binsize for RRI histogram (s). Default: 1/128 s.
    show : bool, optional
        If True, show the RRI histogram. Default: False.

    Returns
    -------
    hti : float
        HTI - HRV triangular index - Integral of the density of the RR interval
        histogram divided by its height.
    tinn : float
        TINN - Baseline width of RR interval histogram (ms).

    """
    binsize = binsize * 1000  # to ms

    # create histogram
    tmin = rri.min()
    tmax = rri.max()
    bins = np.arange(tmin, tmax + binsize, binsize)
    nn_hist = np.histogram(rri, bins)

    # histogram peak
    max_count = np.max(nn_hist[0])
    peak_hist = np.argmax(nn_hist[0])

    # compute HTI
    hti = len(rri) / max_count

    # possible N and M values
    n_values = bins[:peak_hist]
    m_values = bins[peak_hist + 1:]

    # find triangle with base N and M that best approximates the distribution
    error_min = np.inf
    n = 0
    m = 0
    q_hist = None

    for n_ in n_values:

        for m_ in m_values:

            t = np.array([tmin, n_, nn_hist[1][peak_hist], m_, tmax + binsize])
            y = np.array([0, 0, max_count, 0, 0])
            q = interp1d(x=t, y=y, kind='linear')
            q = q(bins)

            # compute the sum of squared differences
            error = np.sum((nn_hist[0] - q[:-1]) ** 2)

            if error < error_min:
                error_min = error
                n, m, q_hist = n_, m_, q

    # compute TINN
    tinn = m - n

    # plot
    if show:
        plotting.plot_hrv_hist(rri=rri,
                               bins=bins,
                               q_hist=q_hist,
                               hti=hti,
                               tinn=tinn,
                               show=show)

    # output
    out = utils.ReturnTuple([hti, tinn, bins, q_hist], ['hti', 'tinn',
                                                        'bins', 'q_hist'])

    return out


def detrend_window(rri, win_len=2000, **kwargs):
    """Facilitates RRI detrending method using a signal window.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    win_len : int, optional
        Length of the window to detrend the RRI signal. Default: 2000.
    kwargs : dict, optional
        Parameters of the detrending method.

    Returns
    -------
    rri_det : array
        Detrended RRI signal.
    rri_trend : array
        Trend of the RRI signal.

    """

    # check input type
    win_len = int(win_len)

    # extract parameters
    smoothing_factor = kwargs['smoothing_factor'] if 'smoothing_factor' in kwargs else 500

    # detrend signal
    if len(rri) > win_len:
        # split the signal
        splits = int(len(rri)/win_len)
        rri_splits = np.array_split(rri, splits)

        # compute the detrended signal for each split
        rri_det = []
        for split in rri_splits:
            split_det = st.detrend_smoothness_priors(split, smoothing_factor)['detrended']
            rri_det.append(split_det)

        # concantenate detrended splits
        rri_det = np.concatenate(rri_det)
        rri_trend = None
    else:
        rri_det, rri_trend = st.detrend_smoothness_priors(rri, smoothing_factor)

    # output
    out = utils.ReturnTuple([rri_det, rri_trend], ['rri_det', 'rri_trend'])
    return out


def sample_entropy(rri, m=2, r=0.2):
    """Computes the sample entropy of an RRI sequence.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    m : int, optional
        Embedding dimension. Default: 2.
    r : int, float, optional
        Tolerance. It is then multiplied by the sequence standard deviation.
        Default: 0.2.

    Returns
    -------
    sampen :  float
        Sample entropy of the RRI sequence.

    References
    ----------
    https://en.wikipedia.org/wiki/Sample_entropy
    """

    # redefine r
    r = r * rri.std()

    n = len(rri)

    # Split time series and save all templates of length m
    xmi = np.array([rri[i: i + m] for i in range(n - m)])
    xmj = np.array([rri[i: i + m] for i in range(n - m + 1)])

    # Save all matches minus the self-match, compute B
    b = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([rri[i: i + m] for i in range(n - m + 1)])

    a = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])
    if a > 0 and b > 0:
        sampen = -np.log(a / b)
    else:
        if a == 0 and b==0:
            # both a and b are zero => cannot determine saen
            sampen = np.nan
        elif a == 0:
            # a is zero => log would be infinite or undefined => cannot determine saen
            sampen = -np.inf
        else:
            # b is zero => a is not zero, but b is zero =>
            sampen = np.inf

    # Return SampEn
    return sampen


def approximate_entropy(rri, m=2, r=0.2):
    """Computes the approximate entropy of an RRI sequence.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    m : int, optional
        Embedding dimension. Default: 2.
    r : int, float, optional
        Tolerance. It is then multiplied by the sequence standard deviation.
        Default: 0.2.

    Returns
    -------
    appen :  float
        Approximate entropy of the RRI sequence.

    References
    ----------
    https://en.wikipedia.org/wiki/Approximate_entropy
    """

    # redefine r
    r = r * rri.std()

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[rri[j] for j in range(i, i + m - 1 + 1)] for i in range(n - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (n - m + 1.0)
            for x_i in x
        ]
        return (n - m + 1.0) ** (-1) * sum(np.log(C))

    n = len(rri)

    return _phi(m) - _phi(m + 1)


def heart_rate_turbulence(rri=None, vpc_indices=None, coupling_interval_min=300,
                          coupling_interval_max=2000, prematurity_threshold=0.8,
                          compensatory_pause_threshold=1.2, show=False):
    """Computes Heart Rate Turbulence (HRT) parameters from RR intervals.

    Heart Rate Turbulence is a method to assess the autonomic cardiac response
    following ventricular premature complexes (VPCs). The main parameters are:
    - Turbulence Onset (TO): Initial acceleration of heart rate after VPC
    - Turbulence Slope (TS): Subsequent deceleration of heart rate
def hht_variability(rri=None, method='ceemdan', num_ensemble=100, noise_std=0.2,
                   max_imf=None, max_iter=1000, sampling_rate=4.0, random_seed=None):
    """Compute HRV features using Hilbert-Huang Transform (HHT).

    Decomposes RR-intervals using CEEMDAN and extracts variability features
    from each Intrinsic Mode Function (IMF).

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    vpc_indices : array, optional
        Indices of VPCs in the RRI sequence. If None, VPCs will be detected
        automatically based on prematurity and compensatory pause criteria.
    coupling_interval_min : float, optional
        Minimum coupling interval (ms) to consider a VPC. Default: 300 ms.
    coupling_interval_max : float, optional
        Maximum coupling interval (ms) to consider a VPC. Default: 2000 ms.
    prematurity_threshold : float, optional
        Threshold for prematurity criterion (ratio to local average).
        Default: 0.8 (VPC is 20% shorter than local average).
    compensatory_pause_threshold : float, optional
        Threshold for compensatory pause criterion (ratio to local average).
        Default: 1.2 (compensatory pause is 20% longer than local average).
    show : bool, optional
        If True, show HRT plot. Default: False.

    Returns
    -------
    to : float
        Turbulence Onset (%) - Measures the initial heart rate acceleration
        after VPC. Normal values: TO < 0%.
    ts : float
        Turbulence Slope (ms/RR) - Measures the subsequent heart rate
        deceleration. Normal values: TS > 2.5 ms/RR.
    vpc_count : int
        Number of VPCs detected and used in the analysis.
    vpc_indices : array
        Indices of VPCs used in the analysis.

    References
    ----------
    Schmidt G, et al. Heart-rate turbulence after ventricular premature beats
    as a predictor of mortality after acute myocardial infarction.
    Lancet. 1999;353(9162):1390-6.

    Notes
    -----
    Standard HRT analysis requires:
    - At least 5 VPCs for reliable analysis
    - VPCs should be isolated (not part of couplets or runs)
    - Surrounding RR intervals should be within physiological range
    """

    # check inputs
    if rri is None:
        raise TypeError("Please specify an RRI list or array.")

    # ensure numpy
    rri = np.array(rri, dtype=float)

    # initialize outputs
    out = utils.ReturnTuple((), ())

    # detect VPCs if not provided
    if vpc_indices is None:
        vpc_indices = _detect_vpcs(rri,
                                    coupling_interval_min=coupling_interval_min,
                                    coupling_interval_max=coupling_interval_max,
                                    prematurity_threshold=prematurity_threshold,
                                    compensatory_pause_threshold=compensatory_pause_threshold)
    else:
        vpc_indices = np.array(vpc_indices, dtype=int)

    vpc_count = len(vpc_indices)

    # check if enough VPCs are available
    if vpc_count < 1:
        warnings.warn("No VPCs detected. Cannot compute HRT parameters.")
        return out.append([np.nan, np.nan, 0, np.array([])],
                         ['to', 'ts', 'vpc_count', 'vpc_indices'])

    if vpc_count < 5:
        warnings.warn(f"Only {vpc_count} VPC(s) detected. Standard HRT analysis "
                     "requires at least 5 VPCs for reliable results.")

    # compute turbulence onset (TO)
    to_values = []
    valid_vpcs = []

    for vpc_idx in vpc_indices:
        # need at least 2 RR intervals before and 2 after the VPC
        if vpc_idx < 2 or vpc_idx + 2 >= len(rri):
            continue

        # RR intervals around VPC
        rr_pre2 = rri[vpc_idx - 2]  # 2 intervals before
        rr_pre1 = rri[vpc_idx - 1]  # 1 interval before
        rr_vpc = rri[vpc_idx]        # coupling interval (to VPC)
        rr_post1 = rri[vpc_idx + 1]  # 1 interval after (compensatory pause)
        rr_post2 = rri[vpc_idx + 2]  # 2 intervals after

        # compute TO for this VPC
        # TO = [(RR1 + RR2) - (RR-2 + RR-1)] / (RR-2 + RR-1) × 100%
        denominator = rr_pre2 + rr_pre1
        if denominator > 0:
            to = ((rr_post1 + rr_post2) - denominator) / denominator * 100
            to_values.append(to)
            valid_vpcs.append(vpc_idx)

    # compute mean TO
    if len(to_values) > 0:
        to = np.mean(to_values)
    else:
        to = np.nan
        warnings.warn("Could not compute TO. No valid VPCs with sufficient "
                     "surrounding intervals.")

    # compute turbulence slope (TS)
    ts_values = []

    for vpc_idx in valid_vpcs:
        # need at least 15 RR intervals after VPC for TS calculation
        if vpc_idx + 15 >= len(rri):
            continue

        # get 15 RR intervals after the compensatory pause
        rr_sequence = rri[vpc_idx + 2:vpc_idx + 17]

        # compute maximum slope of any 5 consecutive RR intervals
        max_slope = -np.inf

        for i in range(len(rr_sequence) - 4):
            # 5 consecutive intervals
            rr_window = rr_sequence[i:i + 5]
            x = np.arange(5)

            # linear regression
            slope = np.polyfit(x, rr_window, 1)[0]

            if slope > max_slope:
                max_slope = slope

        if max_slope > -np.inf:
            ts_values.append(max_slope)

    # compute mean TS
    if len(ts_values) > 0:
        ts = np.mean(ts_values)
    else:
        ts = np.nan
        warnings.warn("Could not compute TS. No valid VPCs with sufficient "
                     "post-VPC intervals.")

    # prepare output
    valid_vpcs = np.array(valid_vpcs)
    out = out.append([to, ts, len(valid_vpcs), valid_vpcs],
                    ['to', 'ts', 'vpc_count', 'vpc_indices'])

    # plot
    if show:
        _plot_hrt(rri, valid_vpcs, to, ts)

    return out


def _detect_vpcs(rri, coupling_interval_min=300, coupling_interval_max=2000,
                 prematurity_threshold=0.8, compensatory_pause_threshold=1.2):
    """Detects ventricular premature complexes (VPCs) in RRI sequence.
    method : str, optional
        Decomposition method: 'emd', 'eemd', or 'ceemdan'. Default: 'ceemdan'.
    num_ensemble : int, optional
        Number of ensemble members (for EEMD/CEEMDAN). Default: 100.
    noise_std : float, optional
        Standard deviation of added noise (for EEMD/CEEMDAN). Default: 0.2.
    max_imf : int, optional
        Maximum number of IMFs to extract. Default: None (automatic).
    max_iter : int, optional
        Maximum sifting iterations. Default: 1000.
    sampling_rate : float, optional
        Effective sampling rate of RRI series (Hz). Default: 4.0 Hz.
    random_seed : int, optional
        Random seed for reproducibility. Default: None.

    Returns
    -------
    imfs : array
        Extracted IMFs (n_imfs, n_samples).
    residue : array
        Final residue.
    inst_amplitude : array
        Instantaneous amplitude for each IMF.
    inst_frequency : array
        Instantaneous frequency for each IMF.
    inst_phase : array
        Instantaneous phase for each IMF.
    imf_energy : array
        Energy of each IMF.
    imf_frequency_mean : array
        Mean instantaneous frequency of each IMF (Hz).
    imf_frequency_std : array
        Standard deviation of instantaneous frequency of each IMF (Hz).
    total_energy : float
        Total energy across all IMFs.
    energy_ratio : array
        Energy ratio of each IMF to total energy.

    Notes
    -----
    - CEEMDAN is recommended for best spectral separation
    - IMFs represent oscillatory components at different time scales
    - Lower-order IMFs contain higher frequency components
    - Energy distribution reveals variability patterns

    References
    ----------
    .. [Huang98] Huang et al. (1998). The empirical mode decomposition and
                 the Hilbert spectrum.
    .. [Torres11] Torres et al. (2011). A complete ensemble EMD with adaptive noise.

    Example
    -------
    >>> from biosppy.signals import hrv
    >>> import numpy as np
    >>> rri = np.random.randn(100) * 50 + 800  # Simulated RRI
    >>> result = hrv.hht_variability(rri=rri, method='ceemdan')
    >>> print(f"Number of IMFs: {len(result['imfs'])}")
    """
    # Check inputs
    if rri is None:
        raise TypeError("Please specify RR-intervals.")

    rri = np.array(rri, dtype=float)

    # Import EMD module
    from . import emd as emd_module

    # Perform decomposition
    if method.lower() == 'emd':
        imfs, residue = emd_module.emd(signal=rri, max_imf=max_imf, max_iter=max_iter)
    elif method.lower() == 'eemd':
        imfs, residue = emd_module.eemd(signal=rri, num_ensemble=num_ensemble,
                                       noise_std=noise_std, max_imf=max_imf,
                                       max_iter=max_iter, random_seed=random_seed)
    elif method.lower() == 'ceemdan':
        imfs, residue = emd_module.ceemdan(signal=rri, num_ensemble=num_ensemble,
                                          noise_std=noise_std, max_imf=max_imf,
                                          max_iter=max_iter, random_seed=random_seed)
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'emd', 'eemd', or 'ceemdan'.")

    # Compute Hilbert spectrum
    inst_amplitude, inst_frequency, inst_phase = emd_module.hilbert_spectrum(
        imfs=imfs, sampling_rate=sampling_rate)

    # Compute IMF features
    n_imfs = len(imfs)
    imf_energy = np.zeros(n_imfs)
    imf_frequency_mean = np.zeros(n_imfs)
    imf_frequency_std = np.zeros(n_imfs)

    for i in range(n_imfs):
        # Energy: integral of squared amplitude
        imf_energy[i] = np.sum(imfs[i] ** 2)

        # Mean and std of instantaneous frequency
        valid_freq = inst_frequency[i][inst_frequency[i] > 0]
        if len(valid_freq) > 0:
            imf_frequency_mean[i] = np.mean(valid_freq)
            imf_frequency_std[i] = np.std(valid_freq)

    # Total energy and energy ratio
    total_energy = np.sum(imf_energy)
    if total_energy > 0:
        energy_ratio = imf_energy / total_energy
    else:
        energy_ratio = np.zeros(n_imfs)

    # Output
    args = (imfs, residue, inst_amplitude, inst_frequency, inst_phase,
            imf_energy, imf_frequency_mean, imf_frequency_std,
            total_energy, energy_ratio)
    names = ('imfs', 'residue', 'inst_amplitude', 'inst_frequency', 'inst_phase',
             'imf_energy', 'imf_frequency_mean', 'imf_frequency_std',
             'total_energy', 'energy_ratio')

    return utils.ReturnTuple(args, names)


def hht_frequency_bands(rri=None, method='ceemdan', fbands=None, num_ensemble=100,
                       noise_std=0.2, max_imf=None, sampling_rate=4.0, random_seed=None):
    """Compute frequency-domain HRV features using HHT decomposition.

    Maps IMFs to traditional HRV frequency bands (VLF, LF, HF) based on
    their mean instantaneous frequency.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    coupling_interval_min : float
        Minimum coupling interval (ms).
    coupling_interval_max : float
        Maximum coupling interval (ms).
    prematurity_threshold : float
        Prematurity criterion threshold.
    compensatory_pause_threshold : float
        Compensatory pause criterion threshold.

    Returns
    -------
    vpc_indices : array
        Indices of detected VPCs.
    """

    vpc_indices = []

    # compute local average using sliding window
    window_size = 5
    local_avg = np.convolve(rri, np.ones(window_size)/window_size, mode='same')

    for i in range(1, len(rri) - 1):
        # get current and next RR interval
        rr_current = rri[i]
        rr_next = rri[i + 1]

        # check coupling interval range
        if rr_current < coupling_interval_min or rr_current > coupling_interval_max:
            continue

        # check prematurity criterion
        # VPC coupling interval should be shorter than local average
        if rr_current > prematurity_threshold * local_avg[i]:
            continue

        # check compensatory pause criterion
        # Post-VPC interval should be longer than local average
        if rr_next < compensatory_pause_threshold * local_avg[i]:
            continue

        # check that it's not part of a couplet or run
        # (i.e., previous interval should be normal)
        if i > 1:
            rr_prev = rri[i - 1]
            if rr_prev < prematurity_threshold * local_avg[i - 1]:
                continue

        vpc_indices.append(i)

    return np.array(vpc_indices, dtype=int)


def _plot_hrt(rri, vpc_indices, to, ts):
    """Plots Heart Rate Turbulence analysis results.
    method : str, optional
        Decomposition method: 'emd', 'eemd', or 'ceemdan'. Default: 'ceemdan'.
    fbands : dict, optional
        Frequency bands (Hz). If None, uses default HRV bands.
        Default: {'vlf': [0.003, 0.04], 'lf': [0.04, 0.15], 'hf': [0.15, 0.4]}.
    num_ensemble : int, optional
        Number of ensemble members. Default: 100.
    noise_std : float, optional
        Noise standard deviation. Default: 0.2.
    max_imf : int, optional
        Maximum number of IMFs. Default: None.
    sampling_rate : float, optional
        Effective sampling rate (Hz). Default: 4.0 Hz.
    random_seed : int, optional
        Random seed. Default: None.

    Returns
    -------
    vlf_power : float
        Very Low Frequency power (ms²).
    lf_power : float
        Low Frequency power (ms²).
    hf_power : float
        High Frequency power (ms²).
    total_power : float
        Total power (ms²).
    lf_hf_ratio : float
        LF/HF ratio.
    vlf_norm : float
        Normalized VLF power (%).
    lf_norm : float
        Normalized LF power (%).
    hf_norm : float
        Normalized HF power (%).
    imf_to_band : dict
        Mapping of IMF index to frequency band.

    Notes
    -----
    - IMFs are mapped to bands based on mean instantaneous frequency
    - Power is computed as sum of squared amplitudes
    - Normalized powers exclude VLF contribution

    Example
    -------
    >>> from biosppy.signals import hrv
    >>> import numpy as np
    >>> rri = np.random.randn(200) * 50 + 800
    >>> result = hrv.hht_frequency_bands(rri=rri)
    >>> print(f"LF/HF ratio: {result['lf_hf_ratio']:.2f}")
    """
    # Check inputs
    if rri is None:
        raise TypeError("Please specify RR-intervals.")

    # Default frequency bands
    if fbands is None:
        fbands = {'vlf': [0.003, 0.04], 'lf': [0.04, 0.15], 'hf': [0.15, 0.4]}

    # Perform HHT decomposition
    hht_result = hht_variability(rri=rri, method=method, num_ensemble=num_ensemble,
                                noise_std=noise_std, max_imf=max_imf,
                                sampling_rate=sampling_rate, random_seed=random_seed)

    imfs = hht_result['imfs']
    inst_frequency = hht_result['inst_frequency']
    imf_frequency_mean = hht_result['imf_frequency_mean']
    imf_energy = hht_result['imf_energy']

    n_imfs = len(imfs)

    # Map IMFs to frequency bands
    imf_to_band = {}
    vlf_power = 0.0
    lf_power = 0.0
    hf_power = 0.0

    for i in range(n_imfs):
        mean_freq = imf_frequency_mean[i]

        # Assign to band
        if fbands['vlf'][0] <= mean_freq < fbands['vlf'][1]:
            band = 'vlf'
            vlf_power += imf_energy[i]
        elif fbands['lf'][0] <= mean_freq < fbands['lf'][1]:
            band = 'lf'
            lf_power += imf_energy[i]
        elif fbands['hf'][0] <= mean_freq < fbands['hf'][1]:
            band = 'hf'
            hf_power += imf_energy[i]
        else:
            band = 'other'

        imf_to_band[i] = {'band': band, 'mean_freq': mean_freq}

    # Total power
    total_power = vlf_power + lf_power + hf_power

    # LF/HF ratio
    if hf_power > 0:
        lf_hf_ratio = lf_power / hf_power
    else:
        lf_hf_ratio = np.nan

    # Normalized powers (excluding VLF)
    power_no_vlf = lf_power + hf_power
    if power_no_vlf > 0:
        lf_norm = (lf_power / power_no_vlf) * 100.0
        hf_norm = (hf_power / power_no_vlf) * 100.0
        vlf_norm = (vlf_power / total_power) * 100.0 if total_power > 0 else 0.0
    else:
        lf_norm = 0.0
        hf_norm = 0.0
        vlf_norm = 0.0

    # Output
    args = (vlf_power, lf_power, hf_power, total_power, lf_hf_ratio,
            vlf_norm, lf_norm, hf_norm, imf_to_band)
    names = ('vlf_power', 'lf_power', 'hf_power', 'total_power', 'lf_hf_ratio',
             'vlf_norm', 'lf_norm', 'hf_norm', 'imf_to_band')

    return utils.ReturnTuple(args, names)


def hht_nonlinear_features(rri=None, method='ceemdan', num_ensemble=100,
                          noise_std=0.2, max_imf=None, sampling_rate=4.0,
                          random_seed=None):
    """Compute non-linear HRV features using HHT.

    Extracts complexity and entropy measures from IMF decomposition.

    Parameters
    ----------
    rri : array
        RR-intervals (ms).
    vpc_indices : array
        Indices of VPCs.
    to : float
        Turbulence Onset.
    ts : float
        Turbulence Slope.
    """

    # average RR intervals around VPCs
    window_pre = 5
    window_post = 20

    averaged_rr = None
    count = 0

    for vpc_idx in vpc_indices:
        if vpc_idx < window_pre or vpc_idx + window_post >= len(rri):
            continue

        # extract window around VPC
        window = rri[vpc_idx - window_pre:vpc_idx + window_post + 1]

        if averaged_rr is None:
            averaged_rr = window.copy()
        else:
            averaged_rr += window

        count += 1

    if count > 0 and averaged_rr is not None:
        averaged_rr /= count

        # create time axis relative to VPC
        time_axis = np.arange(-window_pre, window_post + 1)

        # plot
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis, averaged_rr, 'b-', linewidth=2)
        plt.axvline(x=0, color='r', linestyle='--', label='VPC')
        plt.xlabel('RR Interval Index (relative to VPC)')
        plt.ylabel('RR Interval (ms)')
        plt.title(f'Heart Rate Turbulence\nTO = {to:.2f}%, TS = {ts:.2f} ms/RR')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        warnings.warn("Could not generate HRT plot. No valid VPCs with "
                     "sufficient surrounding intervals.")
    method : str, optional
        Decomposition method. Default: 'ceemdan'.
    num_ensemble : int, optional
        Number of ensemble members. Default: 100.
    noise_std : float, optional
        Noise standard deviation. Default: 0.2.
    max_imf : int, optional
        Maximum number of IMFs. Default: None.
    sampling_rate : float, optional
        Sampling rate (Hz). Default: 4.0 Hz.
    random_seed : int, optional
        Random seed. Default: None.

    Returns
    -------
    n_imfs : int
        Number of extracted IMFs.
    complexity_index : float
        Complexity index (number of IMFs / log2(N)).
    energy_entropy : float
        Shannon entropy of energy distribution.
    frequency_entropy : float
        Shannon entropy of frequency distribution.
    residue_std : float
        Standard deviation of residue (trend component).
    reconstruction_error : float
        RMS error between original and reconstructed signal.

    Notes
    -----
    - Complexity index: higher values indicate more complex variability
    - Energy entropy: uniformity of energy distribution across IMFs
    - Lower reconstruction error indicates better decomposition

    Example
    -------
    >>> from biosppy.signals import hrv
    >>> import numpy as np
    >>> rri = np.random.randn(200) * 50 + 800
    >>> result = hrv.hht_nonlinear_features(rri=rri)
    >>> print(f"Complexity index: {result['complexity_index']:.2f}")
    """
    # Check inputs
    if rri is None:
        raise TypeError("Please specify RR-intervals.")

    rri = np.array(rri, dtype=float)
    n = len(rri)

    # Perform HHT decomposition
    hht_result = hht_variability(rri=rri, method=method, num_ensemble=num_ensemble,
                                noise_std=noise_std, max_imf=max_imf,
                                sampling_rate=sampling_rate, random_seed=random_seed)

    imfs = hht_result['imfs']
    residue = hht_result['residue']
    imf_energy = hht_result['imf_energy']
    energy_ratio = hht_result['energy_ratio']
    total_energy = hht_result['total_energy']

    n_imfs = len(imfs)

    # Complexity index
    max_complexity = np.log2(n)
    if max_complexity > 0:
        complexity_index = n_imfs / max_complexity
    else:
        complexity_index = 0.0

    # Energy entropy (Shannon entropy of energy distribution)
    energy_probs = energy_ratio[energy_ratio > 0]
    if len(energy_probs) > 0:
        energy_entropy = -np.sum(energy_probs * np.log2(energy_probs))
    else:
        energy_entropy = 0.0

    # Frequency entropy (Shannon entropy of mean frequencies)
    imf_freq_mean = hht_result['imf_frequency_mean']
    freq_probs = imf_freq_mean / (np.sum(imf_freq_mean) + 1e-10)
    freq_probs = freq_probs[freq_probs > 0]
    if len(freq_probs) > 0:
        frequency_entropy = -np.sum(freq_probs * np.log2(freq_probs))
    else:
        frequency_entropy = 0.0

    # Residue standard deviation (trend variability)
    residue_std = np.std(residue)

    # Reconstruction error
    reconstructed = np.sum(imfs, axis=0) + residue
    reconstruction_error = np.sqrt(np.mean((rri - reconstructed) ** 2))

    # Output
    args = (n_imfs, complexity_index, energy_entropy, frequency_entropy,
            residue_std, reconstruction_error)
    names = ('n_imfs', 'complexity_index', 'energy_entropy', 'frequency_entropy',
             'residue_std', 'reconstruction_error')

    return utils.ReturnTuple(args, names)
