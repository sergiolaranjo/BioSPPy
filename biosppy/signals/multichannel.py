# -*- coding: utf-8 -*-
"""
biosppy.signals.multichannel
----------------------------

This module provides functionality for simultaneous import and analysis
of multiple signal channels.

:copyright: (c) 2015-2025 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.

"""

# Imports
# standard library
from collections import OrderedDict

# third-party
import numpy as np

# local
from . import ecg, abp, ppg, resp, eda, tools
from .. import utils


class MultiChannelSignal(object):
    """Multi-channel signal container for synchronized analysis.

    This class manages multiple biosignal channels, allowing for:
    - Synchronized temporal alignment
    - Individual channel analysis
    - Combined multi-channel analysis

    Parameters
    ----------
    sampling_rate : float
        Sampling frequency (Hz).

    Attributes
    ----------
    channels : OrderedDict
        Dictionary of channels with signal data.
    sampling_rate : float
        Sampling frequency (Hz).
    """

    def __init__(self, sampling_rate=1000.0):
        """Initialize MultiChannelSignal.

        Parameters
        ----------
        sampling_rate : float, optional
            Sampling frequency (Hz); default is 1000.0 Hz.
        """
        self.channels = OrderedDict()
        self.sampling_rate = sampling_rate
        self._processed = OrderedDict()
        self._sync_offset = OrderedDict()

    def add_channel(self, name, signal, channel_type=None, offset=0.0):
        """Add a signal channel.

        Parameters
        ----------
        name : str
            Channel identifier.
        signal : array
            Signal data.
        channel_type : str, optional
            Type of signal ('ecg', 'abp', 'ppg', 'resp', 'eda').
        offset : float, optional
            Time offset in seconds for synchronization; default is 0.0.

        Returns
        -------
        self : MultiChannelSignal
            Instance of MultiChannelSignal.
        """
        signal = np.array(signal)

        self.channels[name] = {
            'signal': signal,
            'type': channel_type,
            'length': len(signal),
            'duration': len(signal) / self.sampling_rate
        }
        self._sync_offset[name] = offset

        return self

    def get_channel(self, name):
        """Get channel data.

        Parameters
        ----------
        name : str
            Channel identifier.

        Returns
        -------
        signal : array
            Channel signal data.
        """
        if name not in self.channels:
            raise KeyError(f"Channel '{name}' not found.")

        return self.channels[name]['signal']

    def list_channels(self):
        """List all available channels.

        Returns
        -------
        channels : list
            List of channel names.
        """
        return list(self.channels.keys())

    def synchronize(self, reference_channel=None, method='cross_correlation'):
        """Synchronize all channels temporally.

        Parameters
        ----------
        reference_channel : str, optional
            Reference channel name for synchronization.
            If None, uses the first channel.
        method : str, optional
            Synchronization method: 'cross_correlation' or 'manual'.
            Default is 'cross_correlation'.

        Returns
        -------
        offsets : dict
            Time offsets (in seconds) for each channel.
        """
        if not self.channels:
            raise ValueError("No channels available for synchronization.")

        if reference_channel is None:
            reference_channel = list(self.channels.keys())[0]

        if reference_channel not in self.channels:
            raise KeyError(f"Reference channel '{reference_channel}' not found.")

        ref_signal = self.channels[reference_channel]['signal']
        offsets = {reference_channel: 0.0}

        if method == 'cross_correlation':
            for name, channel_data in self.channels.items():
                if name == reference_channel:
                    continue

                signal = channel_data['signal']

                # Compute cross-correlation
                sync_result = tools.synchronize(
                    x=ref_signal,
                    y=signal,
                    detrend=True
                )

                # Convert lag to time offset
                time_offset = sync_result['delay'] / self.sampling_rate
                offsets[name] = time_offset
                self._sync_offset[name] = time_offset

        elif method == 'manual':
            # Use manually specified offsets
            offsets = self._sync_offset.copy()
        else:
            raise ValueError(f"Unknown synchronization method: {method}")

        return offsets

    def process_channel(self, name, **kwargs):
        """Process a single channel based on its type.

        Parameters
        ----------
        name : str
            Channel identifier.
        **kwargs : dict, optional
            Additional keyword arguments for signal processing.

        Returns
        -------
        out : ReturnTuple
            Processing results based on channel type.
        """
        if name not in self.channels:
            raise KeyError(f"Channel '{name}' not found.")

        channel_data = self.channels[name]
        signal = channel_data['signal']
        channel_type = channel_data['type']

        # Process based on channel type
        if channel_type == 'ecg':
            out = ecg.ecg(
                signal=signal,
                sampling_rate=self.sampling_rate,
                show=False,
                **kwargs
            )
        elif channel_type == 'abp':
            out = abp.abp(
                signal=signal,
                sampling_rate=self.sampling_rate,
                show=False,
                **kwargs
            )
        elif channel_type == 'ppg':
            out = ppg.ppg(
                signal=signal,
                sampling_rate=self.sampling_rate,
                show=False,
                **kwargs
            )
        elif channel_type == 'resp':
            out = resp.resp(
                signal=signal,
                sampling_rate=self.sampling_rate,
                show=False,
                **kwargs
            )
        elif channel_type == 'eda':
            out = eda.eda(
                signal=signal,
                sampling_rate=self.sampling_rate,
                show=False,
                **kwargs
            )
        else:
            raise ValueError(
                f"Unknown or unsupported channel type: {channel_type}. "
                f"Supported types: 'ecg', 'abp', 'ppg', 'resp', 'eda'."
            )

        # Store processed results
        self._processed[name] = out

        return out

    def process_all(self, **kwargs):
        """Process all channels.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments for signal processing.
            Channel-specific kwargs can be passed as nested dicts.

        Returns
        -------
        results : dict
            Dictionary of processing results for each channel.
        """
        results = OrderedDict()

        for name in self.channels.keys():
            # Get channel-specific kwargs if available
            channel_kwargs = kwargs.get(name, {})

            try:
                out = self.process_channel(name, **channel_kwargs)
                results[name] = out
            except Exception as e:
                print(f"Warning: Failed to process channel '{name}': {str(e)}")
                results[name] = None

        return results

    def get_processed(self, name):
        """Get processed results for a channel.

        Parameters
        ----------
        name : str
            Channel identifier.

        Returns
        -------
        out : ReturnTuple
            Processed results.
        """
        if name not in self._processed:
            raise KeyError(
                f"Channel '{name}' has not been processed yet. "
                f"Call process_channel('{name}') first."
            )

        return self._processed[name]

    def get_heart_rate_signals(self):
        """Extract heart rate signals from all processed cardiovascular channels.

        Returns
        -------
        hr_signals : dict
            Dictionary with heart rate time series and values for each channel.
        """
        hr_signals = OrderedDict()

        for name, processed in self._processed.items():
            channel_type = self.channels[name]['type']

            # Check if channel has heart rate information
            if channel_type in ['ecg', 'abp', 'ppg']:
                if 'heart_rate_ts' in processed.keys() and 'heart_rate' in processed.keys():
                    hr_signals[name] = {
                        'ts': processed['heart_rate_ts'],
                        'hr': processed['heart_rate']
                    }

        return hr_signals

    def get_time_vector(self, name=None):
        """Get time vector for a channel or all channels.

        Parameters
        ----------
        name : str, optional
            Channel identifier. If None, returns time vectors for all channels.

        Returns
        -------
        time : array or dict
            Time vector(s) in seconds.
        """
        if name is not None:
            if name not in self.channels:
                raise KeyError(f"Channel '{name}' not found.")

            n_samples = self.channels[name]['length']
            offset = self._sync_offset.get(name, 0.0)
            time = np.arange(n_samples) / self.sampling_rate + offset

            return time
        else:
            time_vectors = OrderedDict()
            for ch_name in self.channels.keys():
                time_vectors[ch_name] = self.get_time_vector(ch_name)

            return time_vectors

    def resample_channel(self, name, target_rate):
        """Resample a channel to a target sampling rate.

        Parameters
        ----------
        name : str
            Channel identifier.
        target_rate : float
            Target sampling rate (Hz).

        Returns
        -------
        resampled_signal : array
            Resampled signal.
        """
        if name not in self.channels:
            raise KeyError(f"Channel '{name}' not found.")

        signal = self.channels[name]['signal']

        # Calculate resampling factor
        factor = target_rate / self.sampling_rate
        n_new = int(len(signal) * factor)

        # Resample using scipy
        from scipy import signal as sp_signal
        resampled = sp_signal.resample(signal, n_new)

        return resampled

    def align_channels(self, target_length=None):
        """Align all channels to the same length.

        Parameters
        ----------
        target_length : int, optional
            Target length in samples. If None, uses minimum channel length.

        Returns
        -------
        aligned_signals : dict
            Dictionary of aligned signals.
        """
        if not self.channels:
            raise ValueError("No channels available.")

        if target_length is None:
            # Use minimum length
            target_length = min(ch['length'] for ch in self.channels.values())

        aligned = OrderedDict()
        for name, channel_data in self.channels.items():
            signal = channel_data['signal']

            if len(signal) > target_length:
                # Truncate
                aligned[name] = signal[:target_length]
            elif len(signal) < target_length:
                # Pad with zeros
                pad_width = target_length - len(signal)
                aligned[name] = np.pad(signal, (0, pad_width), mode='constant')
            else:
                aligned[name] = signal

        return aligned

    def __repr__(self):
        """String representation."""
        n_channels = len(self.channels)
        channel_info = []

        for name, data in self.channels.items():
            ch_type = data['type'] or 'unknown'
            duration = data['duration']
            channel_info.append(f"  - {name} ({ch_type}): {duration:.2f}s")

        info = [
            f"MultiChannelSignal(sampling_rate={self.sampling_rate} Hz)",
            f"Channels: {n_channels}",
        ] + channel_info

        return "\n".join(info)


def multichannel(signals, sampling_rate=1000.0, channel_types=None,
                 channel_names=None, process=True, synchronize=True, **kwargs):
    """Process multiple signal channels simultaneously.

    This is a convenience function for creating and processing a
    MultiChannelSignal object.

    Parameters
    ----------
    signals : list or dict
        List of signal arrays or dictionary of {name: signal}.
    sampling_rate : float, optional
        Sampling frequency (Hz); default is 1000.0 Hz.
    channel_types : list or dict, optional
        List of channel types or dictionary of {name: type}.
        Supported types: 'ecg', 'abp', 'ppg', 'resp', 'eda'.
    channel_names : list, optional
        List of channel names (only used if signals is a list).
    process : bool, optional
        If True, process all channels; default is True.
    synchronize : bool, optional
        If True, synchronize channels; default is True.
    **kwargs : dict, optional
        Additional keyword arguments for signal processing.

    Returns
    -------
    mc_signal : MultiChannelSignal
        Multi-channel signal object with processed data.

    Examples
    --------
    >>> # Using lists
    >>> signals = [ecg_signal, abp_signal]
    >>> channel_types = ['ecg', 'abp']
    >>> channel_names = ['ECG', 'ABP']
    >>> mc = multichannel(signals, sampling_rate=1000.0,
    ...                   channel_types=channel_types,
    ...                   channel_names=channel_names)

    >>> # Using dictionaries
    >>> signals = {'ECG': ecg_signal, 'ABP': abp_signal}
    >>> channel_types = {'ECG': 'ecg', 'ABP': 'abp'}
    >>> mc = multichannel(signals, sampling_rate=1000.0,
    ...                   channel_types=channel_types)
    """
    # Create MultiChannelSignal object
    mc_signal = MultiChannelSignal(sampling_rate=sampling_rate)

    # Add channels
    if isinstance(signals, dict):
        # Dictionary input
        for name, signal in signals.items():
            ch_type = None
            if channel_types is not None:
                ch_type = channel_types.get(name)

            mc_signal.add_channel(name, signal, channel_type=ch_type)

    elif isinstance(signals, (list, tuple)):
        # List input
        if channel_names is None:
            channel_names = [f"CH{i}" for i in range(len(signals))]

        if channel_types is None:
            channel_types = [None] * len(signals)
        elif not isinstance(channel_types, (list, tuple)):
            channel_types = [channel_types] * len(signals)

        for i, (name, signal, ch_type) in enumerate(zip(channel_names, signals, channel_types)):
            mc_signal.add_channel(name, signal, channel_type=ch_type)
    else:
        raise TypeError("signals must be a list, tuple, or dict.")

    # Synchronize channels
    if synchronize and len(mc_signal.channels) > 1:
        mc_signal.synchronize()

    # Process channels
    if process:
        mc_signal.process_all(**kwargs)

    return mc_signal
