#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for multichannel signal processing.

Author: BioSPPy Development Team
"""

import unittest
import numpy as np
from biosppy.signals import multichannel, ecg, abp
from biosppy import storage


class TestMultiChannelSignal(unittest.TestCase):
    """Test MultiChannelSignal class."""

    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 1000.0
        self.duration = 10.0  # seconds
        self.n_samples = int(self.sampling_rate * self.duration)

        # Create synthetic signals
        np.random.seed(42)
        t = np.arange(self.n_samples) / self.sampling_rate

        # Synthetic ECG-like signal
        self.ecg_signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(self.n_samples)

        # Synthetic ABP-like signal
        self.abp_signal = 100 + 20 * np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.random.randn(self.n_samples)

        # Synthetic respiratory signal
        self.resp_signal = np.sin(2 * np.pi * 0.25 * t) + 0.05 * np.random.randn(self.n_samples)

    def test_initialization(self):
        """Test MultiChannelSignal initialization."""
        mc = multichannel.MultiChannelSignal(sampling_rate=self.sampling_rate)

        self.assertEqual(mc.sampling_rate, self.sampling_rate)
        self.assertEqual(len(mc.channels), 0)

    def test_add_channel(self):
        """Test adding channels."""
        mc = multichannel.MultiChannelSignal(sampling_rate=self.sampling_rate)

        # Add ECG channel
        mc.add_channel('ECG', self.ecg_signal, channel_type='ecg')

        self.assertEqual(len(mc.channels), 1)
        self.assertIn('ECG', mc.channels)
        self.assertEqual(mc.channels['ECG']['type'], 'ecg')
        self.assertEqual(mc.channels['ECG']['length'], len(self.ecg_signal))

    def test_get_channel(self):
        """Test getting channel data."""
        mc = multichannel.MultiChannelSignal(sampling_rate=self.sampling_rate)
        mc.add_channel('ECG', self.ecg_signal, channel_type='ecg')

        retrieved = mc.get_channel('ECG')
        np.testing.assert_array_equal(retrieved, self.ecg_signal)

    def test_get_channel_not_found(self):
        """Test getting non-existent channel."""
        mc = multichannel.MultiChannelSignal(sampling_rate=self.sampling_rate)

        with self.assertRaises(KeyError):
            mc.get_channel('NonExistent')

    def test_list_channels(self):
        """Test listing channels."""
        mc = multichannel.MultiChannelSignal(sampling_rate=self.sampling_rate)
        mc.add_channel('ECG', self.ecg_signal, channel_type='ecg')
        mc.add_channel('ABP', self.abp_signal, channel_type='abp')

        channels = mc.list_channels()

        self.assertEqual(len(channels), 2)
        self.assertIn('ECG', channels)
        self.assertIn('ABP', channels)

    def test_get_time_vector(self):
        """Test getting time vector."""
        mc = multichannel.MultiChannelSignal(sampling_rate=self.sampling_rate)
        mc.add_channel('ECG', self.ecg_signal, channel_type='ecg')

        time = mc.get_time_vector('ECG')

        self.assertEqual(len(time), len(self.ecg_signal))
        self.assertAlmostEqual(time[-1], self.duration - 1.0/self.sampling_rate)

    def test_get_time_vector_with_offset(self):
        """Test getting time vector with offset."""
        mc = multichannel.MultiChannelSignal(sampling_rate=self.sampling_rate)
        offset = 0.5  # 500 ms offset
        mc.add_channel('ECG', self.ecg_signal, channel_type='ecg', offset=offset)

        time = mc.get_time_vector('ECG')

        self.assertAlmostEqual(time[0], offset)

    def test_align_channels_truncate(self):
        """Test channel alignment with truncation."""
        mc = multichannel.MultiChannelSignal(sampling_rate=self.sampling_rate)

        # Add channels with different lengths
        short_signal = self.ecg_signal[:5000]
        mc.add_channel('ECG', self.ecg_signal, channel_type='ecg')
        mc.add_channel('Short', short_signal, channel_type='ecg')

        aligned = mc.align_channels()

        # All should be aligned to shortest length
        self.assertEqual(len(aligned['ECG']), len(short_signal))
        self.assertEqual(len(aligned['Short']), len(short_signal))

    def test_align_channels_pad(self):
        """Test channel alignment with padding."""
        mc = multichannel.MultiChannelSignal(sampling_rate=self.sampling_rate)

        # Add channels with different lengths
        short_signal = self.ecg_signal[:5000]
        mc.add_channel('ECG', self.ecg_signal, channel_type='ecg')
        mc.add_channel('Short', short_signal, channel_type='ecg')

        # Align to longer length
        aligned = mc.align_channels(target_length=len(self.ecg_signal))

        # All should be aligned to target length
        self.assertEqual(len(aligned['ECG']), len(self.ecg_signal))
        self.assertEqual(len(aligned['Short']), len(self.ecg_signal))

    def test_multichannel_convenience_function_list(self):
        """Test multichannel convenience function with list input."""
        signals = [self.ecg_signal, self.abp_signal]
        channel_types = ['ecg', 'abp']
        channel_names = ['ECG', 'ABP']

        mc = multichannel.multichannel(
            signals=signals,
            sampling_rate=self.sampling_rate,
            channel_types=channel_types,
            channel_names=channel_names,
            process=False,
            synchronize=False
        )

        self.assertEqual(len(mc.channels), 2)
        self.assertIn('ECG', mc.channels)
        self.assertIn('ABP', mc.channels)

    def test_multichannel_convenience_function_dict(self):
        """Test multichannel convenience function with dict input."""
        signals = {'ECG': self.ecg_signal, 'ABP': self.abp_signal}
        channel_types = {'ECG': 'ecg', 'ABP': 'abp'}

        mc = multichannel.multichannel(
            signals=signals,
            sampling_rate=self.sampling_rate,
            channel_types=channel_types,
            process=False,
            synchronize=False
        )

        self.assertEqual(len(mc.channels), 2)
        self.assertIn('ECG', mc.channels)
        self.assertIn('ABP', mc.channels)

    def test_repr(self):
        """Test string representation."""
        mc = multichannel.MultiChannelSignal(sampling_rate=self.sampling_rate)
        mc.add_channel('ECG', self.ecg_signal, channel_type='ecg')

        repr_str = repr(mc)

        self.assertIn('MultiChannelSignal', repr_str)
        self.assertIn('1000', repr_str)  # sampling rate
        self.assertIn('ECG', repr_str)


class TestMultiChannelProcessing(unittest.TestCase):
    """Test multi-channel signal processing."""

    def setUp(self):
        """Set up test fixtures."""
        # Load real ECG signal for processing tests
        try:
            self.ecg_signal, _ = storage.load_txt('./examples/ecg.txt')
            self.sampling_rate = 1000.0
            self.has_real_data = True
        except:
            # Fallback to synthetic data
            self.sampling_rate = 1000.0
            self.duration = 10.0
            self.n_samples = int(self.sampling_rate * self.duration)
            np.random.seed(42)
            t = np.arange(self.n_samples) / self.sampling_rate
            self.ecg_signal = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(self.n_samples)
            self.has_real_data = False

    def test_process_channel_ecg(self):
        """Test processing ECG channel."""
        if not self.has_real_data:
            self.skipTest("Real data not available")

        mc = multichannel.MultiChannelSignal(sampling_rate=self.sampling_rate)
        mc.add_channel('ECG', self.ecg_signal, channel_type='ecg')

        # Process ECG
        results = mc.process_channel('ECG')

        # Check that we got expected outputs
        self.assertIn('rpeaks', results.keys())
        self.assertIn('heart_rate', results.keys())
        self.assertGreater(len(results['rpeaks']), 0)

    def test_process_channel_invalid_type(self):
        """Test processing channel with invalid type."""
        mc = multichannel.MultiChannelSignal(sampling_rate=self.sampling_rate)
        mc.add_channel('Unknown', self.ecg_signal, channel_type='unknown_type')

        with self.assertRaises(ValueError):
            mc.process_channel('Unknown')

    def test_get_processed(self):
        """Test getting processed results."""
        if not self.has_real_data:
            self.skipTest("Real data not available")

        mc = multichannel.MultiChannelSignal(sampling_rate=self.sampling_rate)
        mc.add_channel('ECG', self.ecg_signal, channel_type='ecg')

        # Process first
        mc.process_channel('ECG')

        # Get processed results
        results = mc.get_processed('ECG')

        self.assertIn('rpeaks', results.keys())

    def test_get_processed_before_processing(self):
        """Test getting processed results before processing."""
        mc = multichannel.MultiChannelSignal(sampling_rate=self.sampling_rate)
        mc.add_channel('ECG', self.ecg_signal, channel_type='ecg')

        with self.assertRaises(KeyError):
            mc.get_processed('ECG')


class TestSynchronization(unittest.TestCase):
    """Test signal synchronization."""

    def setUp(self):
        """Set up test fixtures."""
        self.sampling_rate = 1000.0
        self.duration = 10.0
        self.n_samples = int(self.sampling_rate * self.duration)

        # Create two similar signals with known offset
        np.random.seed(42)
        t = np.arange(self.n_samples) / self.sampling_rate

        self.signal1 = np.sin(2 * np.pi * 1.0 * t)
        # Signal2 is signal1 shifted by 100 samples (0.1 seconds)
        self.signal2 = np.roll(self.signal1, 100)

    def test_manual_synchronization(self):
        """Test manual synchronization."""
        mc = multichannel.MultiChannelSignal(sampling_rate=self.sampling_rate)
        mc.add_channel('S1', self.signal1, offset=0.0)
        mc.add_channel('S2', self.signal2, offset=0.1)

        offsets = mc.synchronize(method='manual')

        self.assertEqual(offsets['S1'], 0.0)
        self.assertEqual(offsets['S2'], 0.1)


if __name__ == '__main__':
    unittest.main()
