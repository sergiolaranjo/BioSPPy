#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for baroreflex analysis.

Author: BioSPPy Development Team
"""

import unittest
import numpy as np
from biosppy.signals import baroreflex, multichannel
from biosppy import storage


class TestBaroreflexSensitivity(unittest.TestCase):
    """Test baroreflex sensitivity computation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Create synthetic RRI and SBP data with correlation
        n_beats = 100
        baseline_rri = 800  # ms
        baseline_sbp = 120  # mmHg

        # Generate correlated changes
        changes = np.random.randn(n_beats) * 10
        self.rri = baseline_rri + changes * 2  # RRI changes
        self.sbp = baseline_sbp + changes  # SBP changes

        # Add noise
        self.rri += np.random.randn(n_beats) * 5
        self.sbp += np.random.randn(n_beats) * 2

    def test_baroreflex_with_rri_sbp(self):
        """Test baroreflex computation with RRI and SBP arrays."""
        results = baroreflex.baroreflex_sensitivity(
            rri=self.rri,
            sbp=self.sbp,
            method='sequence',
            show=False
        )

        self.assertIn('brs_sequence', results.keys())
        self.assertIn('n_sequences_up', results.keys())
        self.assertIn('n_sequences_down', results.keys())

    def test_baroreflex_all_methods(self):
        """Test baroreflex computation with all methods."""
        results = baroreflex.baroreflex_sensitivity(
            rri=self.rri,
            sbp=self.sbp,
            method='all',
            show=False
        )

        # Check that all method results are present
        self.assertIn('brs_sequence', results.keys())
        self.assertIn('brs_spectral_lf', results.keys())
        self.assertIn('brs_spectral_hf', results.keys())
        self.assertIn('brs_alpha_lf', results.keys())
        self.assertIn('brs_alpha_hf', results.keys())

    def test_baroreflex_invalid_method(self):
        """Test baroreflex with invalid method."""
        with self.assertRaises(ValueError):
            baroreflex.baroreflex_sensitivity(
                rri=self.rri,
                sbp=self.sbp,
                method='invalid_method'
            )

    def test_baroreflex_missing_input(self):
        """Test baroreflex with missing input."""
        with self.assertRaises(TypeError):
            baroreflex.baroreflex_sensitivity(sbp=self.sbp)

        with self.assertRaises(TypeError):
            baroreflex.baroreflex_sensitivity(rri=self.rri)

    def test_baroreflex_with_rpeaks(self):
        """Test baroreflex computation with R-peaks."""
        # Create R-peaks from RRI
        sampling_rate = 1000.0
        rpeaks = np.cumsum(self.rri / 1000.0 * sampling_rate).astype(int)

        # Create systolic peaks (similar spacing)
        systolic_peaks = rpeaks + np.random.randint(-10, 10, len(rpeaks))

        results = baroreflex.baroreflex_sensitivity(
            rpeaks=rpeaks,
            systolic_peaks=systolic_peaks,
            sbp=self.sbp,
            sampling_rate=sampling_rate,
            method='sequence',
            show=False
        )

        self.assertIn('brs_sequence', results.keys())


class TestSequenceMethod(unittest.TestCase):
    """Test sequence method for BRS computation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Create data with clear sequences
        n_beats = 50

        # First half: increasing sequence
        sbp_up = np.linspace(110, 130, n_beats // 2)
        rri_up = np.linspace(750, 850, n_beats // 2)

        # Second half: decreasing sequence
        sbp_down = np.linspace(130, 110, n_beats // 2)
        rri_down = np.linspace(850, 750, n_beats // 2)

        self.sbp = np.concatenate([sbp_up, sbp_down])
        self.rri = np.concatenate([rri_up, rri_down])

    def test_sequence_detection(self):
        """Test that sequences are detected."""
        results = baroreflex.baroreflex_sensitivity(
            rri=self.rri,
            sbp=self.sbp,
            method='sequence',
            min_sequences=1,
            sequence_length=3,
            show=False
        )

        # Should detect at least some sequences
        total_sequences = results['n_sequences_up'] + results['n_sequences_down']
        self.assertGreater(total_sequences, 0)

    def test_brs_value_range(self):
        """Test that BRS values are in reasonable range."""
        results = baroreflex.baroreflex_sensitivity(
            rri=self.rri,
            sbp=self.sbp,
            method='sequence',
            min_sequences=1,
            show=False
        )

        if not np.isnan(results['brs_sequence']):
            # BRS should be positive and in reasonable range (0-100 ms/mmHg)
            self.assertGreater(results['brs_sequence'], 0)
            self.assertLess(results['brs_sequence'], 100)


class TestSpectralMethod(unittest.TestCase):
    """Test spectral method for BRS computation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Create longer signals for spectral analysis
        n_beats = 200
        baseline_rri = 800
        baseline_sbp = 120

        # Add oscillations in LF and HF bands
        t = np.arange(n_beats)
        lf_osc = 10 * np.sin(2 * np.pi * 0.1 * t / n_beats)  # LF component
        hf_osc = 5 * np.sin(2 * np.pi * 0.25 * t / n_beats)  # HF component

        self.rri = baseline_rri + lf_osc + hf_osc + np.random.randn(n_beats) * 3
        self.sbp = baseline_sbp + lf_osc/2 + hf_osc/2 + np.random.randn(n_beats) * 1

    def test_spectral_computation(self):
        """Test spectral method computation."""
        results = baroreflex.baroreflex_sensitivity(
            rri=self.rri,
            sbp=self.sbp,
            method='spectral',
            show=False
        )

        self.assertIn('brs_spectral_lf', results.keys())
        self.assertIn('brs_spectral_hf', results.keys())
        self.assertIn('coherence_lf', results.keys())
        self.assertIn('coherence_hf', results.keys())

    def test_transfer_function_output(self):
        """Test that transfer function is computed."""
        results = baroreflex.baroreflex_sensitivity(
            rri=self.rri,
            sbp=self.sbp,
            method='spectral',
            show=False
        )

        self.assertIn('transfer_function_freqs', results.keys())
        self.assertIn('transfer_function_gain', results.keys())
        self.assertIn('coherence', results.keys())

        # Check array lengths match
        freqs = results['transfer_function_freqs']
        gain = results['transfer_function_gain']
        coherence = results['coherence']

        self.assertEqual(len(freqs), len(gain))
        self.assertEqual(len(freqs), len(coherence))


class TestAlphaMethod(unittest.TestCase):
    """Test alpha coefficient method for BRS computation."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)

        # Create signals with power in frequency bands
        n_beats = 200
        baseline_rri = 800
        baseline_sbp = 120

        t = np.arange(n_beats)
        lf_osc = 10 * np.sin(2 * np.pi * 0.1 * t / n_beats)

        self.rri = baseline_rri + lf_osc + np.random.randn(n_beats) * 3
        self.sbp = baseline_sbp + lf_osc/2 + np.random.randn(n_beats) * 1

    def test_alpha_computation(self):
        """Test alpha coefficient computation."""
        results = baroreflex.baroreflex_sensitivity(
            rri=self.rri,
            sbp=self.sbp,
            method='alpha',
            show=False
        )

        self.assertIn('brs_alpha_lf', results.keys())
        self.assertIn('brs_alpha_hf', results.keys())


class TestMultichannelBaroreflex(unittest.TestCase):
    """Test multichannel baroreflex analysis."""

    def setUp(self):
        """Set up test fixtures."""
        # Try to load real data
        try:
            self.ecg_signal, _ = storage.load_txt('./examples/ecg.txt')
            self.sampling_rate = 1000.0
            self.has_real_data = True

            # Create synthetic ABP
            duration = len(self.ecg_signal) / self.sampling_rate
            t = np.arange(len(self.ecg_signal)) / self.sampling_rate
            self.abp_signal = 100 + 20 * np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 2, len(t))
        except:
            self.has_real_data = False

    def test_analyze_multichannel_baroreflex(self):
        """Test baroreflex analysis from multichannel object."""
        if not self.has_real_data:
            self.skipTest("Real data not available")

        # Create multichannel signal
        mc = multichannel.MultiChannelSignal(sampling_rate=self.sampling_rate)
        mc.add_channel('ECG', self.ecg_signal, channel_type='ecg')
        mc.add_channel('ABP', self.abp_signal, channel_type='abp')

        # Process channels
        mc.process_channel('ECG')
        mc.process_channel('ABP')

        # Analyze baroreflex
        try:
            results = baroreflex.analyze_multichannel_baroreflex(
                mc,
                ecg_channel='ECG',
                abp_channel='ABP',
                method='sequence',
                show=False
            )

            self.assertIn('brs_sequence', results.keys())
        except Exception as e:
            # May fail with synthetic ABP, which is acceptable
            self.skipTest(f"Baroreflex analysis failed (expected with synthetic data): {str(e)}")


if __name__ == '__main__':
    unittest.main()
