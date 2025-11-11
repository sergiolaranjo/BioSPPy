#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for HRV Wavelet Analysis
====================================

Basic tests for the hrv_wavelet() function.

"""

import sys
import os
import unittest
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from biosppy.signals import hrv


class TestHRVWavelet(unittest.TestCase):
    """Test suite for wavelet-based HRV analysis."""

    def setUp(self):
        """Set up test fixtures."""
        # Load example RRI data
        self.rri = np.loadtxt('./examples/rri.txt')

        # Create a synthetic RRI sequence for testing
        np.random.seed(42)
        self.synthetic_rri = 600 + 50 * np.sin(np.linspace(0, 10*np.pi, 200)) + \
                            10 * np.random.randn(200)

    def test_hrv_wavelet_basic(self):
        """Test basic wavelet analysis with default parameters."""
        result = hrv.hrv_wavelet(rri=self.rri, wavelet='db12', level=6)

        # Check that key features are present
        self.assertIn('wavelet_name', result.keys())
        self.assertIn('decomposition_level', result.keys())
        self.assertIn('dwt_energy_total', result.keys())
        self.assertIn('dwt_entropy', result.keys())

        # Check wavelet name
        self.assertEqual(result['wavelet_name'], 'db12')

        # Check that total energy is positive
        self.assertGreater(result['dwt_energy_total'], 0)

        # Check that entropy is non-negative
        self.assertGreaterEqual(result['dwt_entropy'], 0)

    def test_hrv_wavelet_different_wavelets(self):
        """Test with different wavelet families."""
        wavelets = ['db4', 'db8', 'db12', 'sym5']

        for wavelet_name in wavelets:
            with self.subTest(wavelet=wavelet_name):
                result = hrv.hrv_wavelet(rri=self.rri, wavelet=wavelet_name)
                self.assertEqual(result['wavelet_name'], wavelet_name)
                self.assertGreater(result['dwt_energy_total'], 0)

    def test_hrv_wavelet_energy_conservation(self):
        """Test that relative energies sum to approximately 1."""
        result = hrv.hrv_wavelet(rri=self.rri, wavelet='db12', level=4)

        actual_level = result['decomposition_level']

        # Sum all relative energies
        total_rel_energy = result[f'dwt_energy_rel_a{actual_level}']
        for i in range(1, actual_level + 1):
            total_rel_energy += result[f'dwt_energy_rel_d{i}']

        # Should sum to approximately 1.0
        self.assertAlmostEqual(total_rel_energy, 1.0, places=6)

    def test_hrv_wavelet_with_synthetic_data(self):
        """Test with synthetic RRI data."""
        result = hrv.hrv_wavelet(rri=self.synthetic_rri, wavelet='db12', level=5)

        # Check basic requirements
        self.assertIsNotNone(result)
        self.assertGreater(result['dwt_energy_total'], 0)
        self.assertGreater(len(result.keys()), 5)

    def test_hrv_wavelet_invalid_wavelet(self):
        """Test that invalid wavelet raises an error."""
        with self.assertRaises(ValueError):
            hrv.hrv_wavelet(rri=self.rri, wavelet='invalid_wavelet')

    def test_hrv_wavelet_short_signal(self):
        """Test that short signal raises an error."""
        short_rri = np.array([600, 610, 620, 630])  # Very short

        with self.assertRaises(ValueError):
            hrv.hrv_wavelet(rri=short_rri, wavelet='db12')

    def test_hrv_wavelet_no_rri(self):
        """Test that missing RRI raises an error."""
        with self.assertRaises(TypeError):
            hrv.hrv_wavelet(rri=None, wavelet='db12')

    def test_hrv_integration_with_main_function(self):
        """Test integration with main hrv() function."""
        # Test with wavelet parameter
        result = hrv.hrv(rri=self.rri,
                        parameters='wavelet',
                        wavelet='db12',
                        wavelet_level=4,
                        show=False)

        # Check that wavelet features are present
        self.assertIn('dwt_energy_total', result.keys())
        self.assertIn('dwt_entropy', result.keys())

    def test_hrv_all_parameters_including_wavelet(self):
        """Test that 'all' parameter includes wavelet features."""
        result = hrv.hrv(rri=self.rri,
                        parameters='all',
                        wavelet='db12',
                        wavelet_level=4,
                        show=False)

        # Check that all domains are present
        # Time domain
        self.assertIn('sdnn', result.keys())
        # Frequency domain
        self.assertIn('lf_hf', result.keys())
        # Non-linear
        self.assertIn('sampen', result.keys())
        # Wavelet
        self.assertIn('dwt_energy_total', result.keys())

    def test_hrv_wavelet_detrending(self):
        """Test with and without detrending."""
        # With detrending
        result_with = hrv.hrv_wavelet(rri=self.rri, wavelet='db12',
                                      level=4, detrend_rri=True)

        # Without detrending
        result_without = hrv.hrv_wavelet(rri=self.rri, wavelet='db12',
                                        level=4, detrend_rri=False)

        # Both should work
        self.assertGreater(result_with['dwt_energy_total'], 0)
        self.assertGreater(result_without['dwt_energy_total'], 0)

        # Results should be different
        self.assertNotEqual(result_with['dwt_energy_total'],
                          result_without['dwt_energy_total'])


if __name__ == '__main__':
    # Run tests
    print("Running HRV Wavelet Analysis Tests")
    print("=" * 70)

    unittest.main(verbosity=2)
