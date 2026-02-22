# -*- coding: utf-8 -*-
"""
End-to-end tests using the real sample data files in examples/.

Tests each signal processing pipeline with actual physiological signals
to ensure all bug fixes work correctly in practice.
"""

import os
import sys
import tempfile
import warnings
import numpy as np
import pytest

# Suppress FutureWarnings from dependencies
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

# Use non-interactive matplotlib backend
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'examples')


def data_file(name):
    """Helper to get the full path to a data file."""
    path = os.path.join(EXAMPLES_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"Data file {name} not found")
    return path


def has_key(result, key):
    """Check if a ReturnTuple has a named key."""
    return key in result.keys()


# ============================================================================
# Storage / Data Loading
# ============================================================================
class TestDataLoading:
    """Test that all sample data files can be loaded."""

    @pytest.mark.parametrize("filename", [
        'ecg.txt', 'ppg.txt', 'resp.txt', 'emg.txt', 'eda.txt',
        'eeg_ec.txt', 'eeg_eo.txt', 'emg_1.txt', 'acc.txt',
        'bcg.txt', 'pcg.txt', 'pcg_ecg.txt', 'rri.txt',
    ])
    def test_load_txt(self, filename):
        from biosppy import storage
        signal, mdata = storage.load_txt(data_file(filename))
        assert signal is not None
        assert len(signal) > 0, f"{filename}: signal is empty"
        assert isinstance(signal, np.ndarray)
        print(f"  {filename}: {len(signal)} samples loaded OK")


# ============================================================================
# ECG Processing (BUG 1 fix: module shadowing)
# ============================================================================
class TestECGProcessing:
    """Test ECG processing with real data - validates BUG 1 (module shadowing) fix."""

    @pytest.fixture
    def ecg_data(self):
        from biosppy import storage
        signal, mdata = storage.load_txt(data_file('ecg.txt'))
        return signal

    def test_ecg_via_biosppy_ecg(self, ecg_data):
        """BUG 1: biosppy.ecg should be signals.ecg, not synthesizers.ecg."""
        import biosppy
        # This would fail before the BUG 1 fix
        result = biosppy.ecg.ecg(signal=ecg_data, sampling_rate=1000., show=False)
        assert has_key(result, 'ts')
        assert has_key(result, 'filtered')
        assert has_key(result, 'rpeaks')
        assert has_key(result, 'heart_rate')
        print(f"  ECG: {len(result['rpeaks'])} R-peaks detected, "
              f"HR range: {result['heart_rate'].min():.0f}-{result['heart_rate'].max():.0f} bpm")

    def test_ecg_direct_import(self, ecg_data):
        from biosppy.signals import ecg
        result = ecg.ecg(signal=ecg_data, sampling_rate=1000., show=False)
        assert len(result['ts']) == len(ecg_data)
        assert len(result['rpeaks']) > 0

    def test_ecg_with_different_sampling_rates(self, ecg_data):
        """Test ECG with resampled data."""
        from biosppy.signals import ecg
        # Downsample to 500 Hz
        signal_500 = ecg_data[::2]
        result = ecg.ecg(signal=signal_500, sampling_rate=500., show=False)
        assert len(result['ts']) == len(signal_500)
        # Time axis should match 500 Hz sampling rate
        expected_T = (len(signal_500) - 1) / 500.
        assert np.isclose(result['ts'][-1], expected_T, rtol=1e-6)

    def test_ecg_save_to_file(self, ecg_data):
        """Test saving ECG plot to file."""
        from biosppy.signals import ecg
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            result = ecg.ecg(signal=ecg_data, sampling_rate=1000., path=path, show=True)
            assert os.path.exists(path), "ECG plot file was not created"
            size = os.path.getsize(path)
            print(f"  ECG plot saved: {size} bytes")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_ecg_time_axis_correctness(self, ecg_data):
        """BUG 9: Verify endpoint=True gives correct time axis."""
        from biosppy.signals import ecg
        result = ecg.ecg(signal=ecg_data, sampling_rate=1000., show=False)
        ts = result['ts']
        assert ts[0] == 0.0
        expected_T = (len(ecg_data) - 1) / 1000.
        assert np.isclose(ts[-1], expected_T, rtol=1e-6)
        assert np.isclose(ts[1] - ts[0], 1.0 / 1000., rtol=1e-6)


# ============================================================================
# EMG Processing (BUG 6 fix: hardcoded sampling_rate)
# ============================================================================
class TestEMGProcessing:
    """Test EMG processing - validates BUG 6 (hardcoded sampling_rate) fix."""

    @pytest.fixture
    def emg_data(self):
        from biosppy import storage
        signal, _ = storage.load_txt(data_file('emg.txt'))
        return signal

    def test_emg_default_rate(self, emg_data):
        from biosppy.signals import emg
        result = emg.emg(signal=emg_data, sampling_rate=1000., show=False)
        assert has_key(result, 'ts')
        assert has_key(result, 'filtered')
        assert has_key(result, 'onsets')
        print(f"  EMG: {len(result['onsets'])} onsets detected")

    def test_emg_custom_sampling_rate(self, emg_data):
        """BUG 6: sampling_rate should be forwarded to plot, not hardcoded as 1000."""
        from biosppy.signals import emg
        # Downsample to 500 Hz
        signal_500 = emg_data[::2]
        result = emg.emg(signal=signal_500, sampling_rate=500., show=False)
        ts = result['ts']
        expected_T = (len(signal_500) - 1) / 500.
        assert np.isclose(ts[-1], expected_T, rtol=1e-6), \
            f"Time axis end {ts[-1]} != expected {expected_T}"
        assert np.isclose(ts[1] - ts[0], 1.0 / 500., rtol=1e-6), \
            f"Time step {ts[1]-ts[0]} != expected {1.0/500.}"

    def test_emg_save_to_file(self, emg_data):
        from biosppy.signals import emg
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            result = emg.emg(signal=emg_data, sampling_rate=1000., path=path, show=True)
            assert os.path.exists(path)
            size = os.path.getsize(path)
            print(f"  EMG plot saved: {size} bytes")
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ============================================================================
# PPG Processing (BUGs 7, 9 fix: missing path param, endpoint)
# ============================================================================
class TestPPGProcessing:
    """Test PPG processing - validates BUG 7 (path param) and BUG 9 (endpoint) fixes."""

    @pytest.fixture
    def ppg_data(self):
        from biosppy import storage
        signal, _ = storage.load_txt(data_file('ppg.txt'))
        return signal

    def test_ppg_basic(self, ppg_data):
        from biosppy.signals import ppg
        result = ppg.ppg(signal=ppg_data, sampling_rate=1000., show=False)
        assert has_key(result, 'ts')
        assert has_key(result, 'filtered')
        assert has_key(result, 'peaks')
        assert has_key(result, 'heart_rate')
        print(f"  PPG: {len(result['peaks'])} peaks, "
              f"HR range: {result['heart_rate'].min():.0f}-{result['heart_rate'].max():.0f} bpm")

    def test_ppg_save_to_file(self, ppg_data):
        """BUG 7: ppg() should accept path parameter."""
        from biosppy.signals import ppg
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            result = ppg.ppg(signal=ppg_data, sampling_rate=1000., path=path, show=True)
            assert os.path.exists(path)
            size = os.path.getsize(path)
            print(f"  PPG plot saved: {size} bytes")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_ppg_with_units(self, ppg_data):
        """BUG 7: ppg() should accept units parameter."""
        from biosppy.signals import ppg
        result = ppg.ppg(signal=ppg_data, sampling_rate=1000., units='mV', show=False)
        assert result is not None

    def test_ppg_time_axis(self, ppg_data):
        """BUG 9: endpoint=True gives correct time axis."""
        from biosppy.signals import ppg
        result = ppg.ppg(signal=ppg_data, sampling_rate=1000., show=False)
        ts = result['ts']
        assert ts[0] == 0.0
        expected_T = (len(ppg_data) - 1) / 1000.
        assert np.isclose(ts[-1], expected_T, rtol=1e-6)


# ============================================================================
# EDA Processing
# ============================================================================
class TestEDAProcessing:
    """Test EDA processing with real data."""

    @pytest.fixture
    def eda_data(self):
        from biosppy import storage
        signal, _ = storage.load_txt(data_file('eda.txt'))
        return signal

    def test_eda_basic(self, eda_data):
        from biosppy.signals import eda
        result = eda.eda(signal=eda_data, sampling_rate=1000., show=False)
        assert has_key(result, 'ts')
        assert has_key(result, 'filtered')
        assert has_key(result, 'onsets')
        print(f"  EDA: {len(result['onsets'])} SCR onsets detected")

    def test_eda_save_to_file(self, eda_data):
        from biosppy.signals import eda
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            result = eda.eda(signal=eda_data, sampling_rate=1000., path=path, show=True)
            assert os.path.exists(path)
            size = os.path.getsize(path)
            print(f"  EDA plot saved: {size} bytes")
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ============================================================================
# ABP Processing (BUGs 8, 9 fix: missing units/path, endpoint)
# ============================================================================
class TestABPProcessing:
    """Test ABP processing - validates BUG 8 (units/path) and BUG 9 (endpoint) fixes."""

    @pytest.fixture
    def abp_data(self):
        """Use PPG data as ABP proxy (similar waveform)."""
        from biosppy import storage
        signal, _ = storage.load_txt(data_file('ppg.txt'))
        return signal

    def test_abp_basic(self, abp_data):
        from biosppy.signals import abp
        result = abp.abp(signal=abp_data, sampling_rate=1000., show=False)
        assert has_key(result, 'ts')
        assert has_key(result, 'filtered')
        assert has_key(result, 'onsets')
        assert has_key(result, 'heart_rate')
        print(f"  ABP: {len(result['onsets'])} onsets, "
              f"HR range: {result['heart_rate'].min():.0f}-{result['heart_rate'].max():.0f} bpm")

    def test_abp_with_path(self, abp_data):
        """BUG 8: abp() should accept path parameter."""
        from biosppy.signals import abp
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            result = abp.abp(signal=abp_data, sampling_rate=1000., path=path, show=True)
            assert os.path.exists(path)
            size = os.path.getsize(path)
            print(f"  ABP plot saved: {size} bytes")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_abp_with_units(self, abp_data):
        """BUG 8: abp() should accept units parameter."""
        from biosppy.signals import abp
        # Should not raise TypeError
        result = abp.abp(signal=abp_data, sampling_rate=1000., units='mmHg', show=False)
        assert result is not None

    def test_abp_time_axis(self, abp_data):
        """BUG 9: endpoint=True gives correct time axis."""
        from biosppy.signals import abp
        result = abp.abp(signal=abp_data, sampling_rate=1000., show=False)
        ts = result['ts']
        assert ts[0] == 0.0
        expected_T = (len(abp_data) - 1) / 1000.
        assert np.isclose(ts[-1], expected_T, rtol=1e-6)
        assert np.isclose(ts[1] - ts[0], 1.0 / 1000., rtol=1e-6)


# ============================================================================
# BVP Processing (BUG 9 fix: endpoint)
# ============================================================================
class TestBVPProcessing:
    """Test BVP processing - validates BUG 9 (endpoint) fix."""

    @pytest.fixture
    def bvp_data(self):
        from biosppy import storage
        signal, _ = storage.load_txt(data_file('ppg.txt'))
        return signal

    def test_bvp_basic(self, bvp_data):
        from biosppy.signals import bvp
        result = bvp.bvp(signal=bvp_data, sampling_rate=1000., show=False)
        assert has_key(result, 'ts')
        assert has_key(result, 'filtered')
        assert has_key(result, 'onsets')
        print(f"  BVP: {len(result['onsets'])} onsets detected")

    def test_bvp_time_axis(self, bvp_data):
        """BUG 9: endpoint=True gives correct time axis."""
        from biosppy.signals import bvp
        result = bvp.bvp(signal=bvp_data, sampling_rate=1000., show=False)
        ts = result['ts']
        expected_T = (len(bvp_data) - 1) / 1000.
        assert np.isclose(ts[-1], expected_T, rtol=1e-6)


# ============================================================================
# PCG Processing (BUG 10 fix: filter parameter rename)
# ============================================================================
class TestPCGProcessing:
    """Test PCG processing - validates BUG 10 (apply_filter rename) fix."""

    @pytest.fixture
    def pcg_data(self):
        from biosppy import storage
        signal, _ = storage.load_txt(data_file('pcg.txt'))
        return signal

    def test_pcg_basic(self, pcg_data):
        from biosppy.signals import pcg
        result = pcg.pcg(signal=pcg_data, sampling_rate=1000., show=False)
        assert has_key(result, 'ts')
        assert has_key(result, 'filtered')
        assert has_key(result, 'peaks')
        assert has_key(result, 'heart_sounds')
        assert has_key(result, 'heart_rate')
        print(f"  PCG: {len(result['peaks'])} peaks, HR={result['heart_rate']:.0f} bpm")

    def test_pcg_find_peaks_with_apply_filter(self, pcg_data):
        """BUG 10: find_peaks should use apply_filter, not filter."""
        from biosppy.signals import pcg
        from biosppy.signals import tools as st
        # Pre-filter the signal
        filtered, _, _ = st.filter_signal(pcg_data, 'butter', 'bandpass', 2,
                                           np.array([25, 400]), 1000.)
        # Call with new parameter name
        result = pcg.find_peaks(signal=filtered, sampling_rate=1000., apply_filter=False)
        assert has_key(result, 'peaks')
        assert len(result['peaks']) > 0
        print(f"  PCG find_peaks: {len(result['peaks'])} peaks found")

    def test_pcg_homomorphic_filter_with_apply_filter(self, pcg_data):
        """BUG 10: homomorphic_filter should use apply_filter, not filter."""
        from biosppy.signals import pcg
        result = pcg.homomorphic_filter(signal=pcg_data, sampling_rate=1000., apply_filter=True)
        assert has_key(result, 'homomorphic_envelope')
        assert len(result['homomorphic_envelope']) == len(pcg_data)

    def test_pcg_save_to_file(self, pcg_data):
        from biosppy.signals import pcg
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            result = pcg.pcg(signal=pcg_data, sampling_rate=1000., path=path, show=True)
            assert os.path.exists(path)
            size = os.path.getsize(path)
            print(f"  PCG plot saved: {size} bytes")
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ============================================================================
# ACC Processing
# ============================================================================
class TestACCProcessing:
    """Test Accelerometer processing with real data."""

    @pytest.fixture
    def acc_data(self):
        from biosppy import storage
        signal, _ = storage.load_txt(data_file('acc.txt'))
        return signal

    def test_acc_basic(self, acc_data):
        from biosppy.signals.acc import acc
        result = acc(signal=acc_data, sampling_rate=1000., show=False)
        assert has_key(result, 'ts')
        # ACC returns 'signal' (not 'filtered'), plus 'vm', 'sm', 'freq'
        assert has_key(result, 'signal') or has_key(result, 'filtered')
        print(f"  ACC: {len(result['ts'])} time points, keys: {result.keys()}")

    def test_acc_save_to_file(self, acc_data):
        from biosppy.signals.acc import acc
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            result = acc(signal=acc_data, sampling_rate=1000., path=path, show=True)
            assert os.path.exists(path)
            print(f"  ACC plot saved: {os.path.getsize(path)} bytes")
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ============================================================================
# RESP Processing
# ============================================================================
class TestRESPProcessing:
    """Test Respiration processing with real data."""

    @pytest.fixture
    def resp_data(self):
        from biosppy import storage
        signal, _ = storage.load_txt(data_file('resp.txt'))
        return signal

    def test_resp_basic(self, resp_data):
        from biosppy.signals import resp
        result = resp.resp(signal=resp_data, sampling_rate=1000., show=False)
        assert has_key(result, 'ts')
        assert has_key(result, 'filtered')
        assert has_key(result, 'zeros')
        assert has_key(result, 'resp_rate')
        print(f"  RESP: {len(result['zeros'])} zero crossings, "
              f"rate range: {result['resp_rate'].min():.1f}-{result['resp_rate'].max():.1f}")

    def test_resp_save_to_file(self, resp_data):
        from biosppy.signals import resp
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            result = resp.resp(signal=resp_data, sampling_rate=1000., path=path, show=True)
            assert os.path.exists(path)
            size = os.path.getsize(path)
            print(f"  RESP plot saved: {size} bytes")
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ============================================================================
# EEG Processing
# ============================================================================
class TestEEGProcessing:
    """Test EEG processing with real data."""

    @pytest.fixture
    def eeg_data(self):
        from biosppy import storage
        signal, _ = storage.load_txt(data_file('eeg_ec.txt'))
        return signal

    def test_eeg_basic(self, eeg_data):
        from biosppy.signals import eeg
        result = eeg.eeg(signal=eeg_data, sampling_rate=125., show=False)
        assert has_key(result, 'ts')
        assert has_key(result, 'filtered')
        # EEG should produce frequency band powers
        keys = result.keys()
        assert 'theta' in keys or 'alpha' in keys
        print(f"  EEG: {len(result['ts'])} time points, keys: {keys}")

    def test_eeg_save_to_file(self, eeg_data):
        from biosppy.signals import eeg
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            result = eeg.eeg(signal=eeg_data, sampling_rate=125., path=path, show=True)
            assert os.path.exists(path)
            size = os.path.getsize(path)
            print(f"  EEG plot saved: {size} bytes")
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ============================================================================
# HRV Processing
# ============================================================================
class TestHRVProcessing:
    """Test HRV processing with real RRI data."""

    @pytest.fixture
    def rri_data(self):
        from biosppy import storage
        signal, _ = storage.load_txt(data_file('rri.txt'))
        return signal

    def test_hrv_from_rri(self, rri_data):
        """Test HRV analysis from RR intervals."""
        from biosppy.signals import hrv
        # RRI is in ms, convert to seconds for some functions
        rri_sec = rri_data / 1000.0
        # Test time-domain HRV if available
        if hasattr(hrv, 'time_domain'):
            result = hrv.time_domain(rri=rri_sec)
            print(f"  HRV time-domain: computed OK")
        else:
            print(f"  HRV: time_domain not available, module imported OK")

    def test_hrv_from_ecg_rpeaks(self):
        """Test HRV from ECG R-peaks detection."""
        from biosppy import storage
        from biosppy.signals import ecg
        signal, _ = storage.load_txt(data_file('ecg.txt'))
        ecg_result = ecg.ecg(signal=signal, sampling_rate=1000., show=False)
        rpeaks = ecg_result['rpeaks']
        rri = np.diff(rpeaks) / 1000.0  # Convert to seconds
        assert len(rri) > 0
        print(f"  HRV from ECG: {len(rri)} RR intervals, "
              f"mean={rri.mean()*1000:.0f}ms, std={rri.std()*1000:.1f}ms")


# ============================================================================
# Cross-module Integration
# ============================================================================
class TestCrossModuleIntegration:
    """Test interactions between modules."""

    def test_synthesizers_still_accessible(self):
        """BUG 1: synthesizers should still be accessible via biosppy.synthesizers."""
        import biosppy
        assert hasattr(biosppy.synthesizers, 'ecg')
        assert hasattr(biosppy.synthesizers, 'emg')

    def test_features_accessible(self):
        """BUG 2: All features including wavelet_coherence should be accessible."""
        import biosppy
        assert hasattr(biosppy, 'wavelet_coherence')
        assert hasattr(biosppy, 'frequency')
        assert hasattr(biosppy, 'time')
        assert hasattr(biosppy, 'time_freq')

    def test_version_is_correct(self):
        """BUG 5: Version should be consistent."""
        import biosppy
        assert biosppy.__version__ == '2.2.3'

    def test_all_signal_modules_importable(self):
        """BUG 4: All signal submodules should import without error."""
        from biosppy.signals import (acc, abp, baroreflex, bvp, ecg, eda,
                                      eeg, emd, emg, hrv, multichannel,
                                      pcg, ppg, resp, tools)
        # All imported successfully
        assert all(m is not None for m in [acc, abp, baroreflex, bvp, ecg, eda,
                                            eeg, emd, emg, hrv, multichannel,
                                            pcg, ppg, resp, tools])

    def test_ecg_then_hrv_pipeline(self):
        """Full pipeline: load ECG -> detect R-peaks -> compute HRV."""
        from biosppy import storage
        from biosppy.signals import ecg
        signal, _ = storage.load_txt(data_file('ecg.txt'))
        result = ecg.ecg(signal=signal, sampling_rate=1000., show=False)

        rpeaks = result['rpeaks']
        rri_ms = np.diff(rpeaks)  # in samples at 1000 Hz = ms

        assert len(rri_ms) > 5, f"Expected multiple RR intervals, got {len(rri_ms)}"
        mean_hr = 60000.0 / rri_ms.mean()
        assert 40 < mean_hr < 200, f"Mean HR {mean_hr:.0f} bpm out of physiological range"
        print(f"  ECG->HRV pipeline: {len(rri_ms)} RRI, mean HR={mean_hr:.0f} bpm")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '--tb=long'])
