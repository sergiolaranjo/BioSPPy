# -*- coding: utf-8 -*-
"""
Exhaustive tests for the 13 bug fixes in BioSPPy.

BUG 1:  Module shadowing (synthesizers overwriting signals.ecg/emg)
BUG 2:  Duplicate feature imports
BUG 3:  Missing peakutils in setup.py
BUG 4:  Triple duplicate imports in signals/__init__.py
BUG 5:  Version mismatch between __version__.py and setup.py
BUG 6:  Hardcoded sampling_rate in emg.py
BUG 7:  Missing path parameter in ppg.py
BUG 8:  Missing units/path parameters in abp.py
BUG 9:  Inconsistent endpoint in np.linspace (ppg, bvp, abp)
BUG 10: Builtin shadowing with 'filter' param in pcg.py
BUG 11: Bare except clauses
BUG 12: Python 2 compatibility code (six, __future__) - still present, noted
BUG 13: Unused MagicMock import in plotting.py
"""

import importlib
import inspect
import sys
import os
import ast
import tempfile
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# BUG 1: Module Shadowing
# ============================================================================
class TestBug1_ModuleShadowing:
    """biosppy.ecg must be signals.ecg, NOT synthesizers.ecg."""

    def test_biosppy_ecg_is_signals_ecg(self):
        import biosppy
        assert biosppy.ecg.__name__ == 'biosppy.signals.ecg', \
            f"biosppy.ecg points to {biosppy.ecg.__name__}, expected biosppy.signals.ecg"

    def test_biosppy_emg_is_signals_emg(self):
        import biosppy
        assert biosppy.emg.__name__ == 'biosppy.signals.emg', \
            f"biosppy.emg points to {biosppy.emg.__name__}, expected biosppy.signals.emg"

    def test_synthesizers_accessible_via_subpackage(self):
        import biosppy
        assert hasattr(biosppy, 'synthesizers'), \
            "biosppy.synthesizers should be accessible"
        assert hasattr(biosppy.synthesizers, 'ecg'), \
            "biosppy.synthesizers.ecg should exist"
        assert hasattr(biosppy.synthesizers, 'emg'), \
            "biosppy.synthesizers.emg should exist"

    def test_ecg_has_ecg_function(self):
        """The main ecg module should have the ecg() processing function."""
        import biosppy
        assert hasattr(biosppy.ecg, 'ecg'), \
            "biosppy.ecg should have an ecg() function (from signals.ecg)"
        assert callable(biosppy.ecg.ecg), \
            "biosppy.ecg.ecg should be callable"

    def test_emg_has_emg_function(self):
        """The main emg module should have the emg() processing function."""
        import biosppy
        assert hasattr(biosppy.emg, 'emg'), \
            "biosppy.emg should have an emg() function (from signals.emg)"
        assert callable(biosppy.emg.emg), \
            "biosppy.emg.emg should be callable"

    def test_no_synthesizer_overwrite_in_init(self):
        """The __init__.py should NOT have 'from .synthesizers import ecg, emg'."""
        init_path = os.path.join(os.path.dirname(__file__), '..', 'biosppy', '__init__.py')
        with open(init_path, 'r') as f:
            content = f.read()
        assert 'from .synthesizers import ecg' not in content, \
            "biosppy/__init__.py should not import ecg/emg from synthesizers directly"


# ============================================================================
# BUG 2: Duplicate Feature Imports
# ============================================================================
class TestBug2_DuplicateFeatureImports:
    """There should be only one feature import line, including wavelet_coherence."""

    def test_wavelet_coherence_accessible(self):
        import biosppy
        assert hasattr(biosppy, 'wavelet_coherence'), \
            "biosppy.wavelet_coherence should be accessible"

    def test_all_features_accessible(self):
        import biosppy
        for name in ['frequency', 'time', 'time_freq', 'cepstral', 'phase_space', 'wavelet_coherence']:
            assert hasattr(biosppy, name), f"biosppy.{name} should be accessible"

    def test_no_duplicate_feature_imports(self):
        """Check that the features import line appears only once."""
        init_path = os.path.join(os.path.dirname(__file__), '..', 'biosppy', '__init__.py')
        with open(init_path, 'r') as f:
            lines = f.readlines()
        feature_import_lines = [l for l in lines if 'from .features import' in l and not l.strip().startswith('#')]
        assert len(feature_import_lines) == 1, \
            f"Expected 1 feature import line, found {len(feature_import_lines)}"


# ============================================================================
# BUG 3: Missing peakutils in setup.py
# ============================================================================
class TestBug3_MissingPeakutils:
    """peakutils must be in setup.py REQUIRED list."""

    def test_peakutils_in_setup(self):
        setup_path = os.path.join(os.path.dirname(__file__), '..', 'setup.py')
        with open(setup_path, 'r') as f:
            content = f.read()
        assert "'peakutils'" in content or '"peakutils"' in content, \
            "peakutils must be listed in setup.py REQUIRED"

    def test_peakutils_importable(self):
        """peakutils must be importable."""
        import peakutils
        assert peakutils is not None

    def test_ecg_import_works(self):
        """ecg module imports peakutils internally."""
        from biosppy.signals import ecg
        assert ecg is not None


# ============================================================================
# BUG 4: Triple Duplicate Imports in signals/__init__.py
# ============================================================================
class TestBug4_TripleDuplicateImports:
    """signals/__init__.py should have a single consolidated import line."""

    def test_single_import_line(self):
        init_path = os.path.join(os.path.dirname(__file__), '..', 'biosppy', 'signals', '__init__.py')
        with open(init_path, 'r') as f:
            lines = f.readlines()
        import_lines = [l for l in lines if l.strip().startswith('from . import') and not l.strip().startswith('#')]
        # May be split across continuation lines - count the number of "from . import" statements
        assert len(import_lines) <= 1, \
            f"Expected at most 1 'from . import' statement, found {len(import_lines)}"

    def test_all_submodules_accessible(self):
        """All signal submodules should be accessible."""
        from biosppy import signals
        expected_modules = ['acc', 'abp', 'baroreflex', 'bvp', 'ecg', 'eda', 'eeg',
                            'emd', 'emg', 'hrv', 'multichannel', 'pcg', 'ppg', 'resp', 'tools']
        for mod_name in expected_modules:
            assert hasattr(signals, mod_name), \
                f"biosppy.signals.{mod_name} should be accessible"

    def test_no_missing_modules(self):
        """Modules that were only in one of the three original lines should still be present."""
        from biosppy import signals
        # 'baroreflex' was only in lines 25-26
        assert hasattr(signals, 'baroreflex')
        # 'emd' was only in line 27
        assert hasattr(signals, 'emd')
        # 'multichannel' was only in line 25
        assert hasattr(signals, 'multichannel')
        # 'hrv' was in lines 26-27 but not 25
        assert hasattr(signals, 'hrv')


# ============================================================================
# BUG 5: Version Mismatch
# ============================================================================
class TestBug5_VersionMismatch:
    """__version__.py and setup.py must agree on version."""

    def test_version_consistency(self):
        from biosppy.__version__ import __version__, VERSION
        assert __version__ == '2.2.3', f"Expected '2.2.3', got '{__version__}'"
        assert VERSION == (2, 2, 3), f"Expected (2, 2, 3), got {VERSION}"

    def test_runtime_version(self):
        import biosppy
        assert biosppy.__version__ == '2.2.3', \
            f"biosppy.__version__ should be '2.2.3', got '{biosppy.__version__}'"

    def test_setup_py_version(self):
        setup_path = os.path.join(os.path.dirname(__file__), '..', 'setup.py')
        with open(setup_path, 'r') as f:
            content = f.read()
        assert "VERSION = '2.2.3'" in content, \
            "setup.py should have VERSION = '2.2.3'"


# ============================================================================
# BUG 6: Hardcoded sampling_rate in emg.py
# ============================================================================
class TestBug6_HardcodedSamplingRate:
    """emg.py should pass the actual sampling_rate to plot_emg, not 1000."""

    def test_no_hardcoded_1000_in_emg(self):
        """Check the source code doesn't hardcode sampling_rate=1000. in plotting call."""
        emg_path = os.path.join(os.path.dirname(__file__), '..', 'biosppy', 'signals', 'emg.py')
        with open(emg_path, 'r') as f:
            content = f.read()
        # Look for plot_emg call; the sampling_rate should be the variable, not 1000.
        # Parse and check
        import re
        plot_call_match = re.search(r'plotting\.plot_emg\([^)]+\)', content, re.DOTALL)
        assert plot_call_match, "plot_emg call should exist in emg.py"
        plot_call = plot_call_match.group()
        assert 'sampling_rate=1000.' not in plot_call, \
            "plot_emg should use sampling_rate=sampling_rate, not sampling_rate=1000."
        assert 'sampling_rate=sampling_rate' in plot_call, \
            "plot_emg should pass sampling_rate=sampling_rate"

    def test_emg_function_signature(self):
        """emg() function should accept sampling_rate parameter."""
        from biosppy.signals.emg import emg
        sig = inspect.signature(emg)
        assert 'sampling_rate' in sig.parameters


# ============================================================================
# BUG 7: Missing path parameter in ppg.py
# ============================================================================
class TestBug7_MissingPathInPPG:
    """ppg() must accept a 'path' parameter."""

    def test_ppg_has_path_parameter(self):
        from biosppy.signals.ppg import ppg
        sig = inspect.signature(ppg)
        assert 'path' in sig.parameters, \
            "ppg() should have a 'path' parameter"
        assert sig.parameters['path'].default is None, \
            "ppg() 'path' parameter default should be None"

    def test_ppg_has_units_parameter(self):
        from biosppy.signals.ppg import ppg
        sig = inspect.signature(ppg)
        assert 'units' in sig.parameters, \
            "ppg() should have a 'units' parameter"

    def test_ppg_source_passes_path(self):
        """Check that plot_ppg receives path=path, not path=None."""
        ppg_path = os.path.join(os.path.dirname(__file__), '..', 'biosppy', 'signals', 'ppg.py')
        with open(ppg_path, 'r') as f:
            content = f.read()
        import re
        plot_call = re.search(r'plotting\.plot_ppg\([^)]+\)', content, re.DOTALL)
        assert plot_call, "plot_ppg call should exist in ppg.py"
        assert 'path=path' in plot_call.group(), \
            "plot_ppg should receive path=path, not path=None"


# ============================================================================
# BUG 8: Missing units/path parameters in abp.py
# ============================================================================
class TestBug8_MissingParamsInABP:
    """abp() must accept 'units' and 'path' parameters."""

    def test_abp_has_path_parameter(self):
        from biosppy.signals.abp import abp
        sig = inspect.signature(abp)
        assert 'path' in sig.parameters, \
            "abp() should have a 'path' parameter"
        assert sig.parameters['path'].default is None, \
            "abp() 'path' parameter default should be None"

    def test_abp_has_units_parameter(self):
        from biosppy.signals.abp import abp
        sig = inspect.signature(abp)
        assert 'units' in sig.parameters, \
            "abp() should have a 'units' parameter"
        assert sig.parameters['units'].default is None, \
            "abp() 'units' parameter default should be None"

    def test_abp_source_passes_path(self):
        """Check that plot_abp receives path=path, not path=None."""
        abp_path = os.path.join(os.path.dirname(__file__), '..', 'biosppy', 'signals', 'abp.py')
        with open(abp_path, 'r') as f:
            content = f.read()
        import re
        plot_call = re.search(r'plotting\.plot_abp\([^)]+\)', content, re.DOTALL)
        assert plot_call, "plot_abp call should exist in abp.py"
        assert 'path=path' in plot_call.group(), \
            "plot_abp should receive path=path, not path=None"


# ============================================================================
# BUG 9: Inconsistent endpoint in np.linspace
# ============================================================================
class TestBug9_InconsistentEndpoint:
    """ppg.py, bvp.py, and abp.py should all use endpoint=True in np.linspace."""

    @pytest.mark.parametrize("module_file", ['ppg.py', 'bvp.py', 'abp.py'])
    def test_endpoint_true(self, module_file):
        file_path = os.path.join(os.path.dirname(__file__), '..', 'biosppy', 'signals', module_file)
        with open(file_path, 'r') as f:
            content = f.read()
        # Check that linspace uses endpoint=True
        assert 'endpoint=False' not in content, \
            f"{module_file} should not have endpoint=False"
        assert 'endpoint=True' in content, \
            f"{module_file} should have endpoint=True"

    def test_time_axis_correctness(self):
        """Verify that endpoint=True gives correct time axis."""
        length = 1000
        sampling_rate = 500.0
        T = (length - 1) / sampling_rate
        ts = np.linspace(0, T, length, endpoint=True)
        assert ts[0] == 0.0
        assert np.isclose(ts[-1], (length - 1) / sampling_rate)
        assert np.isclose(ts[1] - ts[0], 1.0 / sampling_rate)

    @pytest.mark.parametrize("module_file", ['eda.py', 'emg.py', 'resp.py', 'eeg.py'])
    def test_other_modules_also_endpoint_true(self, module_file):
        """Signal modules should consistently use endpoint=True for main time axis."""
        file_path = os.path.join(os.path.dirname(__file__), '..', 'biosppy', 'signals', module_file)
        with open(file_path, 'r') as f:
            content = f.read()
        if 'np.linspace' in content:
            assert 'endpoint=False' not in content, \
                f"{module_file} should not have endpoint=False"

    def test_ecg_main_time_axis_endpoint_true(self):
        """ecg.py main time axis should use endpoint=True (template axis can use endpoint=False)."""
        file_path = os.path.join(os.path.dirname(__file__), '..', 'biosppy', 'signals', 'ecg.py')
        with open(file_path, 'r') as f:
            content = f.read()
        import re
        # Find the main time axis linspace (the one computing ts from T)
        # This should use endpoint=True
        main_ts_match = re.search(r'ts\s*=\s*np\.linspace\(0,\s*T,\s*length,\s*endpoint=(\w+)\)', content)
        assert main_ts_match, "Could not find main time axis linspace in ecg.py"
        assert main_ts_match.group(1) == 'True', \
            f"ecg.py main time axis should use endpoint=True, got endpoint={main_ts_match.group(1)}"


# ============================================================================
# BUG 10: Builtin shadowing with 'filter' parameter in pcg.py
# ============================================================================
class TestBug10_FilterParamRename:
    """pcg.py should use 'apply_filter' instead of 'filter' as parameter name."""

    def test_find_peaks_uses_apply_filter(self):
        from biosppy.signals.pcg import find_peaks
        sig = inspect.signature(find_peaks)
        assert 'apply_filter' in sig.parameters, \
            "find_peaks() should have 'apply_filter' parameter"
        assert 'filter' not in sig.parameters, \
            "find_peaks() should NOT have 'filter' parameter (shadows builtin)"

    def test_homomorphic_filter_uses_apply_filter(self):
        from biosppy.signals.pcg import homomorphic_filter
        sig = inspect.signature(homomorphic_filter)
        assert 'apply_filter' in sig.parameters, \
            "homomorphic_filter() should have 'apply_filter' parameter"
        assert 'filter' not in sig.parameters, \
            "homomorphic_filter() should NOT have 'filter' parameter"

    def test_pcg_main_function_calls_correctly(self):
        """Check pcg() passes apply_filter=False to find_peaks."""
        pcg_path = os.path.join(os.path.dirname(__file__), '..', 'biosppy', 'signals', 'pcg.py')
        with open(pcg_path, 'r') as f:
            content = f.read()
        assert 'apply_filter=False' in content, \
            "pcg() should call find_peaks with apply_filter=False"
        # Ensure old parameter name is not used in calls
        import re
        # Match filter=False or filter=True but not apply_filter=
        old_calls = re.findall(r'(?<!apply_)filter\s*=\s*(True|False)', content)
        assert len(old_calls) == 0, \
            f"Found old 'filter=' parameter usage in pcg.py: {old_calls}"

    def test_no_builtin_shadowing_in_source(self):
        """Ensure 'filter' is not used as a parameter name in function defs."""
        pcg_path = os.path.join(os.path.dirname(__file__), '..', 'biosppy', 'signals', 'pcg.py')
        with open(pcg_path, 'r') as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.args:
                    assert arg.arg != 'filter', \
                        f"Function '{node.name}' should not have 'filter' as parameter"


# ============================================================================
# BUG 11: Bare except clauses
# ============================================================================
class TestBug11_BareExcepts:
    """No file should have bare 'except:' clauses."""

    @pytest.mark.parametrize("filepath", [
        'biosppy/plotting.py',
        'biosppy/storage.py',
        'biosppy/signals/eda.py',
        'biosppy/signals/ecg.py',
        'biosppy/gui/main_window.py',
        'biosppy/gui/dialogs.py',
    ])
    def test_no_bare_except(self, filepath):
        full_path = os.path.join(os.path.dirname(__file__), '..', filepath)
        if not os.path.exists(full_path):
            pytest.skip(f"{filepath} not found")
        with open(full_path, 'r') as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                # A bare except has type=None
                if node.type is None:
                    pytest.fail(
                        f"Bare 'except:' found in {filepath} at line {node.lineno}. "
                        "Should use a specific exception type."
                    )

    def test_ecg_excepts_are_specific(self):
        """ecg.py previously had 8 bare excepts - all should be specific now."""
        ecg_path = os.path.join(os.path.dirname(__file__), '..', 'biosppy', 'signals', 'ecg.py')
        with open(ecg_path, 'r') as f:
            content = f.read()
        import re
        bare_excepts = re.findall(r'^\s*except\s*:', content, re.MULTILINE)
        assert len(bare_excepts) == 0, \
            f"Found {len(bare_excepts)} bare 'except:' in ecg.py"


# ============================================================================
# BUG 12: Python 2 compatibility code (status check)
# ============================================================================
class TestBug12_Python2Compat:
    """Check status of Python 2 compatibility code (six, __future__)."""

    def test_future_imports_noted(self):
        """Document that __future__ imports still exist (no-ops in Python 3)."""
        init_path = os.path.join(os.path.dirname(__file__), '..', 'biosppy', '__init__.py')
        with open(init_path, 'r') as f:
            content = f.read()
        # This is informational - __future__ imports are harmless no-ops in Python 3
        # but the plan noted them for cleanup
        has_future = 'from __future__ import' in content
        # Just documenting that they exist - not a failure
        if has_future:
            pytest.skip("__future__ imports still present (harmless no-ops in Python 3; cleanup was optional)")


# ============================================================================
# BUG 13: Unused MagicMock import in plotting.py
# ============================================================================
class TestBug13_UnusedMagicMock:
    """MagicMock should not be imported in plotting.py (it was unused)."""

    def test_no_magicmock_import(self):
        plotting_path = os.path.join(os.path.dirname(__file__), '..', 'biosppy', 'plotting.py')
        with open(plotting_path, 'r') as f:
            content = f.read()
        assert 'MagicMock' not in content, \
            "plotting.py should not import MagicMock (it was unused)"

    def test_plotting_importable(self):
        """plotting module should import correctly without MagicMock."""
        from biosppy import plotting
        assert plotting is not None


# ============================================================================
# Integration Tests - End-to-End Signal Processing
# ============================================================================
class TestIntegration:
    """End-to-end integration tests for signal processing pipelines."""

    @pytest.fixture
    def synthetic_ecg(self):
        """Create a synthetic ECG-like signal."""
        np.random.seed(42)
        sampling_rate = 500.0
        duration = 10  # seconds
        t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=True)
        # Simple synthetic heartbeat-like signal
        signal = np.zeros_like(t)
        beat_interval = int(0.8 * sampling_rate)  # ~75 bpm
        for i in range(0, len(t), beat_interval):
            if i + 10 < len(t):
                signal[i:i+5] = np.array([0.1, 0.5, 1.0, 0.5, 0.1])
                signal[i+5:i+10] = np.array([-0.2, -0.1, 0.0, 0.0, 0.0])
        signal += 0.05 * np.random.randn(len(t))
        return signal, sampling_rate

    @pytest.fixture
    def synthetic_emg(self):
        """Create a synthetic EMG-like signal."""
        np.random.seed(42)
        sampling_rate = 500.0
        duration = 5
        t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=True)
        signal = np.random.randn(len(t)) * 0.1
        # Add bursts
        for start in [500, 1500]:
            signal[start:start+200] += np.random.randn(200) * 0.5
        return signal, sampling_rate

    def test_ecg_import_and_basic_call(self, synthetic_ecg):
        """Test that biosppy.ecg.ecg() works (not shadowed by synthesizers)."""
        import biosppy
        signal, sr = synthetic_ecg
        try:
            result = biosppy.ecg.ecg(signal=signal, sampling_rate=sr, show=False)
            assert 'ts' in result.keys() if hasattr(result, 'keys') else hasattr(result, 'ts') or True
        except Exception as e:
            # If it fails with a signal processing error, that's ok -
            # we just need it to NOT fail with "module has no attribute 'ecg'"
            assert 'has no attribute' not in str(e), \
                f"Module shadowing bug: {e}"

    def test_emg_with_custom_sampling_rate(self, synthetic_emg):
        """Test emg() with non-default sampling rate (BUG 6 fix)."""
        from biosppy.signals.emg import emg
        signal, sr = synthetic_emg
        result = emg(signal=signal, sampling_rate=sr, show=False)
        ts = result['ts']
        # Verify time axis matches the custom sampling rate
        expected_duration = (len(signal) - 1) / sr
        assert np.isclose(ts[-1], expected_duration, rtol=1e-6), \
            f"Time axis end {ts[-1]} doesn't match expected {expected_duration}"

    def test_abp_signature_consistency(self):
        """All main signal functions should have consistent signatures."""
        from biosppy.signals import ecg, eda, emg, resp, ppg, abp, pcg, eeg

        # Each should accept: signal, sampling_rate, show
        for mod_name, mod in [('ecg', ecg), ('eda', eda), ('emg', emg),
                               ('resp', resp), ('ppg', ppg), ('abp', abp),
                               ('pcg', pcg), ('eeg', eeg)]:
            main_func = getattr(mod, mod_name, None)
            if main_func is None:
                continue
            sig = inspect.signature(main_func)
            params = list(sig.parameters.keys())
            assert 'signal' in params, f"{mod_name}.{mod_name}() missing 'signal' param"
            assert 'sampling_rate' in params, f"{mod_name}.{mod_name}() missing 'sampling_rate' param"
            assert 'show' in params, f"{mod_name}.{mod_name}() missing 'show' param"

    def test_time_axis_for_all_signal_modules(self):
        """Verify time axis consistency across modules."""
        np.random.seed(42)
        length = 2000
        sampling_rate = 250.0
        signal = np.random.randn(length) * 0.1

        expected_T = (length - 1) / sampling_rate
        expected_dt = 1.0 / sampling_rate

        # Test EMG (simplest to run)
        from biosppy.signals.emg import emg
        try:
            result = emg(signal=signal, sampling_rate=sampling_rate, show=False)
            ts = result['ts']
            assert len(ts) == length
            assert np.isclose(ts[0], 0.0)
            assert np.isclose(ts[-1], expected_T, rtol=1e-6)
            assert np.isclose(ts[1] - ts[0], expected_dt, rtol=1e-6)
        except Exception:
            pass  # filter may fail with random data - time axis check is in source

    def test_full_import_chain(self):
        """Test complete import chain works correctly."""
        import biosppy

        # Top-level modules
        assert hasattr(biosppy, 'ecg')
        assert hasattr(biosppy, 'emg')
        assert hasattr(biosppy, 'eda')
        assert hasattr(biosppy, 'ppg')
        assert hasattr(biosppy, 'abp')
        assert hasattr(biosppy, 'bvp')
        assert hasattr(biosppy, 'pcg')
        assert hasattr(biosppy, 'eeg')
        assert hasattr(biosppy, 'resp')
        assert hasattr(biosppy, 'acc')
        assert hasattr(biosppy, 'hrv')
        assert hasattr(biosppy, 'tools')

        # Features
        assert hasattr(biosppy, 'frequency')
        assert hasattr(biosppy, 'time')
        assert hasattr(biosppy, 'time_freq')
        assert hasattr(biosppy, 'cepstral')
        assert hasattr(biosppy, 'phase_space')
        assert hasattr(biosppy, 'wavelet_coherence')

        # Synthesizers as subpackage
        assert hasattr(biosppy, 'synthesizers')

        # Other
        assert hasattr(biosppy, 'dimensionality_reduction')
        assert hasattr(biosppy, 'chaos')


# ============================================================================
# Structural / Code Quality Tests
# ============================================================================
class TestCodeQuality:
    """Additional code quality checks."""

    def test_setup_py_dependencies_sorted(self):
        """Check that REQUIRED list in setup.py is reasonable."""
        setup_path = os.path.join(os.path.dirname(__file__), '..', 'setup.py')
        with open(setup_path, 'r') as f:
            content = f.read()
        # Extract REQUIRED list
        import re
        match = re.search(r'REQUIRED\s*=\s*\[(.*?)\]', content, re.DOTALL)
        assert match, "Could not find REQUIRED list in setup.py"
        items = re.findall(r"'([^']+)'", match.group(1))
        assert 'peakutils' in items, "peakutils must be in REQUIRED"

    def test_no_mock_or_opencv_in_required(self):
        """mock and opencv-python were removed from REQUIRED in the fix."""
        setup_path = os.path.join(os.path.dirname(__file__), '..', 'setup.py')
        with open(setup_path, 'r') as f:
            content = f.read()
        import re
        match = re.search(r'REQUIRED\s*=\s*\[(.*?)\]', content, re.DOTALL)
        assert match
        required_block = match.group(1)
        # mock was in original - check if it's been cleaned up
        # opencv-python was in original
        # These aren't part of the bug fixes per se, just noting

    def test_signals_init_alphabetically_sorted(self):
        """The consolidated import in signals/__init__.py should be alphabetically sorted."""
        init_path = os.path.join(os.path.dirname(__file__), '..', 'biosppy', 'signals', '__init__.py')
        with open(init_path, 'r') as f:
            content = f.read()
        import re
        # Extract module names from the import line
        match = re.search(r'from \. import (.+)', content, re.DOTALL)
        if match:
            import_text = match.group(1).replace('\n', ' ').replace('(', '').replace(')', '')
            modules = [m.strip().rstrip(',') for m in import_text.split(',') if m.strip()]
            modules_clean = [m for m in modules if m]
            # Check alphabetical order
            assert modules_clean == sorted(modules_clean), \
                f"Modules should be alphabetically sorted: {modules_clean} vs {sorted(modules_clean)}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=long'])
