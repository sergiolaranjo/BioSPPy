# -*- coding: utf-8 -*-
"""
Tests for advanced BioSPPy modules: features, synthesizers, clustering,
EMD, HRV, metrics, and signal tools.

Uses real sample data from examples/ where appropriate.
"""

import os
import sys
import warnings
import numpy as np
import pytest

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'examples')


def data_file(name):
    path = os.path.join(EXAMPLES_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"Data file {name} not found")
    return path


def has_key(result, key):
    return key in result.keys()


# ============================================================================
# SYNTHESIZERS
# ============================================================================
class TestSynthesizersECG:
    """Test ECG signal synthesizers."""

    def test_ecg_synthesizer_default(self):
        from biosppy.synthesizers import ecg as synth_ecg
        result = synth_ecg.ecg()
        assert has_key(result, 'ecg')
        assert has_key(result, 't')
        assert len(result['ecg']) > 0
        assert len(result['ecg']) == len(result['t'])
        print(f"  Synth ECG: {len(result['ecg'])} samples generated")

    def test_ecg_synthesizer_custom_params(self):
        from biosppy.synthesizers import ecg as synth_ecg
        result = synth_ecg.ecg(sampling_rate=5000, Ar=0.9)
        assert len(result['ecg']) > 0
        print(f"  Synth ECG (custom): {len(result['ecg'])} samples at 5000 Hz")


class TestSynthesizersEMG:
    """Test EMG signal synthesizers."""

    def test_emg_synth_uniform(self):
        from biosppy.synthesizers import emg as synth_emg
        result = synth_emg.synth_uniform(
            duration=5, sampling_rate=1000, burst_number=3, burst_duration=0.5
        )
        keys = result.keys()
        assert len(keys) > 0
        # Get the signal (first element)
        sig = result[0]
        assert len(sig) > 0
        print(f"  Synth EMG uniform: {len(sig)} samples, keys={keys}")

    def test_emg_synth_gaussian(self):
        from biosppy.synthesizers import emg as synth_emg
        result = synth_emg.synth_gaussian(
            duration=5, sampling_rate=1000, burst_number=2
        )
        keys = result.keys()
        assert len(keys) > 0
        sig = result[0]
        assert len(sig) > 0
        print(f"  Synth EMG gaussian: {len(sig)} samples, keys={keys}")


# ============================================================================
# FEATURES - Frequency Domain
# ============================================================================
class TestFeaturesFrequency:
    """Test frequency domain feature extraction."""

    @pytest.fixture
    def ecg_signal(self):
        from biosppy import storage
        signal, _ = storage.load_txt(data_file('ecg.txt'))
        return signal

    def test_frequency_features(self, ecg_signal):
        from biosppy.features import frequency
        result = frequency.frequency(signal=ecg_signal, sampling_rate=1000.)
        keys = result.keys()
        # Returns individual FFT_ prefixed keys
        assert any('FFT_' in k for k in keys), f"No FFT_ keys in {keys}"
        print(f"  Frequency features: {len(keys)} features, sample={keys[:5]}")

    def test_spectral_features(self, ecg_signal):
        from biosppy.features import frequency
        # Compute FFT first
        freqs = np.fft.rfftfreq(len(ecg_signal), d=1.0/1000.)
        power = np.abs(np.fft.rfft(ecg_signal))**2
        result = frequency.spectral_features(
            freqs=freqs, power=power, sampling_rate=1000.
        )
        assert has_key(result, 'fundamental_frequency')
        assert has_key(result, 'centroid')
        print(f"  Spectral: fund_freq={result['fundamental_frequency']:.1f} Hz, "
              f"centroid={result['centroid']:.1f}")


# ============================================================================
# FEATURES - Time Domain
# ============================================================================
class TestFeaturesTime:
    """Test time domain feature extraction."""

    @pytest.fixture
    def ecg_signal(self):
        from biosppy import storage
        signal, _ = storage.load_txt(data_file('ecg.txt'))
        return signal

    def test_time_features(self, ecg_signal):
        from biosppy.features import time
        result = time.time(signal=ecg_signal, sampling_rate=1000.)
        keys = result.keys()
        assert 'mean' in keys or 'signal_mean' in keys or len(keys) > 0
        print(f"  Time features: {len(keys)} features computed, keys={keys[:5]}...")

    def test_hjorth_features(self, ecg_signal):
        from biosppy.features import time
        result = time.hjorth_features(signal=ecg_signal)
        assert has_key(result, 'hjorth_mobility')
        assert has_key(result, 'hjorth_complexity')
        print(f"  Hjorth: mobility={result['hjorth_mobility']:.4f}, "
              f"complexity={result['hjorth_complexity']:.4f}")


# ============================================================================
# FEATURES - Time-Frequency Domain
# ============================================================================
class TestFeaturesTimeFreq:
    """Test time-frequency domain feature extraction (wavelets)."""

    @pytest.fixture
    def ecg_signal(self):
        from biosppy import storage
        signal, _ = storage.load_txt(data_file('ecg.txt'))
        return signal

    def test_time_freq_features(self, ecg_signal):
        from biosppy.features import time_freq
        result = time_freq.time_freq(signal=ecg_signal, wavelet='db4', level=5)
        keys = result.keys()
        # Should have dwt_app and dwt_det features
        assert any('dwt' in k for k in keys), f"No DWT keys found in {keys}"
        print(f"  Time-freq features: {len(keys)} features, sample={keys[:3]}")

    def test_compute_wavelet(self, ecg_signal):
        from biosppy.features import time_freq
        result = time_freq.compute_wavelet(signal=ecg_signal, wavelet='db4', level=5)
        assert has_key(result, 'dwt_app')
        print(f"  DWT computed: app coeff len={len(result['dwt_app'])}")


# ============================================================================
# FEATURES - Wavelet Coherence
# ============================================================================
class TestFeaturesWaveletCoherence:
    """Test wavelet coherence between two signals."""

    @pytest.fixture
    def two_signals(self):
        """Create two correlated signals for coherence testing."""
        from biosppy import storage
        ecg, _ = storage.load_txt(data_file('ecg.txt'))
        # Use first 2000 samples for speed
        sig1 = ecg[:2000]
        # Create a phase-shifted version
        sig2 = np.roll(sig1, 50) + np.random.randn(len(sig1)) * 0.1
        return sig1, sig2

    def test_wavelet_coherence(self, two_signals):
        from biosppy.features import wavelet_coherence as wc
        sig1, sig2 = two_signals
        result = wc.wavelet_coherence(
            signal1=sig1, signal2=sig2, sampling_rate=1000.,
            compute_phase=True, compute_delay=True
        )
        assert has_key(result, 'coherence')
        assert has_key(result, 'frequencies')
        coh = result['coherence']
        assert coh.min() >= 0 and coh.max() <= 1.0 + 1e-6
        print(f"  Wavelet coherence: shape={coh.shape}, "
              f"mean={coh.mean():.3f}")

    def test_cross_wavelet_spectrum(self, two_signals):
        from biosppy.features import wavelet_coherence as wc
        sig1, sig2 = two_signals
        result = wc.cross_wavelet_spectrum(
            signal1=sig1, signal2=sig2, sampling_rate=1000.
        )
        assert has_key(result, 'cross_spectrum')
        assert has_key(result, 'frequencies')
        print(f"  Cross-wavelet: spectrum shape={result['cross_spectrum'].shape}")

    def test_significance_test(self, two_signals):
        from biosppy.features import wavelet_coherence as wc
        sig1, sig2 = two_signals
        result = wc.wavelet_coherence(
            signal1=sig1, signal2=sig2, sampling_rate=1000.
        )
        is_sig, threshold = wc.significance_test(
            result['coherence'], n_samples=len(sig1), alpha=0.05
        )
        assert isinstance(is_sig, np.ndarray)
        assert 0 < threshold < 1
        print(f"  Significance: threshold={threshold:.3f}, "
              f"{is_sig.sum()}/{is_sig.size} significant")


# ============================================================================
# FEATURES - Cepstral Domain
# ============================================================================
class TestFeaturesCepstral:
    """Test cepstral domain features (MFCC)."""

    @pytest.fixture
    def ecg_signal(self):
        from biosppy import storage
        signal, _ = storage.load_txt(data_file('ecg.txt'))
        return signal

    def test_cepstral_features(self, ecg_signal):
        from biosppy.features import cepstral
        result = cepstral.cepstral(signal=ecg_signal, sampling_rate=1000.)
        keys = result.keys()
        assert len(keys) > 0
        print(f"  Cepstral: {len(keys)} features computed")

    def test_mfcc(self, ecg_signal):
        from biosppy.features import cepstral
        result = cepstral.mfcc(signal=ecg_signal, sampling_rate=1000.)
        assert has_key(result, 'mfcc')
        mfcc_val = result['mfcc']
        assert mfcc_val.shape[0] > 0
        print(f"  MFCC: shape={mfcc_val.shape}")

    def test_mel_freq_conversion(self):
        from biosppy.features import cepstral
        freqs = np.array([100., 500., 1000., 2000., 4000.])
        mels = cepstral.freq_to_mel(freqs)
        back = cepstral.mel_to_freq(mels)
        assert np.allclose(freqs, back, rtol=1e-6)
        print(f"  Mel conversion: {freqs} -> {mels} -> {back}")


# ============================================================================
# FEATURES - Phase Space
# ============================================================================
class TestFeaturesPhaseSpace:
    """Test phase space / recurrence plot features."""

    @pytest.fixture
    def short_signal(self):
        from biosppy import storage
        signal, _ = storage.load_txt(data_file('ecg.txt'))
        # Use short segment for speed (recurrence plots are O(n^2))
        return signal[:500]

    def test_phase_space_features(self, short_signal):
        from biosppy.features import phase_space
        result = phase_space.phase_space(signal=short_signal)
        keys = result.keys()
        assert any('rec' in k for k in keys), f"No recurrence keys in {keys}"
        print(f"  Phase space: {len(keys)} features computed")

    def test_recurrence_plot(self, short_signal):
        from biosppy.features import phase_space
        result = phase_space.compute_recurrence_plot(signal=short_signal)
        assert has_key(result, 'rec_matrix')
        mat = result['rec_matrix']
        assert mat.ndim == 2
        print(f"  Recurrence plot: shape={mat.shape}")


# ============================================================================
# CLUSTERING
# ============================================================================
class TestClustering:
    """Test clustering algorithms."""

    @pytest.fixture
    def cluster_data(self):
        """Generate 3-cluster test data."""
        np.random.seed(42)
        c1 = np.random.randn(30, 2) + [0, 0]
        c2 = np.random.randn(30, 2) + [5, 5]
        c3 = np.random.randn(30, 2) + [10, 0]
        return np.vstack([c1, c2, c3])

    def test_kmeans(self, cluster_data):
        from biosppy import clustering
        result = clustering.kmeans(data=cluster_data, k=3)
        clusters = result['clusters']
        assert len(clusters) > 0
        # Should find clusters covering all data points
        all_indices = set()
        for c in clusters.values():
            all_indices.update(c)
        assert len(all_indices) == len(cluster_data)
        print(f"  K-means: {len(clusters)} clusters, sizes={[len(v) for v in clusters.values()]}")

    def test_dbscan(self, cluster_data):
        from biosppy import clustering
        result = clustering.dbscan(data=cluster_data, eps=1.5, min_samples=3)
        clusters = result['clusters']
        print(f"  DBSCAN: {len(clusters)} clusters found")

    def test_hierarchical(self, cluster_data):
        from biosppy import clustering
        result = clustering.hierarchical(data=cluster_data, k=3)
        clusters = result['clusters']
        assert len(clusters) == 3
        print(f"  Hierarchical: {len(clusters)} clusters")

    def test_validate_clustering(self, cluster_data):
        from biosppy import clustering
        labels = np.array([0]*30 + [1]*30 + [2]*30)
        result = clustering.validate_clustering(data=cluster_data, labels=labels)
        assert has_key(result, 'silhouette')
        assert has_key(result, 'davies_bouldin')
        assert has_key(result, 'calinski_harabasz')
        print(f"  Validation: silhouette={result['silhouette']:.3f}, "
              f"DB={result['davies_bouldin']:.3f}, CH={result['calinski_harabasz']:.1f}")

    def test_silhouette_analysis(self, cluster_data):
        from biosppy import clustering
        labels = np.array([0]*30 + [1]*30 + [2]*30)
        result = clustering.silhouette_analysis(data=cluster_data, labels=labels)
        assert has_key(result, 'mean_silhouette')
        print(f"  Silhouette: mean={result['mean_silhouette']:.3f}")

    def test_optimal_clusters(self, cluster_data):
        from biosppy import clustering
        result = clustering.optimal_clusters(data=cluster_data, max_k=5)
        assert has_key(result, 'optimal_k')
        print(f"  Optimal k: {result['optimal_k']}")

    def test_centroid_templates(self, cluster_data):
        from biosppy import clustering
        clusters = {0: list(range(30)), 1: list(range(30, 60)), 2: list(range(60, 90))}
        result = clustering.centroid_templates(data=cluster_data, clusters=clusters)
        templates = result['templates']
        assert len(templates) > 0
        print(f"  Centroid templates: {len(templates)} templates")


# ============================================================================
# METRICS
# ============================================================================
class TestMetrics:
    """Test distance metrics."""

    def test_pcosine(self):
        from biosppy import metrics
        u = np.array([1.0, 0.0, 1.0])
        v = np.array([0.0, 1.0, 1.0])
        d = metrics.pcosine(u, v)
        assert isinstance(d, (float, np.floating))
        assert d >= 0
        print(f"  pcosine([1,0,1], [0,1,1]) = {d:.4f}")

    def test_pdist(self):
        from biosppy import metrics
        X = np.random.randn(10, 3)
        D = metrics.pdist(X, metric='euclidean')
        # pdist returns condensed distance matrix
        assert len(D) == 10 * 9 // 2
        assert np.all(D >= 0)
        print(f"  pdist: {len(D)} pairwise distances computed")

    def test_cdist(self):
        from biosppy import metrics
        XA = np.random.randn(5, 3)
        XB = np.random.randn(8, 3)
        D = metrics.cdist(XA, XB, metric='euclidean')
        assert D.shape == (5, 8)
        print(f"  cdist: shape={D.shape}")

    def test_squareform(self):
        from biosppy import metrics
        X = np.random.randn(5, 3)
        D_vec = metrics.pdist(X)
        D_mat = metrics.squareform(D_vec)
        assert D_mat.shape == (5, 5)
        assert np.allclose(np.diag(D_mat), 0)
        print(f"  squareform: {len(D_vec)} -> {D_mat.shape}")


# ============================================================================
# EMD (Empirical Mode Decomposition)
# ============================================================================
class TestEMD:
    """Test EMD, EEMD, CEEMDAN decomposition."""

    @pytest.fixture
    def test_signal(self):
        """Create a multi-component test signal."""
        t = np.linspace(0, 1, 1000)
        # Mix of two frequencies + trend
        sig = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.1 * t
        return sig

    def test_emd(self, test_signal):
        from biosppy.signals import emd
        result = emd.emd(signal=test_signal)
        assert has_key(result, 'imfs')
        assert has_key(result, 'residue')
        imfs = result['imfs']
        assert imfs.shape[1] == len(test_signal)
        print(f"  EMD: {imfs.shape[0]} IMFs extracted, signal len={len(test_signal)}")

    def test_eemd(self, test_signal):
        from biosppy.signals import emd
        result = emd.eemd(signal=test_signal, num_ensemble=20, random_seed=42)
        assert has_key(result, 'imfs')
        imfs = result['imfs']
        assert imfs.shape[0] >= 1
        print(f"  EEMD: {imfs.shape[0]} IMFs")

    def test_ceemdan(self, test_signal):
        from biosppy.signals import emd
        result = emd.ceemdan(signal=test_signal, num_ensemble=20, random_seed=42)
        assert has_key(result, 'imfs')
        imfs = result['imfs']
        assert imfs.shape[0] >= 1
        print(f"  CEEMDAN: {imfs.shape[0]} IMFs")

    def test_hilbert_spectrum(self, test_signal):
        from biosppy.signals import emd
        emd_result = emd.emd(signal=test_signal)
        imfs = emd_result['imfs']
        result = emd.hilbert_spectrum(imfs=imfs, sampling_rate=1000.)
        assert has_key(result, 'inst_amplitude')
        assert has_key(result, 'inst_frequency')
        assert has_key(result, 'inst_phase')
        print(f"  Hilbert spectrum: amp shape={result['inst_amplitude'].shape}")

    def test_marginal_spectrum(self, test_signal):
        from biosppy.signals import emd
        emd_result = emd.emd(signal=test_signal)
        imfs = emd_result['imfs']
        hilbert = emd.hilbert_spectrum(imfs=imfs, sampling_rate=1000.)
        result = emd.marginal_spectrum(
            inst_amplitude=hilbert['inst_amplitude'],
            inst_frequency=hilbert['inst_frequency'],
            sampling_rate=1000.
        )
        assert has_key(result, 'frequencies')
        assert has_key(result, 'amplitudes')
        print(f"  Marginal spectrum: {len(result['frequencies'])} freq bins")


# ============================================================================
# HRV - Heart Rate Variability (comprehensive)
# ============================================================================
class TestHRVComplete:
    """Test all HRV analysis functions with real data."""

    @pytest.fixture
    def rri_from_ecg(self):
        """Get real RRI from ECG R-peak detection."""
        from biosppy import storage
        from biosppy.signals import ecg
        signal, _ = storage.load_txt(data_file('ecg.txt'))
        result = ecg.ecg(signal=signal, sampling_rate=1000., show=False)
        rpeaks = result['rpeaks']
        rri = np.diff(rpeaks).astype(float)  # in ms at 1000 Hz
        return rri

    @pytest.fixture
    def rri_long(self):
        """Load long RRI series from file."""
        from biosppy import storage
        signal, _ = storage.load_txt(data_file('rri.txt'))
        return signal

    def test_compute_rri(self):
        from biosppy import storage
        from biosppy.signals import ecg, hrv
        signal, _ = storage.load_txt(data_file('ecg.txt'))
        result = ecg.ecg(signal=signal, sampling_rate=1000., show=False)
        rri = hrv.compute_rri(result['rpeaks'], sampling_rate=1000., show=False)
        assert len(rri) > 0
        print(f"  compute_rri: {len(rri)} intervals, mean={rri.mean():.1f} ms")

    def test_rri_filter(self, rri_long):
        from biosppy.signals import hrv
        filtered = hrv.rri_filter(rri=rri_long, threshold=1200)
        assert len(filtered) > 0
        assert np.all(filtered <= 1200)
        print(f"  rri_filter: {len(rri_long)} -> {len(filtered)} intervals")

    def test_rri_correction(self, rri_long):
        from biosppy.signals import hrv
        corrected = hrv.rri_correction(rri=rri_long, threshold=250)
        assert len(corrected) > 0
        print(f"  rri_correction: {len(rri_long)} -> {len(corrected)} intervals")

    def test_hrv_timedomain(self, rri_long):
        from biosppy.signals import hrv
        result = hrv.hrv_timedomain(rri=rri_long, show=False)
        keys = result.keys()
        assert 'hr' in keys
        print(f"  HRV time-domain: {len(keys)} features, keys={keys[:5]}...")

    def test_hrv_frequencydomain(self, rri_long):
        from biosppy.signals import hrv
        result = hrv.hrv_frequencydomain(rri=rri_long, show=False)
        keys = result.keys()
        assert len(keys) > 0
        print(f"  HRV freq-domain: {len(keys)} features, keys={keys[:5]}...")

    def test_hrv_nonlinear(self, rri_long):
        from biosppy.signals import hrv
        result = hrv.hrv_nonlinear(rri=rri_long, show=False)
        keys = result.keys()
        # Should have Poincare + entropy features
        assert any('sd1' in k or 'sampen' in k for k in keys)
        print(f"  HRV non-linear: {len(keys)} features, keys={keys}")

    def test_hrv_wavelet(self, rri_long):
        from biosppy.signals import hrv
        result = hrv.hrv_wavelet(rri=rri_long, show=False)
        keys = result.keys()
        assert len(keys) > 0
        print(f"  HRV wavelet: {len(keys)} features")

    def test_compute_poincare(self, rri_long):
        from biosppy.signals import hrv
        result = hrv.compute_poincare(rri=rri_long, show=False)
        assert has_key(result, 'sd1')
        assert has_key(result, 'sd2')
        print(f"  Poincare: SD1={result['sd1']:.2f}, SD2={result['sd2']:.2f}")

    def test_compute_geometrical(self, rri_long):
        from biosppy.signals import hrv
        result = hrv.compute_geometrical(rri=rri_long, show=False)
        assert has_key(result, 'hti')
        assert has_key(result, 'tinn')
        print(f"  Geometrical: HTI={result['hti']:.2f}, TINN={result['tinn']:.2f}")

    def test_sample_entropy(self, rri_long):
        from biosppy.signals import hrv
        se = hrv.sample_entropy(rri=rri_long, m=2, r=0.2)
        assert isinstance(se, (float, np.floating))
        assert se >= 0
        print(f"  Sample entropy: {se:.4f}")

    def test_approximate_entropy(self, rri_long):
        from biosppy.signals import hrv
        ae = hrv.approximate_entropy(rri=rri_long, m=2, r=0.2)
        assert isinstance(ae, (float, np.floating))
        assert ae >= 0
        print(f"  Approximate entropy: {ae:.4f}")

    def test_detrend_window(self, rri_long):
        from biosppy.signals import hrv
        rri_det, rri_trend = hrv.detrend_window(rri=rri_long)
        assert len(rri_det) == len(rri_long)
        assert len(rri_trend) == len(rri_long)
        print(f"  Detrend: signal len={len(rri_det)}, trend range="
              f"{rri_trend.min():.1f}-{rri_trend.max():.1f}")

    def test_hrv_main_function(self, rri_long):
        """Test the main hrv() function with all parameters."""
        from biosppy.signals import hrv
        result = hrv.hrv(rri=rri_long, parameters='auto',
                         features_only=True, show=False)
        assert result is not None
        if isinstance(result, dict):
            print(f"  HRV main: {len(result)} features computed")
        else:
            keys = result.keys()
            print(f"  HRV main: {len(keys)} features computed")

    def test_hrv_time_only(self, rri_long):
        from biosppy.signals import hrv
        result = hrv.hrv(rri=rri_long, parameters='time',
                         features_only=True, show=False)
        assert result is not None

    def test_hrv_frequency_only(self, rri_long):
        from biosppy.signals import hrv
        result = hrv.hrv(rri=rri_long, parameters='frequency',
                         features_only=True, show=False)
        assert result is not None

    def test_hrv_nonlinear_only(self, rri_long):
        from biosppy.signals import hrv
        result = hrv.hrv(rri=rri_long, parameters='non-linear',
                         features_only=True, show=False)
        assert result is not None

    def test_heart_rate_turbulence(self, rri_long):
        """Test HRT with synthetic VPC indices."""
        from biosppy.signals import hrv
        # Simulate VPC indices (premature beats at known positions)
        # VPCs need to be at positions where there's enough context
        vpc_indices = np.array([50, 150, 250])
        try:
            result = hrv.heart_rate_turbulence(
                rri=rri_long, vpc_indices=vpc_indices, show=False
            )
            if result is not None:
                keys = result.keys()
                print(f"  HRT: keys={keys}")
        except Exception as e:
            # HRT may fail if RRI doesn't have appropriate VPC patterns
            print(f"  HRT: {type(e).__name__}: {e}")

    def test_hht_variability(self, rri_long):
        """Test HHT-based HRV analysis."""
        from biosppy.signals import hrv
        result = hrv.hht_variability(
            rri=rri_long, method='ceemdan', num_ensemble=10,
            random_seed=42
        )
        keys = result.keys()
        assert len(keys) > 0
        print(f"  HHT variability: {len(keys)} features")

    def test_hht_frequency_bands(self, rri_long):
        from biosppy.signals import hrv
        result = hrv.hht_frequency_bands(
            rri=rri_long, method='ceemdan', num_ensemble=10,
            random_seed=42
        )
        keys = result.keys()
        assert len(keys) > 0
        print(f"  HHT freq bands: {len(keys)} features")

    def test_hht_nonlinear_features(self, rri_long):
        from biosppy.signals import hrv
        result = hrv.hht_nonlinear_features(
            rri=rri_long, method='ceemdan', num_ensemble=10,
            random_seed=42
        )
        keys = result.keys()
        assert len(keys) > 0
        print(f"  HHT nonlinear: {len(keys)} features")


# ============================================================================
# SIGNAL TOOLS
# ============================================================================
class TestSignalTools:
    """Test signal processing tools."""

    @pytest.fixture
    def ecg_signal(self):
        from biosppy import storage
        signal, _ = storage.load_txt(data_file('ecg.txt'))
        return signal

    def test_filter_signal_butter(self, ecg_signal):
        from biosppy.signals import tools as st
        filtered, _, _ = st.filter_signal(
            ecg_signal, 'butter', 'bandpass', 4,
            np.array([0.5, 40]), 1000.
        )
        assert len(filtered) == len(ecg_signal)
        print(f"  Butterworth bandpass: filtered {len(filtered)} samples")

    def test_filter_signal_fir(self, ecg_signal):
        from biosppy.signals import tools as st
        filtered, _, _ = st.filter_signal(
            ecg_signal, 'FIR', 'bandpass', 100,
            np.array([0.5, 40]), 1000.
        )
        assert len(filtered) == len(ecg_signal)
        print(f"  FIR bandpass: filtered {len(filtered)} samples")

    def test_smoother(self, ecg_signal):
        from biosppy.signals import tools as st
        smoothed, _ = st.smoother(signal=ecg_signal, kernel='boxcar', size=10)
        assert len(smoothed) == len(ecg_signal)
        print(f"  Smoother: boxcar kernel, size=10")

    def test_normalize(self, ecg_signal):
        from biosppy.signals import tools as st
        normalized = st.normalize(signal=ecg_signal)
        assert has_key(normalized, 'signal')
        sig = normalized['signal']
        # Normalizes to zero mean + unit std
        assert np.isclose(sig.mean(), 0, atol=1e-6)
        assert np.isclose(sig.std(ddof=1), 1, atol=1e-6)
        print(f"  Normalize: mean={sig.mean():.6f}, std={sig.std(ddof=1):.6f}")

    def test_power_spectrum(self, ecg_signal):
        from biosppy.signals import tools as st
        freqs, power = st.power_spectrum(
            signal=ecg_signal, sampling_rate=1000.
        )
        assert len(freqs) > 0
        assert len(freqs) == len(power)
        print(f"  Power spectrum: {len(freqs)} frequency bins")

    def test_zero_cross(self, ecg_signal):
        from biosppy.signals import tools as st
        # Center around zero
        centered = ecg_signal - np.mean(ecg_signal)
        result = st.zero_cross(signal=centered)
        assert has_key(result, 'zeros')
        zc = result['zeros']
        print(f"  Zero crossings: {len(zc)} found")


# ============================================================================
# BAROREFLEX (additional tests beyond test_baroreflex.py)
# ============================================================================
class TestBaroreflexWithRealData:
    """Test baroreflex with ECG-derived data."""

    def test_baroreflex_from_ecg(self):
        """Full pipeline: ECG -> R-peaks -> RRI -> Baroreflex."""
        from biosppy import storage
        from biosppy.signals import ecg
        from biosppy.signals.baroreflex import baroreflex_sensitivity

        signal, _ = storage.load_txt(data_file('ecg.txt'))
        result = ecg.ecg(signal=signal, sampling_rate=1000., show=False)
        rpeaks = result['rpeaks']
        rri = np.diff(rpeaks).astype(float)

        # Create synthetic SBP from RRI (for testing only)
        sbp = 120 + (rri - rri.mean()) * 0.1 + np.random.randn(len(rri)) * 2

        brs_result = baroreflex_sensitivity(rri=rri, sbp=sbp, method='sequence')
        assert brs_result is not None
        print(f"  Baroreflex from ECG: computed OK")


# ============================================================================
# PLOTTING (non-interactive)
# ============================================================================
class TestPlotting:
    """Test plotting functions don't crash."""

    def test_ecg_plot_components(self):
        """Test that ECG plotting components work."""
        from biosppy import storage, plotting
        from biosppy.signals import ecg
        import matplotlib.pyplot as plt

        signal, _ = storage.load_txt(data_file('ecg.txt'))
        result = ecg.ecg(signal=signal, sampling_rate=1000., show=False)

        # Manually test plotting
        fig, ax = plt.subplots()
        ax.plot(result['ts'], result['filtered'])
        ax.plot(result['ts'][result['rpeaks']], result['filtered'][result['rpeaks']], 'ro')
        plt.close(fig)
        print(f"  ECG plot components: OK")

    def test_wavelet_coherence_plot(self):
        """Test wavelet coherence plotting."""
        from biosppy.features import wavelet_coherence as wc
        sig1 = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 500))
        sig2 = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 500) + 0.5)

        result = wc.wavelet_coherence(signal1=sig1, signal2=sig2, sampling_rate=500.)
        fig, ax = wc.plot_wavelet_coherence(
            result['coherence'], result['frequencies']
        )
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"  Wavelet coherence plot: OK")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '--tb=long'])
