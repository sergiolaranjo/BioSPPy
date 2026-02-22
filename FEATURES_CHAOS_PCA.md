# New Features: PCA, Clustering and Chaos Analysis

This document describes the new functionality added to BioSPPy for advanced signal analysis, including dimensionality reduction, unsupervised learning, and nonlinear dynamics analysis.

## 📊 Dimensionality Reduction Module

**Location:** `biosppy/dimensionality_reduction.py`

### Available Methods

#### 1. Principal Component Analysis (PCA)
```python
from biosppy import dimensionality_reduction

# Reduce to 10 dimensions
result = dimensionality_reduction.pca(data=X, n_components=10)

# Or preserve 95% of variance
result = dimensionality_reduction.pca(data=X, n_components=0.95)

# Access results
transformed_data = result['transformed_data']
components = result['components']
explained_variance = result['explained_variance_ratio']
```

**Use cases:**
- Feature reduction before classification
- Noise reduction in biosignals
- Visualization of high-dimensional physiological data

#### 2. Independent Component Analysis (ICA)
```python
# Blind source separation (e.g., EEG artifact removal)
result = dimensionality_reduction.ica(data=X, n_components=5)

sources = result['sources']
mixing_matrix = result['mixing_matrix']
```

**Use cases:**
- EEG/MEG artifact removal (eye blinks, muscle artifacts)
- Separating mixed physiological signals
- fMRI analysis

#### 3. Non-negative Matrix Factorization (NMF)
```python
# For non-negative data (e.g., power spectral densities)
result = dimensionality_reduction.nmf(data=X, n_components=5)
```

**Use cases:**
- Decomposing power spectral densities into frequency bands
- Feature extraction from spectrograms
- Interpretable component analysis

#### 4. t-SNE (t-Distributed Stochastic Neighbor Embedding)
```python
# Visualization in 2D/3D
result = dimensionality_reduction.tsne(
    data=X,
    n_components=2,
    perplexity=30
)

embedding = result['embedding']
```

**Use cases:**
- Visualizing high-dimensional biosignal features
- Exploring clusters in physiological data
- Pattern discovery in medical datasets

#### 5. UMAP (Uniform Manifold Approximation and Projection)
```python
# Modern alternative to t-SNE (faster, preserves global structure)
result = dimensionality_reduction.umap_reduction(
    data=X,
    n_components=2,
    n_neighbors=15
)
```

**Note:** Requires `umap-learn` package: `pip install umap-learn`

**Use cases:**
- Large-scale biosignal visualization
- Better preservation of global structure than t-SNE
- General dimensionality reduction

---

## 🎯 Enhanced Clustering Module

**Location:** `biosppy/clustering.py` (extended)

### New Functions

#### 1. Cluster Validation
```python
from biosppy import clustering

# Validate clustering quality
validation = clustering.validate_clustering(data=X, labels=labels)

print(f"Silhouette Score: {validation['silhouette']}")
print(f"Davies-Bouldin Index: {validation['davies_bouldin']}")
print(f"Calinski-Harabasz Score: {validation['calinski_harabasz']}")
```

**Metrics:**
- **Silhouette Score** (-1 to 1): Higher is better
- **Davies-Bouldin Index** (0 to ∞): Lower is better
- **Calinski-Harabasz Score** (0 to ∞): Higher is better

#### 2. Silhouette Analysis
```python
# Detailed per-sample analysis
result = clustering.silhouette_analysis(data=X, labels=labels)

sample_scores = result['sample_silhouette']
mean_score = result['mean_silhouette']
cluster_scores = result['cluster_silhouettes']
```

**Use cases:**
- Identifying poorly clustered samples
- Detecting outliers
- Comparing clustering quality across methods

#### 3. Optimal Cluster Number Detection
```python
# Automatically find optimal k
result = clustering.optimal_clusters(
    data=X,
    max_k=10,
    method='kmeans',
    criterion='silhouette'
)

optimal_k = result['optimal_k']
scores = result['scores']
```

**Use cases:**
- Determining the natural number of clusters in data
- Elbow method alternative
- Model selection for clustering algorithms

---

## 🌀 Chaos Theory Analysis Module

**Location:** `biosppy/chaos.py`

### Entropy Measures

#### 1. Shannon Entropy
```python
from biosppy import chaos

result = chaos.shannon_entropy(signal=signal)
entropy = result['entropy']
```

**Interpretation:** Measures information content and complexity
- Low entropy: Regular, predictable signal
- High entropy: Random, complex signal

**Applications:** Signal quality assessment, complexity quantification

#### 2. Sample Entropy (SampEn)
```python
result = chaos.sample_entropy(signal=signal, m=2, r=0.2)
sampen = result['sampen']
```

**Interpretation:** Quantifies regularity and predictability
- Lower values: More regular patterns
- Higher values: More complex patterns

**Applications:** Heart rate variability (HRV) analysis, signal complexity

#### 3. Approximate Entropy (ApEn)
```python
result = chaos.approximate_entropy(signal=signal, m=2, r=0.2)
apen = result['apen']
```

**Similar to Sample Entropy but includes self-matches**

#### 4. Permutation Entropy (PE)
```python
result = chaos.permutation_entropy(signal=signal, order=3)
pe = result['pe']
```

**Interpretation:** Robust to noise, fast computation
- Values near 0: Deterministic
- Values near 1: Random

**Applications:** EEG analysis, fast complexity assessment

#### 5. Multiscale Entropy (MSE)
```python
result = chaos.multiscale_entropy(signal=signal, max_scale=20)
mse_values = result['mse']
scales = result['scales']
```

**Interpretation:** Analyzes complexity across multiple time scales
- Complex signals: High entropy across scales
- Random signals: High entropy only at small scales

**Applications:** Distinguishing healthy vs. pathological signals

### Fractal Dimension Estimators

#### 1. Detrended Fluctuation Analysis (DFA)
```python
result = chaos.dfa(signal=signal)
alpha = result['alpha']
```

**Interpretation of α:**
- α < 0.5: Anti-correlated (mean-reverting)
- α = 0.5: White noise (uncorrelated)
- 0.5 < α < 1.0: Correlated (long-range correlations)
- α = 1.0: Pink noise (1/f)
- α > 1.0: Non-stationary

**Applications:**
- Heart rate variability (healthy individuals: α ≈ 1.0)
- Long-range correlation detection
- Signal classification

#### 2. Higuchi Fractal Dimension
```python
result = chaos.higuchi_fd(signal=signal, k_max=10)
hfd = result['hfd']
```

**Interpretation:**
- Values: 1 (smooth) to 2 (highly irregular)
- Higher values: More complex/irregular

**Applications:** EEG analysis, seizure detection

#### 3. Petrosian Fractal Dimension
```python
result = chaos.petrosian_fd(signal=signal)
pfd = result['pfd']
```

**Fast computation, suitable for real-time applications**

#### 4. Katz Fractal Dimension
```python
result = chaos.katz_fd(signal=signal)
kfd = result['kfd']
```

**Simple and fast waveform complexity measure**

### Long-Range Correlation Measures

#### 1. Hurst Exponent
```python
result = chaos.hurst_exponent(signal=signal)
H = result['H']
```

**Interpretation of H:**
- H < 0.5: Mean-reverting (anti-persistent)
- H = 0.5: Random walk
- H > 0.5: Trending (persistent)

**Applications:**
- Financial time series
- Hydrology
- Physiological signal analysis

#### 2. Lyapunov Exponent
```python
result = chaos.lyapunov_exponent(signal=signal, emb_dim=10)
lambda_max = result['lambda_max']
```

**Interpretation of λ:**
- λ > 0: Chaotic (exponential divergence)
- λ = 0: Periodic/quasi-periodic
- λ < 0: Stable fixed point

**Applications:**
- Detecting deterministic chaos
- Characterizing nonlinear dynamics
- Seizure prediction in EEG

---

## 📖 Usage Examples

### Example 1: HRV Analysis with Chaos Measures
```python
from biosppy import signals, chaos
import numpy as np

# Load ECG and extract RR intervals
ecg_result = signals.ecg.ecg(signal=ecg_data, sampling_rate=1000)
rr_intervals = np.diff(ecg_result['rpeaks']) / 1000.0  # in seconds

# Compute nonlinear HRV features
sampen = chaos.sample_entropy(signal=rr_intervals)['sampen']
dfa_alpha = chaos.dfa(signal=rr_intervals)['alpha']
hurst = chaos.hurst_exponent(signal=rr_intervals)['H']

print(f"Sample Entropy: {sampen:.3f}")
print(f"DFA α: {dfa_alpha:.3f}")
print(f"Hurst Exponent: {hurst:.3f}")
```

### Example 2: EEG Complexity Analysis
```python
from biosppy import chaos, dimensionality_reduction

# Compute multiple entropy measures
shannon = chaos.shannon_entropy(signal=eeg_signal)['entropy']
sampen = chaos.sample_entropy(signal=eeg_signal)['sampen']
pe = chaos.permutation_entropy(signal=eeg_signal)['pe']

# Compute fractal dimensions
higuchi = chaos.higuchi_fd(signal=eeg_signal)['hfd']
petrosian = chaos.petrosian_fd(signal=eeg_signal)['pfd']

# Create feature vector
features = [shannon, sampen, pe, higuchi, petrosian]
```

### Example 3: Multi-Channel Signal Clustering
```python
from biosppy import clustering, dimensionality_reduction
import numpy as np

# Extract features from multiple ECG/EEG channels
features = extract_features(multi_channel_data)  # Your feature extraction

# Reduce dimensions
pca_result = dimensionality_reduction.pca(data=features, n_components=0.95)

# Find optimal clusters
optimal = clustering.optimal_clusters(
    data=pca_result['transformed_data'],
    max_k=10
)

# Cluster with optimal k
clusters = clustering.kmeans(
    data=pca_result['transformed_data'],
    k=optimal['optimal_k']
)

# Validate
validation = clustering.validate_clustering(
    data=pca_result['transformed_data'],
    labels=convert_to_labels(clusters['clusters'])
)
```

---

## 🔬 Applications in Biological Signal Analysis

### Heart Rate Variability (HRV)
- **DFA**: Assess autonomic nervous system function
- **Sample Entropy**: Quantify HRV complexity
- **Multiscale Entropy**: Distinguish healthy vs. disease states
- **Hurst Exponent**: Long-term correlation analysis

### Electroencephalography (EEG)
- **Higuchi FD**: Seizure detection and prediction
- **Lyapunov Exponent**: Characterize brain dynamics
- **ICA**: Artifact removal (eye blinks, muscle)
- **Permutation Entropy**: Fast complexity assessment

### Electromyography (EMG)
- **Fractal Dimension**: Muscle fatigue assessment
- **Shannon Entropy**: Signal quality evaluation
- **PCA**: Dimensionality reduction for classification

### Blood Pressure & PPG
- **DFA**: Cardiovascular risk assessment
- **Sample Entropy**: Autonomic function evaluation
- **Clustering**: Patient stratification

---

## 📚 References

### Entropy Methods
1. Richman JS, Moorman JR. (2000). Physiological time-series analysis using approximate entropy and sample entropy. Am J Physiol Heart Circ Physiol.
2. Costa M, Goldberger AL, Peng CK. (2002). Multiscale entropy analysis of complex physiologic time series. Phys Rev Lett.
3. Bandt C, Pompe B. (2002). Permutation entropy: a natural complexity measure for time series. Phys Rev Lett.

### Fractal Analysis
4. Peng CK, et al. (1995). Quantification of scaling exponents and crossover phenomena in nonstationary heartbeat time series. Chaos.
5. Higuchi T. (1988). Approach to an irregular time series on the basis of the fractal theory. Physica D.
6. Hurst HE. (1951). Long-term storage capacity of reservoirs. Trans Am Soc Civ Eng.

### Chaos Theory
7. Rosenstein MT, Collins JJ, De Luca CJ. (1993). A practical method for calculating largest Lyapunov exponents from small data sets. Physica D.

---

## 🚀 Quick Start

```python
# Install BioSPPy with new features
pip install biosppy

# Basic usage
from biosppy import chaos, dimensionality_reduction, clustering

# Chaos analysis
signal = your_biosignal_data
entropy_result = chaos.sample_entropy(signal=signal)
dfa_result = chaos.dfa(signal=signal)
fractal_result = chaos.higuchi_fd(signal=signal)

# Dimensionality reduction
features = your_feature_matrix
pca_result = dimensionality_reduction.pca(data=features, n_components=0.95)

# Clustering with validation
clusters = clustering.kmeans(data=pca_result['transformed_data'], k=3)
validation = clustering.validate_clustering(data=..., labels=...)
```

---

## 💡 Tips for Best Results

1. **Signal Preprocessing**: Always filter and clean signals before analysis
2. **Parameter Selection**:
   - SampEn: Use m=2, r=0.1-0.25 * std(signal)
   - DFA: Ensure signal length > 100 * max_window_size
   - Lyapunov: Requires longer signals (>1000 points)
3. **Normalization**: Normalize features before PCA/clustering
4. **Validation**: Always validate clustering results with multiple metrics
5. **Multiscale Analysis**: Use MSE for comprehensive complexity assessment

---

## 📝 Notes

- All functions follow BioSPPy's `ReturnTuple` pattern for consistent API
- Comprehensive docstrings with examples included
- State-of-the-art algorithms from peer-reviewed literature
- Optimized for physiological signals (ECG, EEG, EMG, HRV, etc.)
- Compatible with existing BioSPPy signal processing pipelines

---

## 🤝 Contributing

These modules follow BioSPPy's coding standards:
- BSD 3-clause license
- PEP 8 style guidelines
- Comprehensive documentation
- Example-driven development

---

**For detailed examples, see:**
- `examples/example_pca_clustering.py`
- `examples/example_chaos_analysis.py`
