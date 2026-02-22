# -*- coding: utf-8 -*-
"""
biosppy.dimensionality_reduction
---------------------------------

This module provides dimensionality reduction techniques for feature extraction
and visualization of high-dimensional biological signals.

:copyright: (c) 2015-2025 by Instituto de Telecomunicacoes
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# compat
from __future__ import absolute_import, division, print_function
from six.moves import range

# 3rd party
import numpy as np
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.manifold import TSNE, MDS, Isomap
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# local
from . import utils


def pca(data=None, n_components=None, whiten=False, svd_solver='auto'):
    """Principal Component Analysis (PCA) for dimensionality reduction.

    PCA is a linear dimensionality reduction technique that uses Singular Value
    Decomposition to project the data to a lower dimensional space while
    preserving as much variance as possible.

    Parameters
    ----------
    data : array
        An m by n array of m data samples in an n-dimensional space.
    n_components : int, float, optional
        Number of components to keep. If None, all components are kept.
        If int, n_components principal components are kept.
        If 0 < n_components < 1, select the number of components such that
        the amount of variance explained is greater than this value.
    whiten : bool, optional
        When True, the components are divided by the square root of the
        explained variance and multiplied by sqrt(n_samples). This ensures
        that the components vectors have unit variance.
    svd_solver : str, optional
        Algorithm to use for SVD computation. One of 'auto', 'full', 'arpack',
        'randomized'. Default is 'auto'.

    Returns
    -------
    transformed_data : array
        Data transformed to the principal component space.
    components : array
        Principal components (eigenvectors).
    explained_variance : array
        Amount of variance explained by each component.
    explained_variance_ratio : array
        Percentage of variance explained by each component.
    mean : array
        Per-feature empirical mean.

    Notes
    -----
    * PCA is best suited for data with linear relationships between features.
    * For non-linear relationships, consider t-SNE or UMAP.
    * For non-negative data (e.g., power spectral densities), consider NMF.

    Examples
    --------
    >>> import numpy as np
    >>> from biosppy import dimensionality_reduction
    >>> # Generate sample data
    >>> X = np.random.randn(100, 50)  # 100 samples, 50 features
    >>> # Reduce to 10 dimensions
    >>> result = dimensionality_reduction.pca(data=X, n_components=10)
    >>> print(result['transformed_data'].shape)
    (100, 10)
    >>> # Reduce to preserve 95% of variance
    >>> result = dimensionality_reduction.pca(data=X, n_components=0.95)

    """

    # check inputs
    if data is None:
        raise TypeError("Please specify input data.")

    # ensure 2D array
    data = np.atleast_2d(data)

    # fit PCA
    pca_model = PCA(n_components=n_components, whiten=whiten, svd_solver=svd_solver)
    transformed_data = pca_model.fit_transform(data)

    # prepare output
    args = (
        transformed_data,
        pca_model.components_,
        pca_model.explained_variance_,
        pca_model.explained_variance_ratio_,
        pca_model.mean_
    )
    names = (
        'transformed_data',
        'components',
        'explained_variance',
        'explained_variance_ratio',
        'mean'
    )

    return utils.ReturnTuple(args, names)


def ica(data=None, n_components=None, algorithm='parallel', whiten='unit-variance',
        max_iter=200, tol=1e-4, random_state=None):
    """Independent Component Analysis (ICA) for blind source separation.

    ICA is a computational method for separating a multivariate signal into
    additive subcomponents that are maximally independent. Commonly used for
    artifact removal in EEG/MEG signals.

    Parameters
    ----------
    data : array
        An m by n array of m data samples in an n-dimensional space.
    n_components : int, optional
        Number of components to extract. If None, n_components = min(m, n).
    algorithm : str, optional
        Algorithm for ICA: 'parallel' or 'deflation'. Default is 'parallel'.
    whiten : str, bool, optional
        Whitening strategy: 'unit-variance', 'arbitrary-variance', True, or False.
        Default is 'unit-variance'.
    max_iter : int, optional
        Maximum number of iterations for the algorithm.
    tol : float, optional
        Tolerance for stopping criterion.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    sources : array
        Independent source signals (transformed data).
    mixing_matrix : array
        Mixing matrix (components).
    unmixing_matrix : array
        Unmixing matrix (inverse of mixing matrix).
    mean : array
        Per-feature empirical mean.

    Notes
    -----
    * ICA assumes that the observed signals are linear mixtures of independent sources.
    * Commonly used for EEG/EMG artifact removal (e.g., eye blinks, muscle artifacts).
    * The order of components is arbitrary (unlike PCA).

    Examples
    --------
    >>> import numpy as np
    >>> from biosppy import dimensionality_reduction
    >>> # Generate mixed signals
    >>> s1 = np.sin(2 * np.pi * np.linspace(0, 1, 100))
    >>> s2 = np.sign(np.sin(3 * np.pi * np.linspace(0, 1, 100)))
    >>> S = np.c_[s1, s2]
    >>> A = np.array([[1, 1], [0.5, 2]])  # Mixing matrix
    >>> X = S @ A.T  # Mixed signals
    >>> # Separate sources
    >>> result = dimensionality_reduction.ica(data=X, n_components=2)

    """

    # check inputs
    if data is None:
        raise TypeError("Please specify input data.")

    # ensure 2D array
    data = np.atleast_2d(data)

    # fit ICA
    ica_model = FastICA(
        n_components=n_components,
        algorithm=algorithm,
        whiten=whiten,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )
    sources = ica_model.fit_transform(data)

    # prepare output
    args = (
        sources,
        ica_model.mixing_,
        ica_model.components_,  # This is the unmixing matrix
        ica_model.mean_
    )
    names = (
        'sources',
        'mixing_matrix',
        'unmixing_matrix',
        'mean'
    )

    return utils.ReturnTuple(args, names)


def nmf(data=None, n_components=None, init='nndsvda', max_iter=200,
        tol=1e-4, random_state=None):
    """Non-negative Matrix Factorization (NMF) for non-negative data.

    NMF decomposes non-negative data into non-negative components, making it
    interpretable for applications like spectral decomposition, topic modeling,
    and feature extraction from power spectral densities.

    Parameters
    ----------
    data : array
        An m by n array of m non-negative data samples in an n-dimensional space.
    n_components : int, optional
        Number of components to extract.
    init : str, optional
        Initialization method: 'random', 'nndsvd', 'nndsvda', 'nndsvdar'.
        Default is 'nndsvda'.
    max_iter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Tolerance for stopping criterion.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    transformed_data : array
        Transformed data in the NMF space (W matrix).
    components : array
        Non-negative components (H matrix).
    reconstruction_error : float
        Frobenius norm of the reconstruction error.

    Notes
    -----
    * NMF requires all values in the input data to be non-negative.
    * Useful for decomposing power spectral densities into frequency bands.
    * Produces sparse, interpretable components.

    Examples
    --------
    >>> import numpy as np
    >>> from biosppy import dimensionality_reduction
    >>> # Generate non-negative data (e.g., power spectral density)
    >>> X = np.abs(np.random.randn(100, 50))
    >>> result = dimensionality_reduction.nmf(data=X, n_components=5)

    """

    # check inputs
    if data is None:
        raise TypeError("Please specify input data.")

    # ensure 2D array
    data = np.atleast_2d(data)

    # check for non-negative data
    if np.any(data < 0):
        raise ValueError("NMF requires non-negative data. All values must be >= 0.")

    # fit NMF
    nmf_model = NMF(
        n_components=n_components,
        init=init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )
    transformed_data = nmf_model.fit_transform(data)

    # prepare output
    args = (
        transformed_data,
        nmf_model.components_,
        nmf_model.reconstruction_err_
    )
    names = (
        'transformed_data',
        'components',
        'reconstruction_error'
    )

    return utils.ReturnTuple(args, names)


def tsne(data=None, n_components=2, perplexity=30.0, learning_rate='auto',
         n_iter=1000, metric='euclidean', random_state=None):
    """t-distributed Stochastic Neighbor Embedding (t-SNE) for visualization.

    t-SNE is a non-linear dimensionality reduction technique particularly well
    suited for visualizing high-dimensional data in 2D or 3D spaces. It preserves
    local structure and reveals clusters in the data.

    Parameters
    ----------
    data : array
        An m by n array of m data samples in an n-dimensional space.
    n_components : int, optional
        Dimension of the embedded space (typically 2 or 3). Default is 2.
    perplexity : float, optional
        Related to the number of nearest neighbors used. Larger datasets
        usually require larger perplexity. Typical values: 5-50. Default is 30.
    learning_rate : float, str, optional
        Learning rate for t-SNE optimization. If 'auto', uses n_samples / 12.
        Typical values: 10-1000. Default is 'auto'.
    n_iter : int, optional
        Number of iterations for optimization. Default is 1000.
    metric : str, optional
        Distance metric to use. Default is 'euclidean'.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    embedding : array
        Low-dimensional embedding of the data.
    kl_divergence : float
        Kullback-Leibler divergence after optimization.

    Notes
    -----
    * t-SNE is computationally expensive for large datasets (>10,000 samples).
    * Results can vary between runs due to randomness; set random_state for reproducibility.
    * Not suitable for general dimensionality reduction (only visualization).
    * Does not preserve global structure, only local neighborhoods.

    Examples
    --------
    >>> import numpy as np
    >>> from biosppy import dimensionality_reduction
    >>> # Generate high-dimensional data
    >>> X = np.random.randn(200, 50)
    >>> # Visualize in 2D
    >>> result = dimensionality_reduction.tsne(data=X, n_components=2, perplexity=30)
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(result['embedding'][:, 0], result['embedding'][:, 1])

    """

    # check inputs
    if data is None:
        raise TypeError("Please specify input data.")

    # ensure 2D array
    data = np.atleast_2d(data)

    # fit t-SNE
    tsne_kwargs = dict(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        metric=metric,
        random_state=random_state
    )
    # scikit-learn >= 1.6 renamed n_iter to max_iter
    import inspect
    tsne_params = inspect.signature(TSNE).parameters
    if 'max_iter' in tsne_params:
        tsne_kwargs['max_iter'] = n_iter
    else:
        tsne_kwargs['n_iter'] = n_iter
    tsne_model = TSNE(**tsne_kwargs)
    embedding = tsne_model.fit_transform(data)

    # prepare output
    args = (embedding, tsne_model.kl_divergence_)
    names = ('embedding', 'kl_divergence')

    return utils.ReturnTuple(args, names)


def umap_reduction(data=None, n_components=2, n_neighbors=15, min_dist=0.1,
                   metric='euclidean', random_state=None):
    """Uniform Manifold Approximation and Projection (UMAP) for dimensionality reduction.

    UMAP is a modern non-linear dimensionality reduction technique that preserves
    both local and global structure better than t-SNE, and is significantly faster.

    Parameters
    ----------
    data : array
        An m by n array of m data samples in an n-dimensional space.
    n_components : int, optional
        Dimension of the embedded space. Default is 2.
    n_neighbors : int, optional
        Number of neighbors to consider for manifold approximation.
        Larger values focus on global structure. Default is 15.
    min_dist : float, optional
        Minimum distance between points in the low-dimensional space.
        Smaller values produce tighter clusters. Default is 0.1.
    metric : str, optional
        Distance metric to use. Default is 'euclidean'.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    embedding : array
        Low-dimensional embedding of the data.

    Notes
    -----
    * UMAP is faster than t-SNE and better preserves global structure.
    * Requires the 'umap-learn' package to be installed.
    * Good for both visualization and general dimensionality reduction.
    * More stable than t-SNE across different runs.

    Examples
    --------
    >>> import numpy as np
    >>> from biosppy import dimensionality_reduction
    >>> # Generate high-dimensional data
    >>> X = np.random.randn(500, 100)
    >>> # Reduce to 2D
    >>> result = dimensionality_reduction.umap_reduction(data=X, n_components=2)
    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(result['embedding'][:, 0], result['embedding'][:, 1])

    """

    # check if UMAP is available
    if not UMAP_AVAILABLE:
        raise ImportError(
            "UMAP is not installed. Please install it with: pip install umap-learn"
        )

    # check inputs
    if data is None:
        raise TypeError("Please specify input data.")

    # ensure 2D array
    data = np.atleast_2d(data)

    # fit UMAP
    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    embedding = umap_model.fit_transform(data)

    # prepare output
    args = (embedding,)
    names = ('embedding',)

    return utils.ReturnTuple(args, names)


def mds(data=None, n_components=2, metric=True, random_state=None):
    """Multidimensional Scaling (MDS) for dimensionality reduction.

    MDS seeks a low-dimensional representation of the data in which the distances
    respect well the distances in the original high-dimensional space.

    Parameters
    ----------
    data : array
        An m by n array of m data samples in an n-dimensional space.
    n_components : int, optional
        Number of dimensions in which to embed. Default is 2.
    metric : bool, optional
        If True, performs metric MDS; if False, performs non-metric MDS.
        Default is True.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    embedding : array
        Low-dimensional embedding of the data.
    stress : float
        The final value of the stress (sum of squared distances of disparities
        and distances for all constrained points).

    Examples
    --------
    >>> import numpy as np
    >>> from biosppy import dimensionality_reduction
    >>> X = np.random.randn(100, 50)
    >>> result = dimensionality_reduction.mds(data=X, n_components=2)

    """

    # check inputs
    if data is None:
        raise TypeError("Please specify input data.")

    # ensure 2D array
    data = np.atleast_2d(data)

    # fit MDS
    mds_model = MDS(
        n_components=n_components,
        metric=metric,
        random_state=random_state
    )
    embedding = mds_model.fit_transform(data)

    # prepare output
    args = (embedding, mds_model.stress_)
    names = ('embedding', 'stress')

    return utils.ReturnTuple(args, names)


def isomap(data=None, n_components=2, n_neighbors=5):
    """Isomap embedding for non-linear dimensionality reduction.

    Isomap seeks a lower-dimensional embedding which maintains geodesic distances
    between all points on a manifold.

    Parameters
    ----------
    data : array
        An m by n array of m data samples in an n-dimensional space.
    n_components : int, optional
        Number of coordinates for the manifold. Default is 2.
    n_neighbors : int, optional
        Number of neighbors to consider for each point. Default is 5.

    Returns
    -------
    embedding : array
        Low-dimensional embedding of the data.
    reconstruction_error : float
        Reconstruction error for the embedding.

    Examples
    --------
    >>> import numpy as np
    >>> from biosppy import dimensionality_reduction
    >>> X = np.random.randn(100, 50)
    >>> result = dimensionality_reduction.isomap(data=X, n_components=2, n_neighbors=5)

    """

    # check inputs
    if data is None:
        raise TypeError("Please specify input data.")

    # ensure 2D array
    data = np.atleast_2d(data)

    # fit Isomap
    isomap_model = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    embedding = isomap_model.fit_transform(data)

    # prepare output
    args = (embedding, isomap_model.reconstruction_error())
    names = ('embedding', 'reconstruction_error')

    return utils.ReturnTuple(args, names)
