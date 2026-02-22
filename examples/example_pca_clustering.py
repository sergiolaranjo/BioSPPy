#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example: PCA and Clustering Analysis for Biological Signals

This example demonstrates:
1. Dimensionality reduction using PCA and t-SNE
2. Clustering analysis with K-means
3. Cluster validation using silhouette analysis

"""

import numpy as np
import matplotlib.pyplot as plt
from biosppy import dimensionality_reduction, clustering

# Generate synthetic multi-feature biological signal data
# Simulating features extracted from ECG signals (e.g., HRV features)
np.random.seed(42)

# Create 3 clusters of signals with different characteristics
n_samples_per_cluster = 50
n_features = 20

# Cluster 1: Normal signals
cluster1 = np.random.randn(n_samples_per_cluster, n_features) * 0.5 + np.array([1, 2] + [0] * 18)

# Cluster 2: Abnormal signals (elevated features)
cluster2 = np.random.randn(n_samples_per_cluster, n_features) * 0.5 + np.array([4, 1] + [0] * 18)

# Cluster 3: Different pathology
cluster3 = np.random.randn(n_samples_per_cluster, n_features) * 0.5 + np.array([2, 5] + [0] * 18)

# Combine all data
X = np.vstack([cluster1, cluster2, cluster3])
true_labels = np.array([0] * n_samples_per_cluster +
                       [1] * n_samples_per_cluster +
                       [2] * n_samples_per_cluster)

print("=" * 60)
print("PCA and Clustering Analysis Example")
print("=" * 60)
print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")

# ============================================================================
# 1. PCA - Dimensionality Reduction
# ============================================================================
print("\n" + "=" * 60)
print("1. Principal Component Analysis (PCA)")
print("=" * 60)

# Reduce to 2 dimensions for visualization
pca_result = dimensionality_reduction.pca(data=X, n_components=2)
print(f"\nOriginal dimensions: {X.shape[1]}")
print(f"Reduced dimensions: {pca_result['transformed_data'].shape[1]}")
print(f"\nVariance explained by each component:")
for i, var_ratio in enumerate(pca_result['explained_variance_ratio']):
    print(f"  PC{i+1}: {var_ratio*100:.2f}%")
print(f"Total variance explained: {np.sum(pca_result['explained_variance_ratio'])*100:.2f}%")

# Reduce to preserve 95% of variance
pca_result_95 = dimensionality_reduction.pca(data=X, n_components=0.95)
print(f"\nDimensions to preserve 95% variance: {pca_result_95['transformed_data'].shape[1]}")

# ============================================================================
# 2. K-means Clustering
# ============================================================================
print("\n" + "=" * 60)
print("2. K-means Clustering")
print("=" * 60)

# Perform clustering on reduced data
kmeans_result = clustering.kmeans(data=pca_result['transformed_data'], k=3)
print(f"\nNumber of clusters found: {len(kmeans_result['clusters'])}")
for cluster_id, indices in kmeans_result['clusters'].items():
    if cluster_id != -1:
        print(f"  Cluster {cluster_id}: {len(indices)} samples")

# ============================================================================
# 3. Cluster Validation
# ============================================================================
print("\n" + "=" * 60)
print("3. Cluster Validation")
print("=" * 60)

# Convert clusters to labels
labels = np.full(len(X), -1, dtype=int)
for cluster_id, indices in kmeans_result['clusters'].items():
    if cluster_id != -1:
        labels[indices] = cluster_id

# Validate clustering
validation = clustering.validate_clustering(
    data=pca_result['transformed_data'],
    labels=labels
)

print(f"\nSilhouette Score: {validation['silhouette']:.3f}")
print(f"  (Range: -1 to 1, higher is better)")
print(f"\nDavies-Bouldin Index: {validation['davies_bouldin']:.3f}")
print(f"  (Lower is better)")
print(f"\nCalinski-Harabasz Score: {validation['calinski_harabasz']:.1f}")
print(f"  (Higher is better)")

# Detailed silhouette analysis
sil_analysis = clustering.silhouette_analysis(
    data=pca_result['transformed_data'],
    labels=labels
)

print(f"\nPer-cluster Silhouette Scores:")
for cluster_id, score in sil_analysis['cluster_silhouettes'].items():
    print(f"  Cluster {cluster_id}: {score:.3f}")

# ============================================================================
# 4. Find Optimal Number of Clusters
# ============================================================================
print("\n" + "=" * 60)
print("4. Finding Optimal Number of Clusters")
print("=" * 60)

optimal_result = clustering.optimal_clusters(
    data=pca_result['transformed_data'],
    max_k=6,
    method='kmeans',
    criterion='silhouette'
)

print(f"\nOptimal number of clusters: {optimal_result['optimal_k']}")
print(f"\nSilhouette scores for k=2 to k=6:")
for k, score in enumerate(optimal_result['scores'], start=2):
    print(f"  k={k}: {score:.3f}")

# ============================================================================
# 5. t-SNE for Visualization
# ============================================================================
print("\n" + "=" * 60)
print("5. t-SNE Visualization")
print("=" * 60)

tsne_result = dimensionality_reduction.tsne(
    data=X,
    n_components=2,
    perplexity=30,
    random_state=42
)
print(f"\nt-SNE KL divergence: {tsne_result['kl_divergence']:.3f}")
print("  (Lower values indicate better fit)")

# ============================================================================
# 6. Visualization
# ============================================================================
print("\n" + "=" * 60)
print("6. Generating Visualizations")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# PCA with true labels
axes[0, 0].scatter(pca_result['transformed_data'][:, 0],
                   pca_result['transformed_data'][:, 1],
                   c=true_labels, cmap='viridis', s=50, alpha=0.6)
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')
axes[0, 0].set_title('PCA - True Labels')
axes[0, 0].grid(True, alpha=0.3)

# PCA with predicted labels
axes[0, 1].scatter(pca_result['transformed_data'][:, 0],
                   pca_result['transformed_data'][:, 1],
                   c=labels, cmap='viridis', s=50, alpha=0.6)
axes[0, 1].set_xlabel('PC1')
axes[0, 1].set_ylabel('PC2')
axes[0, 1].set_title('PCA - K-means Clusters')
axes[0, 1].grid(True, alpha=0.3)

# t-SNE with predicted labels
axes[1, 0].scatter(tsne_result['embedding'][:, 0],
                   tsne_result['embedding'][:, 1],
                   c=labels, cmap='viridis', s=50, alpha=0.6)
axes[1, 0].set_xlabel('t-SNE 1')
axes[1, 0].set_ylabel('t-SNE 2')
axes[1, 0].set_title('t-SNE - K-means Clusters')
axes[1, 0].grid(True, alpha=0.3)

# Optimal k curve
axes[1, 1].plot(range(2, 7), optimal_result['scores'], marker='o', linewidth=2)
axes[1, 1].axvline(x=optimal_result['optimal_k'], color='r',
                   linestyle='--', label=f"Optimal k={optimal_result['optimal_k']}")
axes[1, 1].set_xlabel('Number of Clusters (k)')
axes[1, 1].set_ylabel('Silhouette Score')
axes[1, 1].set_title('Optimal Number of Clusters')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('pca_clustering_analysis.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved as 'pca_clustering_analysis.png'")

print("\n" + "=" * 60)
print("Analysis Complete!")
print("=" * 60)
