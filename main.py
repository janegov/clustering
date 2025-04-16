import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import adjusted_rand_score
from sklearn.datasets import fetch_openml
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

# Load MNIST dataset
mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# Reduce dataset size for faster computation
X, y = X[:1000], y[:1000]

# PCA reduction levels
pca_levels = [2, 50, 100, 200]

# Results dictionary to store Rand Index
results = {}

for n_components in pca_levels:
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)

    # Gaussian Mixture Model (GMM)
    results[n_components] = {}
    results[n_components]['gmm'] = {}
    for k in range(5, 16):
        gmm = GaussianMixture(n_components=k, covariance_type='diag', random_state=42)
        labels = gmm.fit_predict(X_reduced)
        rand_index = adjusted_rand_score(y, labels)
        results[n_components]['gmm'][k] = rand_index

    # Mean Shift
    results[n_components]['mean_shift'] = {}
    bandwidth = estimate_bandwidth(X_reduced, quantile=0.2, random_state=42)
    for kernel_width in np.linspace(bandwidth / 2, bandwidth * 2, 5):
        mean_shift = MeanShift(bandwidth=kernel_width, bin_seeding=True)
        labels = mean_shift.fit_predict(X_reduced)
        rand_index = adjusted_rand_score(y, labels)
        results[n_components]['mean_shift'][kernel_width] = rand_index

    # Normalized Cut
    results[n_components]['normalized_cut'] = {}
    for k in range(5, 16):
        # Create similarity graph using pairwise distances
        similarity_matrix = np.exp(-np.linalg.norm(X_reduced[:, None] - X_reduced[None, :], axis=2) ** 2 / (2. * np.var(X_reduced)))

        # Compute Laplacian and its eigenvectors
        L = laplacian(similarity_matrix, normed=True)
        eigvals, eigvecs = eigh(L, subset_by_index=[0, k - 1])

        # Use k-means on the eigenvectors for clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(eigvecs)
        rand_index = adjusted_rand_score(y, labels)
        results[n_components]['normalized_cut'][k] = rand_index

# Print Results
for n_components, res in results.items():
    print(f"\nPCA Components: {n_components}")
    print("Gaussian Mixture Model:")
    for k, rand_index in res['gmm'].items():
        print(f"  k={k}, Rand Index: {rand_index:.4f}")
    print("Mean Shift:")
    for kernel_width, rand_index in res['mean_shift'].items():
        print(f"  Kernel Width={kernel_width:.2f}, Rand Index: {rand_index:.4f}")
    print("Normalized Cut:")
    for k, rand_index in res['normalized_cut'].items():
        print(f"  k={k}, Rand Index: {rand_index:.4f}")




