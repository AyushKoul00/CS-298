#!/usr/bin/env python3
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from multiprocessing import cpu_count

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.decomposition import PCA

# --- Global Constants and Configuration ---
MODEL = "bert"
MALWARE_DIR = Path("../dataset/")  # Directory containing malware type folders
SAVED_MODELS_DIR = Path(f"../saved_models/{MODEL}/")
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
FILE_NAME = "mean_embedding_per_file2.pkl"

NUM_CORES = cpu_count()

import os
os.environ['OPENBLAS_NUM_THREADS'] = '64'

# Option: if True, apply L2 normalization to embeddings before clustering/visualization.
NORMALIZE_EMBEDDINGS: bool = False

# Expectation Maximization (Gaussian Mixture) Hyperparameters
EM_N_COMPONENTS: int = 10  # Set desired number of mixture components (clusters)

# --- Logging Setup ---
logging.basicConfig(
    filename=SAVED_MODELS_DIR / 'em.log',  # Log file path
    level=logging.INFO,  # Logging level: INFO and above
    format="%(asctime)s - %(levelname)s - %(message)s",  # Format as per template
    filemode='w'  # Override log file on each run
)
logger = logging.getLogger(__name__)


# --- Data Loading and Preprocessing ---
def load_mean_embeddings(filepath: Path) -> Dict[Any, np.ndarray]:
    """Load mean embeddings from a pickle file."""
    logger.info("Loading embeddings from %s", filepath)
    with filepath.open("rb") as f:
        data = pickle.load(f)
    logger.info("Loaded %d embeddings from %s", len(data), filepath)
    return data


def prepare_data(mean_embeddings: Dict[Any, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare embeddings and their associated true labels.
    The expected key in mean_embeddings is a tuple where the first element is the malware type.
    """
    embeddings: List[np.ndarray] = []
    labels: List[Any] = []
    first: bool = True
    for (malware_type, _), vector in mean_embeddings.items():
        if first:
            logger.info("Embedding shape: %s", vector.shape)
            first = False
        embeddings.append(vector)
        labels.append(malware_type)
    return np.array(embeddings), np.array(labels)


def maybe_normalize_embeddings(embeddings: np.ndarray, normalize: bool) -> np.ndarray:
    """Optionally apply L2 normalization to embeddings."""
    if normalize:
        normalizer = Normalizer(norm="l2")
        logger.info("Normalizing embeddings with L2 norm.")
        return normalizer.fit_transform(embeddings)
    else:
        logger.info("Skipping normalization of embeddings.")
        return embeddings


# --- Clustering with Expectation Maximization (Gaussian Mixture) ---
def perform_em(embeddings: np.ndarray, n_components: int = EM_N_COMPONENTS) -> np.ndarray:
    """
    Perform clustering using the Gaussian Mixture Model (Expectation Maximization).
    
    Returns:
        cluster_labels: An array of predicted cluster labels.
    """
    logger.info("Running Gaussian Mixture Model clustering with n_components=%d", n_components)
    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
    gmm.fit(embeddings)
    cluster_labels = gmm.predict(embeddings)
    return cluster_labels


def evaluate_clustering(true_labels: np.ndarray, cluster_labels: np.ndarray, embeddings: np.ndarray) -> None:
    """Evaluate the clustering with ARI, NMI, and Silhouette score (computed on numeric embeddings)."""
    if len(np.unique(true_labels)) > 1:
        ari = adjusted_rand_score(true_labels, cluster_labels)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)
        try:
            silhouette = silhouette_score(embeddings, cluster_labels)
            logger.info("Silhouette Coefficient: %.4f", silhouette)
        except Exception as e:
            logger.warning("Silhouette score calculation failed: %s", e)
        logger.info("Adjusted Rand Index: %.4f", ari)
        logger.info("Normalized Mutual Information: %.4f", nmi)
    else:
        logger.warning("Not enough unique true labels for evaluation.")


def print_cluster_composition(true_labels: np.ndarray, cluster_labels: np.ndarray) -> None:
    """Print the cluster composition by malware type."""
    unique_clusters = np.unique(cluster_labels)
    logger.info("Cluster composition:")
    for cluster in unique_clusters:
        cluster_mask = cluster_labels == cluster
        cluster_name = f"Cluster {cluster}"
        types, counts = np.unique(true_labels[cluster_mask], return_counts=True)
        type_counts = dict(zip(types, counts))
        logger.info(f"{cluster_name}:")
        for malware_type, count in type_counts.items():
            logger.info(f"\t{malware_type}: {count} samples")


# --- Dimensionality Reduction ---
def reduce_dimensions(embeddings: np.ndarray, method: str = "umap", n_components: int = 2) -> np.ndarray:
    """
    Reduce dimensionality of embeddings.
    
    :param method: Either 'umap' or 'pca'
    """
    logger.info("Reducing dimensions using %s to %d components.", method, n_components)
    if method == "umap":
        reducer = umap.UMAP(n_components=n_components)
    elif method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError("Method must be either 'umap' or 'pca'")
    return reducer.fit_transform(embeddings)


# --- Visualization ---
def plot_true_labels(projected_embeddings: np.ndarray, true_labels: np.ndarray, save_path: Path) -> None:
    """Visualize the embeddings colored by true malware types."""
    plt.figure(figsize=(8, 8))
    factorized, uniques = pd.factorize(true_labels)
    scatter = plt.scatter(
        projected_embeddings[:, 0],
        projected_embeddings[:, 1],
        c=factorized,
        cmap="Spectral",
        alpha=0.7,
        s=15,
    )
    plt.title("True Malware Types Visualization")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    n_uniques = len(uniques)
    legend_handles = [
        mpatches.Patch(color=scatter.cmap(i / max(1, n_uniques - 1)), label=label)
        for i, label in enumerate(uniques)
    ]
    plt.legend(handles=legend_handles, title="Malware Types")
    plt.tight_layout()
    plt.savefig(save_path / "true_labels_visualization_em.png", dpi=300)
    plt.show()


def plot_cluster_assignments(projected_embeddings: np.ndarray, cluster_labels: np.ndarray, save_path: Path) -> None:
    """Visualize the embeddings colored by Gaussian Mixture cluster assignments."""
    plt.figure(figsize=(8, 8))
    unique_clusters = sorted(np.unique(cluster_labels))
    n_clusters = len(unique_clusters)
    cmap = plt.colormaps.get_cmap("tab20")
    cluster_color_map = {cluster: cmap(i / max(1, n_clusters - 1)) for i, cluster in enumerate(unique_clusters)}
    colors = [cluster_color_map[label] for label in cluster_labels]
    plt.scatter(
        projected_embeddings[:, 0],
        projected_embeddings[:, 1],
        c=colors,
        alpha=0.7,
        s=15,
    )
    plt.title(f"Gaussian Mixture Clustering (n_components={EM_N_COMPONENTS})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    legend_handles = [
        mpatches.Patch(color=cluster_color_map[cluster], label=f"Cluster {cluster}")
        for cluster in unique_clusters
    ]
    plt.legend(handles=legend_handles, title="Clusters")
    plt.tight_layout()
    plt.savefig(save_path / "cluster_visualization_em.png", dpi=300)
    plt.show()


def plot_3d_visualization(embeddings: np.ndarray, true_labels: np.ndarray) -> None:
    """3D visualization of the embeddings colored by true malware types."""
    projected_3d = reduce_dimensions(embeddings, method="umap", n_components=3)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    factorized, _ = pd.factorize(true_labels)
    scatter = ax.scatter3D(
        projected_3d[:, 0],
        projected_3d[:, 1],
        projected_3d[:, 2],
        c=factorized,
        cmap="Spectral",
        alpha=0.7,
        s=15,
    )
    plt.title("3D Malware Embedding Visualization")
    legend1 = ax.legend(*scatter.legend_elements(), title="Malware Types")
    ax.add_artist(legend1)
    plt.show()


def plot_pca_em(embeddings: np.ndarray, n_components: int = EM_N_COMPONENTS) -> None:
    """
    Visualize Gaussian Mixture clustering on PCA-reduced embeddings.
    Also computes the silhouette score.
    """
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca = pca_2d.fit_transform(embeddings)
    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
    gmm.fit(embeddings)
    labels = gmm.predict(embeddings)
    try:
        silhouette = silhouette_score(embeddings, labels)
        logger.info("PCA-EM Silhouette Coefficient: %.3f", silhouette)
    except Exception as e:
        logger.warning("Silhouette score calculation failed: %s", e)

    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        class_member_mask = labels == k
        xy = X_pca[class_member_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=8,
            alpha=0.7,
            label=f"Cluster {k}",
        )
    plt.title(f"PCA Gaussian Mixture Clustering (n_components={n_components})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Clusters")
    plt.tight_layout()
    plt.show()


# --- Main Execution ---
def main() -> None:
    # Load and prepare data.
    embeddings_path = SAVED_MODELS_DIR / FILE_NAME
    mean_embeddings = load_mean_embeddings(embeddings_path)
    embeddings, true_labels = prepare_data(mean_embeddings)

    # Optionally normalize embeddings.
    processed_embeddings = maybe_normalize_embeddings(embeddings, NORMALIZE_EMBEDDINGS)

    # Perform clustering using Gaussian Mixture Model (Expectation Maximization).
    cluster_labels = perform_em(processed_embeddings, n_components=EM_N_COMPONENTS)
    evaluate_clustering(true_labels, cluster_labels, processed_embeddings)
    print_cluster_composition(true_labels, cluster_labels)

    # Reduce dimensions for 2D visualization (using UMAP).
    projected_embeddings = reduce_dimensions(processed_embeddings, method="umap", n_components=2)

    # Visualize true labels and cluster assignments.
    plot_true_labels(projected_embeddings, true_labels, SAVED_MODELS_DIR)
    plot_cluster_assignments(projected_embeddings, cluster_labels, SAVED_MODELS_DIR)

    # Additional 3D visualization.
    plot_3d_visualization(processed_embeddings, true_labels)

    # PCA-based Gaussian Mixture clustering visualization.
    plot_pca_em(processed_embeddings, n_components=EM_N_COMPONENTS)


if __name__ == "__main__":
    main()
