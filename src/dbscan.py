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

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import Normalizer
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                            silhouette_score, homogeneity_score, completeness_score,
                            v_measure_score)
from sklearn.decomposition import PCA

# --- Global Constants and Configuration ---
MODEL = "distilbert"
MALWARE_DIR = Path("../dataset/")  # Directory containing malware type folders
SAVED_MODELS_DIR = Path(f"../saved_models/{MODEL}/")
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# MAX_SAMPLES_PER_TYPE: you can change this if you wish to limit files per malware type.
MAX_SAMPLES_PER_TYPE = -1  # -1 to read all files per folder
NUM_CORES = cpu_count()

# New option: if True, apply L2 normalization to embeddings before clustering/visualization.
NORMALIZE_EMBEDDINGS: bool = True

# DBSCAN Hyperparameters (Global Constants)
DBSCAN_EPS: float = 0.01      # Epsilon value for DBSCAN
DBSCAN_MIN_SAMPLES: int = 5  # Minimum samples per cluster for DBSCAN


# --- Logging Setup ---
logging.basicConfig(
    filename=SAVED_MODELS_DIR / 'dbscan.log',  # Log file path
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


# --- Clustering ---
def perform_dbscan(embeddings: np.ndarray, eps: float = DBSCAN_EPS, min_samples: int = DBSCAN_MIN_SAMPLES) -> np.ndarray: # Using global constants for defaults
    """Perform DBSCAN clustering using cosine distance."""
    logger.info("Running DBSCAN (eps=%.2f, min_samples=%d, metric='cosine')", eps, min_samples)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=NUM_CORES)
    return dbscan.fit_predict(embeddings)


def evaluate_clustering(true_labels: np.ndarray, cluster_labels: np.ndarray) -> None:
    """Evaluate the clustering with ARI and NMI."""
    if len(np.unique(true_labels)) > 1:
        ari = adjusted_rand_score(true_labels, cluster_labels)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)
        logger.info("Adjusted Rand Index: %.4f", ari)
        logger.info("Normalized Mutual Information: %.4f", nmi)
    else:
        logger.warning("Not enough unique true labels for evaluation.")


def print_cluster_composition(true_labels: np.ndarray, cluster_labels: np.ndarray) -> None:
    """Print the cluster composition by malware type."""
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    n_noise = np.sum(cluster_labels == -1)
    logger.info("Estimated number of clusters: %d", n_clusters)
    logger.info("Number of noise points: %d", n_noise)

    logger.info("Cluster composition:")
    for cluster in unique_clusters:
        cluster_mask = cluster_labels == cluster
        cluster_name = f"Cluster {cluster}" if cluster != -1 else "Noise"
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
        # Removed random_state=42 to enable potential parallelism and address UserWarning.
        reducer = umap.UMAP(n_components=n_components) # n_jobs will now be used if set globally in UMAP
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
    # Map each unique label to a color.
    n_uniques = len(uniques)
    legend_handles = [
        mpatches.Patch(color=scatter.cmap(i / max(1, n_uniques - 1)), label=label)
        for i, label in enumerate(uniques)
    ]
    plt.legend(handles=legend_handles, title="Malware Types")
    plt.tight_layout()
    plt.savefig(save_path / "true_labels_visualization.png", dpi=300)
    plt.show()


def plot_cluster_assignments(projected_embeddings: np.ndarray, cluster_labels: np.ndarray, save_path: Path) -> None:
    """Visualize the embeddings colored by DBSCAN cluster assignments."""
    plt.figure(figsize=(8, 8))
    unique_clusters = sorted([c for c in np.unique(cluster_labels) if c != -1])
    n_clusters = len(unique_clusters)
    # Corrected line for Matplotlib deprecation warning and TypeError:
    cmap = plt.colormaps.get_cmap("tab20") # Removed n_clusters argument here
    # Build a mapping from cluster label to a color (normalized to [0,1]).
    cluster_color_map = {cluster: cmap(i / max(1, n_clusters - 1)) for i, cluster in enumerate(unique_clusters)}

    # Assign colors to each sample.
    colors = [
        cluster_color_map[label] if label != -1 else (0.5, 0.5, 0.5, 0.7)
        for label in cluster_labels
    ]
    plt.scatter(
        projected_embeddings[:, 0],
        projected_embeddings[:, 1],
        c=colors,
        alpha=0.7,
        s=15,
    )
    plt.title(f"DBSCAN Clustering (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})\n{n_clusters} clusters found") # Updated title to use global constants
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    # Build legend for clusters.
    legend_handles = [
        mpatches.Patch(color=cluster_color_map[cluster], label=f"Cluster {cluster}")
        for cluster in unique_clusters
    ]
    # Add a legend entry for noise.
    legend_handles.append(mpatches.Patch(color=(0.5, 0.5, 0.5, 0.7), label="Noise"))
    plt.legend(handles=legend_handles, title="Clusters")
    plt.tight_layout()
    plt.savefig(save_path / "cluster_visualization.png", dpi=300)
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


def plot_pca_dbscan(embeddings: np.ndarray, eps: float = DBSCAN_EPS, min_samples: int = DBSCAN_MIN_SAMPLES) -> None: # Updated eps and min_samples here to use global constants
    """
    Visualize DBSCAN clustering on PCA-reduced embeddings.
    Also computes several evaluation metrics.
    """
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca = pca_2d.fit_transform(embeddings)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", n_jobs=NUM_CORES).fit(embeddings)
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    logger.info("PCA-DBSCAN: Estimated clusters: %d", n_clusters)
    logger.info("PCA-DBSCAN: Estimated noise points: %d", n_noise)

    try:
        silhouette = silhouette_score(embeddings, labels)
        logger.info("Silhouette Coefficient: %.3f", silhouette)
    except Exception as e:
        logger.warning("Silhouette score calculation failed: %s", e)

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    plt.figure(figsize=(12, 8))
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Black for noise
        class_member_mask = labels == k
        xy_core = X_pca[class_member_mask & core_samples_mask]
        plt.plot(
            xy_core[:, 0],
            xy_core[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
            alpha=0.7,
        )
        xy_non_core = X_pca[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy_non_core[:, 0],
            xy_non_core[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
            alpha=0.7,
        )
    plt.title(f"PCA-DBSCAN Clustering (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})\nEstimated clusters: {n_clusters} (Noise: {n_noise})") # Updated title to use global constants
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    legend_handles = [
        mpatches.Patch(color=tuple(colors[i]), label=f"Cluster {label}" if label != -1 else "Noise")
        for i, label in enumerate(unique_labels)
    ]
    plt.legend(handles=legend_handles, title="Cluster Assignments")
    plt.tight_layout()
    plt.show()


# --- Main Execution ---
def main() -> None:
    # Load and prepare data.
    embeddings_path = SAVED_MODELS_DIR / "mean_embedding_per_file.pkl"
    mean_embeddings = load_mean_embeddings(embeddings_path)
    embeddings, true_labels = prepare_data(mean_embeddings)

    # Optionally normalize embeddings.
    processed_embeddings = maybe_normalize_embeddings(embeddings, NORMALIZE_EMBEDDINGS)

    # Perform clustering.
    cluster_labels = perform_dbscan(processed_embeddings)
    evaluate_clustering(true_labels, cluster_labels)
    print_cluster_composition(true_labels, cluster_labels)

    # Reduce dimensions for 2D visualization (using UMAP).
    projected_embeddings = reduce_dimensions(processed_embeddings, method="umap", n_components=2)

    # Visualize true labels and cluster assignments.
    plot_true_labels(projected_embeddings, true_labels, SAVED_MODELS_DIR)
    plot_cluster_assignments(projected_embeddings, cluster_labels, SAVED_MODELS_DIR)

    # Additional 3D visualization.
    plot_3d_visualization(processed_embeddings, true_labels)

    # PCA-based DBSCAN clustering visualization.
    plot_pca_dbscan(processed_embeddings, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES) # Updated eps and min_samples here for consistency


if __name__ == "__main__":
    main()