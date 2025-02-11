import pickle
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import cudf  # GPU DataFrame for cuML
from cuml.cluster import DBSCAN
from sklearn.preprocessing import Normalizer
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, homogeneity_score, completeness_score,
                             v_measure_score)

# --- Global Constants and Configuration ---
SAVED_MODELS_DIR = Path("../saved_models/word2vec/")
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_SIZE = 128  # Dimension of the embeddings (vectors)
NORMALIZE_EMBEDDINGS: bool = False  # Option to L2-normalize embeddings

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Data Loading and Preprocessing ---
def load_mean_embeddings(filepath: Path) -> Dict[Any, np.ndarray]:
    """Load mean embeddings from a pickle file."""
    logger.info("Loading embeddings from %s", filepath)
    with filepath.open("rb") as f:
        data = pickle.load(f)
    return data


def prepare_data(mean_embeddings: Dict[Any, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare embeddings and their associated true labels.
    The expected key in mean_embeddings is a tuple where the first element is the label.
    """
    embeddings: List[np.ndarray] = []
    labels: List[Any] = []
    for (label, _), vector in mean_embeddings.items():
        embeddings.append(vector)
        labels.append(label)
    return np.array(embeddings), np.array(labels)


def maybe_normalize_embeddings(embeddings: np.ndarray, normalize: bool) -> np.ndarray:
    """Optionally apply L2 normalization to embeddings."""
    if normalize:
        logger.info("Normalizing embeddings with L2 norm.")
        return Normalizer(norm="l2").fit_transform(embeddings)
    logger.info("Skipping normalization of embeddings.")
    return embeddings


# --- Clustering ---
def perform_dbscan(embeddings: np.ndarray, eps: float = 0.25, min_samples: int = 5) -> np.ndarray:
    """
    Perform DBSCAN clustering using cosine distance with cuML.
    Converts the embeddings to a cuDF DataFrame before clustering.
    """
    logger.info("Running cuML DBSCAN (eps=%.2f, min_samples=%d, metric='cosine')", eps, min_samples)
    embeddings_df = cudf.DataFrame(embeddings)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine", algorithm="brute", verbose=True)
    labels = dbscan.fit_predict(embeddings_df)
    return labels


# --- Evaluation ---
def evaluate_clustering(true_labels: np.ndarray, cluster_labels: np.ndarray, embeddings: np.ndarray) -> None:
    """
    Evaluate clustering performance using:
    - Adjusted Rand Index (ARI)
    - Normalized Mutual Information (NMI)
    - Homogeneity, Completeness, and V-Measure
    - Silhouette Score (computed on the embeddings)
    """
    ari = adjusted_rand_score(true_labels, cluster_labels)
    nmi = normalized_mutual_info_score(true_labels, cluster_labels)
    homogeneity = homogeneity_score(true_labels, cluster_labels)
    completeness = completeness_score(true_labels, cluster_labels)
    v_measure = v_measure_score(true_labels, cluster_labels)

    logger.info("Adjusted Rand Index: %.4f", ari)
    logger.info("Normalized Mutual Information: %.4f", nmi)
    logger.info("Homogeneity: %.4f", homogeneity)
    logger.info("Completeness: %.4f", completeness)
    logger.info("V-Measure: %.4f", v_measure)

    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) > 1 and not (len(unique_labels) == 1 and unique_labels[0] == -1):
        sil_score = silhouette_score(embeddings, cluster_labels, metric="cosine")
        logger.info("Silhouette Score: %.4f", sil_score)
    else:
        logger.info("Silhouette Score: Not computed (only one cluster found).")


# --- Main Execution ---
if __name__ == "__main__":
    embedding_file = SAVED_MODELS_DIR / "mean_embeddings.pkl"
    
    # Load and prepare embeddings.
    mean_embeddings = load_mean_embeddings(embedding_file)
    embeddings, true_labels = prepare_data(mean_embeddings)
    
    # Optionally normalize the embeddings.
    embeddings = maybe_normalize_embeddings(embeddings, NORMALIZE_EMBEDDINGS)
    
    # Perform DBSCAN clustering.
    cluster_labels = perform_dbscan(embeddings, eps=0.25, min_samples=5)
    # Convert cluster labels to NumPy array if needed.
    if hasattr(cluster_labels, "to_array"):
        cluster_labels = cluster_labels.to_array()
    
    logger.info("DBSCAN cluster labels: %s", cluster_labels)
    print("Final cluster labels:")
    print(cluster_labels)
    
    # Evaluate clustering performance.
    evaluate_clustering(true_labels, cluster_labels, embeddings)