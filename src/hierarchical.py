import os
os.environ["OPENBLAS_NUM_THREADS"] = "32"

import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
import matplotlib.patches as mpatches
import umap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from multiprocessing import cpu_count

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import Normalizer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import optuna

# --- Global Configuration ---
MODEL = "distilbert"
MALWARE_DIR = Path("../dataset/")
SAVED_MODELS_DIR = Path(f"../saved_models/{MODEL}/")
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# All outputs are stored in the "hierarchical" folder.
OUTPUT_DIR = SAVED_MODELS_DIR / "hierarchical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_NAME = "mean_embedding_per_file2.pkl"
NORMALIZE_EMBEDDINGS = False  # Option to apply L2 normalization

# --- Logging Setup ---
logging.basicConfig(
    filename=OUTPUT_DIR / "hierarchical.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)
logger = logging.getLogger(__name__)

# --- Data Loading and Preprocessing ---
def load_mean_embeddings(filepath: Path) -> Dict[Any, np.ndarray]:
    logger.info("Loading embeddings from %s", filepath)
    with filepath.open("rb") as f:
        data = pickle.load(f)
    logger.info("Loaded %d embeddings from %s", len(data), filepath)
    return data

def prepare_data(mean_embeddings: Dict[Any, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    embeddings, labels = [], []
    for (malware_type, _), vector in mean_embeddings.items():
        embeddings.append(vector)
        labels.append(malware_type)
    return np.array(embeddings), np.array(labels)

def maybe_normalize_embeddings(embeddings: np.ndarray, normalize: bool) -> np.ndarray:
    if normalize:
        normalizer = Normalizer(norm="l2")
        logger.info("Normalizing embeddings with L2 norm.")
        return normalizer.fit_transform(embeddings)
    logger.info("Skipping normalization of embeddings.")
    return embeddings

# --- Clustering with Hierarchical Clustering ---
def perform_hierarchical(embeddings: np.ndarray, n_clusters: int, linkage: str) -> np.ndarray:
    logger.info("Running AgglomerativeClustering with n_clusters=%d, linkage=%s", n_clusters, linkage)
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    return hierarchical.fit_predict(embeddings)

# --- Evaluation ---
def evaluate_clustering(embeddings: np.ndarray, true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    if len(np.unique(true_labels)) > 1:
        ari = adjusted_rand_score(true_labels, cluster_labels)
        nmi = normalized_mutual_info_score(true_labels, cluster_labels)
        homo = homogeneity_score(true_labels, cluster_labels)
        comp = completeness_score(true_labels, cluster_labels)
        vmeasure = v_measure_score(true_labels, cluster_labels)
        logger.info("ARI: %.4f, NMI: %.4f, Homogeneity: %.4f, Completeness: %.4f, V-measure: %.4f",
                    ari, nmi, homo, comp, vmeasure)
        unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]
        if len(unique_clusters) > 1:
            try:
                silhouette = silhouette_score(embeddings, cluster_labels, metric="euclidean", ensure_all_finite=True)
                logger.info("Silhouette Coefficient: %.4f", silhouette)
            except Exception as e:
                logger.warning("Silhouette score calculation failed: %s", e)
        else:
            logger.warning("Not enough clusters for silhouette score.")
        return ari
    logger.warning("Not enough unique true labels for evaluation.")
    return 0.0

def print_cluster_composition(true_labels: np.ndarray, cluster_labels: np.ndarray) -> None:
    unique_clusters = np.unique(cluster_labels)
    logger.info("Cluster composition:")
    for cluster in unique_clusters:
        mask = cluster_labels == cluster
        name = f"Cluster {cluster}" if cluster != -1 else "Noise"
        types, counts = np.unique(true_labels[mask], return_counts=True)
        logger.info(f"{name}: {dict(zip(types, counts))}")

# --- Dimensionality Reduction ---
def reduce_dimensions(embeddings: np.ndarray, method: str = "umap", n_components: int = 2) -> np.ndarray:
    logger.info("Reducing dimensions using %s to %d components.", method, n_components)
    if method.lower() == "umap":
        reducer = umap.UMAP(n_components=n_components, n_jobs=-1)
    elif method.lower() == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError("Method must be either 'umap' or 'pca'")
    return reducer.fit_transform(embeddings)

# --- Combined 2D Visualization ---
def plot_combined_2d(embeddings_2d: np.ndarray, true_labels: np.ndarray,
                     cluster_labels: np.ndarray, method: str, save_path: Path,
                     model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ax_true = axes[0]
    factorized, uniques = pd.factorize(true_labels)
    scatter = ax_true.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                              c=factorized, cmap="Spectral", alpha=0.7, s=15)
    ax_true.set_title(f"{method}: True Labels (Model: {model_name})")
    ax_true.set_xlabel("Component 1")
    ax_true.set_ylabel("Component 2")
    true_handles = [mpatches.Patch(color=scatter.cmap(i / max(1, len(uniques)-1)),
                                   label=f"Type: {label}") for i, label in enumerate(uniques)]
    ax_true.legend(handles=true_handles, title="Malware Types", loc="upper left", bbox_to_anchor=(1.05, 1))
    
    ax_pred = axes[1]
    unique_clusters = sorted([c for c in np.unique(cluster_labels) if c != -1])
    cmap_pred = plt.get_cmap("tab20", lut=max(len(unique_clusters), 1))
    cluster_color_map = {cluster: cmap_pred(i) for i, cluster in enumerate(unique_clusters)}
    colors = [cluster_color_map[label] if label != -1 else (0.5, 0.5, 0.5, 0.7)
              for label in cluster_labels]
    ax_pred.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                    c=colors, alpha=0.7, s=15)
    ax_pred.set_title(f"{method}: Predicted Clusters (Model: {model_name})")
    ax_pred.set_xlabel("Component 1")
    ax_pred.set_ylabel("Component 2")
    pred_handles = [mpatches.Patch(color=cluster_color_map[cluster], label=f"Cluster {cluster}")
                    for cluster in unique_clusters]
    pred_handles.append(mpatches.Patch(color=(0.5, 0.5, 0.5, 0.7), label="Noise"))
    ax_pred.legend(handles=pred_handles, title="Clusters", loc="upper left", bbox_to_anchor=(1.05, 1))
    
    fig.tight_layout()
    filename = save_path / f"combined_true_pred_2d_{method.lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined 2D visualization as {filename}")

# --- Combined 3D Visualization ---
def plot_combined_3d(embeddings_3d: np.ndarray, true_labels: np.ndarray,
                     cluster_labels: np.ndarray, method: str, save_path: Path,
                     model_name: str) -> None:
    fig = plt.figure(figsize=(16, 8))
    ax_true = fig.add_subplot(1, 2, 1, projection="3d")
    ax_pred = fig.add_subplot(1, 2, 2, projection="3d")
    
    factorized, uniques = pd.factorize(true_labels)
    for i, label in enumerate(uniques):
        mask = factorized == i
        ax_true.scatter(embeddings_3d[mask, 0], embeddings_3d[mask, 1], embeddings_3d[mask, 2],
                        label=f"Type: {label}", alpha=0.7, s=20)
    ax_true.set_title(f"{method}: True Labels (Model: {model_name})")
    ax_true.set_xlabel("Component 1")
    ax_true.set_ylabel("Component 2")
    ax_true.set_zlabel("Component 3")
    ax_true.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    
    unique_clusters = sorted([c for c in np.unique(cluster_labels) if c != -1])
    cmap_pred = plt.get_cmap("tab20", lut=max(len(unique_clusters), 1))
    cluster_color_map = {cluster: cmap_pred(i) for i, cluster in enumerate(unique_clusters)}
    for cluster in np.unique(cluster_labels):
        mask = cluster_labels == cluster
        label_str = f"Cluster {cluster}" if cluster != -1 else "Noise"
        color_val = cluster_color_map.get(cluster, "gray") if cluster != -1 else "gray"
        ax_pred.scatter(embeddings_3d[mask, 0], embeddings_3d[mask, 1], embeddings_3d[mask, 2],
                        label=label_str, alpha=0.7, s=20, color=color_val)
    ax_pred.set_title(f"{method}: Predicted Clusters (Model: {model_name})")
    ax_pred.set_xlabel("Component 1")
    ax_pred.set_ylabel("Component 2")
    ax_pred.set_zlabel("Component 3")
    ax_pred.legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    
    fig.tight_layout()
    filename = save_path / f"combined_true_pred_3d_{method.lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved combined 3D visualization as {filename}")

# --- Optuna Objective Function ---
def objective(trial: optuna.Trial) -> float:
    n_clusters = trial.suggest_int("n_clusters", 2, 20)
    linkage = trial.suggest_categorical("linkage", ["ward", "complete", "average", "single"])
    normalize = trial.suggest_categorical("normalize", [True, False])
    
    data = load_mean_embeddings(SAVED_MODELS_DIR / FILE_NAME)
    embeddings, true_labels = prepare_data(data)
    processed = maybe_normalize_embeddings(embeddings, normalize)
    
    cluster_labels = perform_hierarchical(processed, n_clusters=n_clusters, linkage=linkage)
    ari = adjusted_rand_score(true_labels, cluster_labels)
    return ari

# --- Main Execution ---
def main() -> None:
    study = optuna.create_study(direction="maximize", study_name=f"hierarchical_{MODEL}",
                                storage=f"sqlite:///{OUTPUT_DIR / 'hierarchical.db'}")
    study.optimize(objective, n_trials=50)
    best_params = study.best_trial.params
    logger.info("Best hyperparameters: %s", best_params)
    print("Best hyperparameters:", best_params)
    
    data = load_mean_embeddings(SAVED_MODELS_DIR / FILE_NAME)
    embeddings, true_labels = prepare_data(data)
    processed = maybe_normalize_embeddings(embeddings, best_params["normalize"])
    
    cluster_labels = perform_hierarchical(processed, n_clusters=best_params["n_clusters"], linkage=best_params["linkage"])
    evaluate_clustering(processed, true_labels, cluster_labels)
    print_cluster_composition(true_labels, cluster_labels)
    
    # Combined 2D visualizations
    pca_2d = PCA(n_components=2, random_state=42)
    emb_pca = pca_2d.fit_transform(processed)
    plot_combined_2d(emb_pca, true_labels, cluster_labels, method="PCA", save_path=OUTPUT_DIR, model_name=MODEL)
    
    tsne_2d = TSNE(n_components=2, n_jobs=-1)
    emb_tsne = tsne_2d.fit_transform(processed)
    plot_combined_2d(emb_tsne, true_labels, cluster_labels, method="t-SNE", save_path=OUTPUT_DIR, model_name=MODEL)
    
    umap_2d = umap.UMAP(n_components=2, n_jobs=-1)
    emb_umap = umap_2d.fit_transform(processed)
    plot_combined_2d(emb_umap, true_labels, cluster_labels, method="UMAP", save_path=OUTPUT_DIR, model_name=MODEL)
    
    # Combined 3D visualizations
    pca_3d = PCA(n_components=3, random_state=42)
    emb_pca_3d = pca_3d.fit_transform(processed)
    plot_combined_3d(emb_pca_3d, true_labels, cluster_labels, method="PCA", save_path=OUTPUT_DIR, model_name=MODEL)
    
    tsne_3d = TSNE(n_components=3, n_jobs=-1)
    emb_tsne_3d = tsne_3d.fit_transform(processed)
    plot_combined_3d(emb_tsne_3d, true_labels, cluster_labels, method="t-SNE", save_path=OUTPUT_DIR, model_name=MODEL)
    
    umap_3d = umap.UMAP(n_components=3, n_jobs=-1)
    emb_umap_3d = umap_3d.fit_transform(processed)
    plot_combined_3d(emb_umap_3d, true_labels, cluster_labels, method="UMAP", save_path=OUTPUT_DIR, model_name=MODEL)

if __name__ == "__main__":
    main()
