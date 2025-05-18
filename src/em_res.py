import os
os.environ["OPENBLAS_NUM_THREADS"] = "32"

import pickle
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
import matplotlib.patches as mpatches
import umap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import optuna

# ─── Configuration ─────────────────────────────────────────────────────────────
MODEL = "one_hot"
SAVED_MODELS_DIR = Path(f"../saved_models/{MODEL}/")
EM_DB_PATH = SAVED_MODELS_DIR / "em" / "em.db"
EMBEDDING_FILE = SAVED_MODELS_DIR / "opcode_distribution_embeddings.pkl"
OUTPUT_DIR = SAVED_MODELS_DIR / "em"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=OUTPUT_DIR / "visualize_best.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w",
)
logger = logging.getLogger(__name__)

# ─── Data Loading & Preprocessing ───────────────────────────────────────────────
def load_embeddings(filepath: Path) -> Dict[Any, np.ndarray]:
    logger.info("Loading embeddings from %s", filepath)
    with filepath.open("rb") as f:
        return pickle.load(f)

def prepare_data(embeddings_dict: Dict[Any, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for (malware_type, _), vec in embeddings_dict.items():
        X.append(vec)
        y.append(malware_type)
    return np.vstack(X), np.array(y)

def maybe_normalize(X: np.ndarray, do_norm: bool) -> np.ndarray:
    if do_norm:
        logger.info("Applying L2 normalization")
        return Normalizer(norm="l2").fit_transform(X)
    logger.info("Skipping normalization")
    return X

# ─── Clustering & Evaluation ───────────────────────────────────────────────────
def cluster_em(X: np.ndarray, n_components: int) -> np.ndarray:
    logger.info("Clustering with GMM, n_components=%d", n_components)
    gmm = GaussianMixture(n_components=n_components, n_init=10)
    return gmm.fit_predict(X)

def evaluate(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    logger.info("Evaluating clustering")
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    comp = completeness_score(y_true, y_pred)
    vmes = v_measure_score(y_true, y_pred)
    logger.info("ARI: %.4f, NMI: %.4f, Hom.: %.4f, Comp.: %.4f, V-meas.: %.4f",
                ari, nmi, homo, comp, vmes)
    
    if len(set(y_pred)) > 1:
        try:
            sil = silhouette_score(X, y_pred, metric="cosine")
            logger.info("Silhouette: %.4f", sil)
        except Exception as e:
            logger.warning("Silhouette failed: %s", e)
    else:
        logger.warning("Not enough clusters for silhouette")

    # print composition
    for cluster in np.unique(y_pred):
        mask = (y_pred == cluster)
        types, counts = np.unique(y_true[mask], return_counts=True)
        logger.info("Cluster %s → %s", cluster, dict(zip(types, counts)))

# ─── Visualization ─────────────────────────────────────────────────────────────
def plot_2d(emb2d: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
            method: str, outdir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    # true
    fac, labels = pd.factorize(y_true)
    sc = ax1.scatter(emb2d[:,0], emb2d[:,1], c=fac, cmap="Spectral", s=15)
    ax1.set_title(f"{method} – True Labels")
    ax1.legend(handles=[
        mpatches.Patch(color=sc.cmap(i/(len(labels)-1)), label=lab)
        for i, lab in enumerate(labels)
    ], bbox_to_anchor=(1.05,1), loc="upper left")
    # pred
    clusters = sorted(c for c in np.unique(y_pred) if c!=-1)
    cmap = plt.get_cmap("tab20", lut=max(len(clusters),1))
    color_map = {c: cmap(i) for i,c in enumerate(clusters)}
    cols = [color_map.get(c,(0.5,0.5,0.5,0.7)) for c in y_pred]
    ax2.scatter(emb2d[:,0], emb2d[:,1], c=cols, s=15)
    ax2.set_title(f"{method} – Predicted Clusters")
    ax2.legend(handles=[
        mpatches.Patch(color=color_map.get(c,"gray"), label=f"Cluster {c}") 
        for c in clusters
    ] + [mpatches.Patch(color="gray", label="Noise")],
    bbox_to_anchor=(1.05,1), loc="upper left")
    
    plt.tight_layout()
    fp = outdir / f"cluster_{method.lower()}_2d.png"
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 2D plot → {fp}")

def plot_3d(emb3d: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
            method: str, outdir: Path) -> None:
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1,2,1, projection="3d")
    ax2 = fig.add_subplot(1,2,2, projection="3d")
    # true
    fac, labels = pd.factorize(y_true)
    for i, lab in enumerate(labels):
        mask = fac==i
        ax1.scatter(emb3d[mask,0], emb3d[mask,1], emb3d[mask,2],
                    label=lab, s=20, alpha=0.7)
    ax1.set_title(f"{method} – True Labels")
    ax1.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    # pred
    clusters = sorted(c for c in np.unique(y_pred) if c!=-1)
    cmap = plt.get_cmap("tab20", lut=max(len(clusters),1))
    color_map = {c: cmap(i) for i,c in enumerate(clusters)}
    for c in np.unique(y_pred):
        mask = (y_pred==c)
        col = color_map.get(c,"gray")
        ax2.scatter(emb3d[mask,0], emb3d[mask,1], emb3d[mask,2],
                    label=f"Cluster {c}" if c!=-1 else "Noise",
                    s=20, alpha=0.7, color=col)
    ax2.set_title(f"{method} – Predicted Clusters")
    ax2.legend(bbox_to_anchor=(1.05,1), loc="upper left")
    
    plt.tight_layout()
    fp = outdir / f"cluster_{method.lower()}_3d.png"
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D plot → {fp}")

# ─── Main: load best trial & visualize ─────────────────────────────────────────
def main():
    # load Optuna study
    study = optuna.load_study(
        study_name=f"em_{MODEL}",
        storage=f"sqlite:///{EM_DB_PATH}"
    )
    best = study.best_trial.params
    print("Best hyperparameters:", best)
    logger.info("Best params: %s", best)

    # load & preprocess
    data = load_embeddings(EMBEDDING_FILE)
    X, y = prepare_data(data)
    Xp = maybe_normalize(X, best["normalize"])

    # cluster & evaluate
    labels = cluster_em(Xp, n_components=best["n_components"])
    evaluate(Xp, y, labels)

    # 2D reductions
    for method, reducer in [
        ("PCA", PCA(n_components=2, random_state=42)),
        ("t-SNE", TSNE(n_components=2, n_jobs=-1, random_state=42)),
        ("UMAP", umap.UMAP(n_components=2, n_jobs=-1, random_state=42)),
    ]:
        emb2 = reducer.fit_transform(Xp)
        plot_2d(emb2, y, labels, method, OUTPUT_DIR)

    # 3D reductions
    for method, reducer in [
        ("PCA", PCA(n_components=3, random_state=42)),
        ("t-SNE", TSNE(n_components=3, n_jobs=-1, random_state=42)),
        ("UMAP", umap.UMAP(n_components=3, n_jobs=-1, random_state=42)),
    ]:
        emb3 = reducer.fit_transform(Xp)
        plot_3d(emb3, y, labels, method, OUTPUT_DIR)

if __name__ == "__main__":
    main()