import pickle
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

# --- Global Constants and Configuration ---
MODEL = "bert"
SAVED_MODELS_DIR = Path(f"../saved_models/{MODEL}/")
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
FILE_NAME = "mean_embedding_per_file2.pkl"
NORMALIZE_EMBEDDINGS: bool = False

# --- Logging Setup ---
logging.basicConfig(
    filename=SAVED_MODELS_DIR / 'svm.log',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='w'
)
logger = logging.getLogger(__name__)

# --- Data Loading and Preprocessing ---
def load_mean_embeddings(filepath: Path) -> Dict[Any, np.ndarray]:
    logger.info("Loading embeddings from %s", filepath)
    with filepath.open("rb") as f:
        data = pickle.load(f)
    logger.info("Loaded %d embeddings", len(data))
    return data

def prepare_data(mean_embeddings: Dict[Any, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    embeddings, labels = [], []
    for (label, _), vector in mean_embeddings.items():
        embeddings.append(vector)
        labels.append(label)
    return np.array(embeddings), np.array(labels)

def maybe_normalize_embeddings(embeddings: np.ndarray, normalize: bool) -> np.ndarray:
    if normalize:
        normalizer = Normalizer(norm="l2")
        logger.info("Applying L2 normalization.")
        return normalizer.fit_transform(embeddings)
    return embeddings

# --- SVM Classification ---
def train_svm_classifier(embeddings: np.ndarray, labels: np.ndarray) -> Tuple[SVC, LabelEncoder]:
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    classifier = SVC(kernel='linear', C=1.0)
    logger.info("Training SVM classifier.")
    classifier.fit(embeddings, encoded_labels)
    return classifier, label_encoder

def evaluate_classifier(classifier: SVC, embeddings: np.ndarray, labels: np.ndarray, label_encoder: LabelEncoder) -> None:
    encoded_labels = label_encoder.transform(labels)
    predictions = classifier.predict(embeddings)
    report = classification_report(encoded_labels, predictions, target_names=label_encoder.classes_)
    matrix = confusion_matrix(encoded_labels, predictions)
    logger.info("Classification Report:\n%s", report)
    logger.info("Confusion Matrix:\n%s", matrix)
    print(report)
    print(matrix)

# --- Visualization ---
def plot_pca_visualization(embeddings: np.ndarray, labels: np.ndarray, label_encoder: LabelEncoder) -> None:
    pca = PCA(n_components=2)
    transformed_embeddings = pca.fit_transform(embeddings)
    plt.figure(figsize=(8, 8))
    factorized_labels = label_encoder.transform(labels)
    scatter = plt.scatter(
        transformed_embeddings[:, 0],
        transformed_embeddings[:, 1],
        c=factorized_labels,
        cmap="Spectral",
        alpha=0.7,
        s=15,
    )
    plt.title("PCA Visualization of Malware Types")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter, label="Malware Type")
    plt.savefig(SAVED_MODELS_DIR / "svm_pca_visualization.png", dpi=300)
    plt.show()

# --- Main Execution ---
def main() -> None:
    embeddings_path = SAVED_MODELS_DIR / FILE_NAME
    mean_embeddings = load_mean_embeddings(embeddings_path)
    embeddings, labels = prepare_data(mean_embeddings)
    processed_embeddings = maybe_normalize_embeddings(embeddings, NORMALIZE_EMBEDDINGS)
    classifier, label_encoder = train_svm_classifier(processed_embeddings, labels)
    evaluate_classifier(classifier, processed_embeddings, labels, label_encoder)
    plot_pca_visualization(processed_embeddings, labels, label_encoder)

if __name__ == "__main__":
    main()
