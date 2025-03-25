#!/usr/bin/env python3
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Global Constants and Configuration ---
MODEL = "word2vec"
SAVED_MODELS_DIR = Path(f"../saved_models/{MODEL}/")
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
FILE_NAME = "mean_embedding_per_file.pkl"

# --- Logging Setup ---
logging.basicConfig(
    filename=SAVED_MODELS_DIR / 'random_forest.log',
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='w'
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
    """Prepare embeddings and their associated true labels."""
    embeddings: List[np.ndarray] = []
    labels: List[Any] = []
    for (malware_type, _), vector in mean_embeddings.items():
        embeddings.append(vector)
        labels.append(malware_type)
    return np.array(embeddings), np.array(labels)

# --- Classification ---
def train_random_forest(embeddings: np.ndarray, labels: np.ndarray):
    """Train a Random Forest classifier and evaluate its performance."""
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info("Random Forest Accuracy: %.4f", accuracy)
    logger.info("Classification Report:\n%s", classification_report(y_test, y_pred))
    logger.info("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
    
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# --- Main Execution ---
def main() -> None:
    embeddings_path = SAVED_MODELS_DIR / FILE_NAME
    mean_embeddings = load_mean_embeddings(embeddings_path)
    embeddings, true_labels = prepare_data(mean_embeddings)
    
    train_random_forest(embeddings, true_labels)

if __name__ == "__main__":
    main()
