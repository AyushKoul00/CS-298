#!/usr/bin/env python3
"""
Train GloVe models for malware opcode sequences and compute mean embeddings per file.

This implementation uses the glove-python package. It builds a co-occurrence
matrix from the corpus of malware opcode files, trains a GloVe model on it,
and then computes a mean embedding for each file. If a file is large (i.e. has
more opcodes than CHUNK_SIZE), it is split into chunks; the embedding for each
chunk is computed and then averaged to yield the final file embedding.
"""

import logging
import pickle
from pathlib import Path
from multiprocessing import cpu_count
from typing import Dict, List, Tuple

import numpy as np
from glove import Corpus, Glove

# ----------------------- Constants and Hyperparameters -----------------------

# Data and saving configuration
MALWARE_DIR = Path("../dataset/")  # Directory containing malware type folders
SAVED_MODELS_DIR = Path("../saved_models/glove/")
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = SAVED_MODELS_DIR / "process.log"

# Each subfolder in MALWARE_DIR is a malware type
MALWARE_TYPES = [folder for folder in MALWARE_DIR.iterdir() if folder.is_dir()]
# Set to -1 to read all files, or set a maximum number of files per folder
MAX_SAMPLES_PER_TYPE = [-1] * len(MALWARE_TYPES)

# Hyperparameters for GloVe training
EMBEDDING_SIZE = 128       # Dimension of the embeddings
WINDOW_SIZE = 5            # Context window size
EPOCHS = 30                # Number of training epochs
LEARNING_RATE = 0.05       # Learning rate for training
NUM_CORES = cpu_count()    # Number of CPU cores to use

# Processing configuration
CHUNK_SIZE = 1000          # Number of opcodes per chunk for large files

# ------------------------------ Logging Setup ----------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="w")
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------- Function Definitions ---------------------------

def build_corpus(filepaths: List[Path]) -> List[List[str]]:
    """
    Build a corpus from the given filepaths.
    Each file is read and converted into a list of opcodes.

    Args:
        filepaths (List[Path]): List of file paths to process.

    Returns:
        List[List[str]]: A list of opcode lists.
    """
    corpus = []
    for filepath in filepaths:
        try:
            with filepath.open("r") as file:
                opcodes = [line.strip() for line in file if line.strip()]
                corpus.append(opcodes)
            logger.info(f"Built corpus for file: {filepath.name}")
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
    return corpus


def process_malware_type(
    malware_dir: Path, max_samples: int
) -> Tuple[Glove, List[Path], List[List[str]]]:
    """
    Process a malware type folder:
        - Read opcode files.
        - Build corpus.
        - Train a GloVe model.

    Args:
        malware_dir (Path): Path to the malware type folder.
        max_samples (int): Maximum number of samples to process (-1 for all).

    Returns:
        Tuple[Glove, List[Path], List[List[str]]]:
            The trained GloVe model, list of processed filepaths, and the corpus.
    """
    filepaths = list(malware_dir.glob("*.txt"))
    if 0 <= max_samples < len(filepaths):
        filepaths = filepaths[:max_samples]

    logger.info(f"Processing {len(filepaths)} files in malware type '{malware_dir.name}'")
    corpus_data = build_corpus(filepaths)

    # Create the co-occurrence matrix using the Corpus class from glove-python.
    logger.info("Building co-occurrence matrix...")
    corpus_obj = Corpus()
    corpus_obj.fit(corpus_data, window=WINDOW_SIZE)

    # Initialize and train the GloVe model.
    logger.info("Training GloVe model...")
    glove_model = Glove(no_components=EMBEDDING_SIZE, learning_rate=LEARNING_RATE)
    glove_model.fit(corpus_obj.matrix, epochs=EPOCHS, no_threads=NUM_CORES, verbose=True)
    glove_model.add_dictionary(corpus_obj.dictionary)
    logger.info("GloVe model training complete.")

    return glove_model, filepaths, corpus_data


def compute_mean_embeddings(
    model: Glove, filepaths: List[Path], corpus: List[List[str]], malware_type: str
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Compute the mean embedding for each file in the corpus.
    For files that exceed CHUNK_SIZE opcodes, break the file into chunks,
    compute the mean embedding for each chunk, and then average these
    to obtain the file-level embedding.

    Args:
        model (Glove): Trained GloVe model.
        filepaths (List[Path]): List of file paths.
        corpus (List[List[str]]): List of opcode lists.
        malware_type (str): The malware type (used as a key).

    Returns:
        Dict[Tuple[str, str], np.ndarray]: Dictionary with keys as (malware_type, filename)
            and values as mean embeddings.
    """
    mean_embeddings: Dict[Tuple[str, str], np.ndarray] = {}

    for filepath, opcodes in zip(filepaths, corpus):
        def compute_chunk_mean(tokens: List[str]) -> np.ndarray:
            """
            Compute the mean embedding for a list of tokens.
            Tokens not found in the model's dictionary are skipped.
            """
            embeddings = []
            for token in tokens:
                if token in model.dictionary:
                    idx = model.dictionary[token]
                    embeddings.append(model.word_vectors[idx])
            if embeddings:
                return np.mean(embeddings, axis=0)
            else:
                return np.zeros(EMBEDDING_SIZE)

        # For large files, process in chunks.
        if len(opcodes) > CHUNK_SIZE:
            chunk_embeddings = []
            for i in range(0, len(opcodes), CHUNK_SIZE):
                chunk = opcodes[i : i + CHUNK_SIZE]
                chunk_mean = compute_chunk_mean(chunk)
                chunk_embeddings.append(chunk_mean)
            mean_embedding = np.mean(chunk_embeddings, axis=0)
            logger.info(f"Computed chunked mean embedding for file: {filepath.name}")
        else:
            mean_embedding = compute_chunk_mean(opcodes)
            logger.info(f"Computed mean embedding for file: {filepath.name}")

        mean_embeddings[(malware_type, filepath.name)] = mean_embedding

    return mean_embeddings


def main() -> None:
    """
    Main function to process malware types, train GloVe models,
    compute mean embeddings, and save results.
    """
    logger.info("Starting processing of malware opcode files.")
    all_mean_embeddings: Dict[Tuple[str, str], np.ndarray] = {}

    for malware_dir, max_samples in zip(MALWARE_TYPES, MAX_SAMPLES_PER_TYPE):
        malware_type = malware_dir.name
        logger.info(f"Processing malware type: {malware_type}")
        try:
            glove_model, filepaths, corpus_data = process_malware_type(malware_dir, max_samples)

            # Save the Glove model
            model_filename = SAVED_MODELS_DIR / f"{malware_type}.model"
            glove_model.save(str(model_filename))
            logger.info(f"Saved GloVe model for '{malware_type}' to {model_filename}")

            # Compute and collect mean embeddings
            mean_embeddings = compute_mean_embeddings(glove_model, filepaths, corpus_data, malware_type)
            all_mean_embeddings.update(mean_embeddings)
        except Exception as e:
            logger.error(f"Error processing malware type '{malware_type}': {e}")

    # Save the mean embeddings to a pickle file
    pickle_filepath = SAVED_MODELS_DIR / "mean_embedding_per_file.pkl"
    try:
        with pickle_filepath.open("wb") as f:
            pickle.dump(all_mean_embeddings, f)
        logger.info(f"Saved mean embeddings to pickle file: {pickle_filepath}")
    except Exception as e:
        logger.error(f"Failed to save mean embeddings: {e}")

    logger.info("Processing complete.")


if __name__ == "__main__":
    main()
