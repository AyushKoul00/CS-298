#!/usr/bin/env python3
"""
Train Doc2Vec models for malware opcode sequences and compute embeddings per file.
Each file is treated as a single document.
"""

import logging
import pickle
from pathlib import Path
from multiprocessing import cpu_count
from typing import Dict, List, Tuple

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Constants and configuration
MALWARE_DIR = Path("../dataset/")  # Directory containing malware type folders
SAVED_MODELS_DIR = Path("../saved_models/doc2vec/")
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging: output will be written to a log file "doc2vec.log", overwritten on each run.
logging.basicConfig(
    filename=SAVED_MODELS_DIR / "doc2vec.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"  # Override log file on each run
)

# Each subfolder in MALWARE_DIR is a malware type
MALWARE_TYPES = [folder for folder in MALWARE_DIR.iterdir() if folder.is_dir()]
# Set to -1 to read all files, or set a maximum number of files per folder
MAX_SAMPLES_PER_TYPE = [-1] * len(MALWARE_TYPES)
EMBEDDING_SIZE = 128  # Dimension of the embeddings (vectors)
NUM_CORES = cpu_count()


def build_corpus(filepaths: List[Path]) -> Tuple[List[TaggedDocument], List[str]]:
    """
    Build a corpus of TaggedDocuments for Doc2Vec from the given filepaths.
    Each file is read and treated as a single document with a unique tag.

    Args:
        filepaths (List[Path]): List of file paths to process.

    Returns:
        Tuple[List[TaggedDocument], List[str]]:
            - A list of TaggedDocuments.
            - A list of file identifiers (filenames) corresponding to each document.
    """
    corpus: List[TaggedDocument] = []
    file_ids: List[str] = []

    for filepath in filepaths:
        with filepath.open("r") as file:
            # Read opcodes (one per line) and remove empty lines.
            opcodes = [line.strip() for line in file if line.strip()]
        file_id = filepath.name  # Use file name as unique tag
        file_ids.append(file_id)
        corpus.append(TaggedDocument(words=opcodes, tags=[file_id]))
    return corpus, file_ids


def process_malware_type(
    malware_dir: Path, max_samples: int
) -> Tuple[Doc2Vec, List[Path], List[str]]:
    """
    Process a malware type folder:
      - Read opcode files.
      - Build corpus.
      - Train a Doc2Vec model using best practices:
          * Build the vocabulary.
          * Train the model explicitly.
    
    Args:
        malware_dir (Path): Path to the malware type folder.
        max_samples (int): Maximum number of samples to process (-1 for all).

    Returns:
        Tuple[Doc2Vec, List[Path], List[str]]:
          The trained Doc2Vec model, list of processed filepaths, and the list of file identifiers.
    """
    filepaths = list(malware_dir.glob("*.txt"))
    if 0 <= max_samples < len(filepaths):
        filepaths = filepaths[:max_samples]

    corpus, file_ids = build_corpus(filepaths)

    # Initialize the Doc2Vec model without passing the corpus directly.
    model = Doc2Vec(
        vector_size=EMBEDDING_SIZE,
        window=5,
        min_count=1,
        workers=NUM_CORES,
        epochs=20  # Adjust epochs as needed
    )

    # Build the vocabulary from the corpus (recommended best practice)
    model.build_vocab(corpus)
    logging.info("Vocabulary built for %s with %d documents.", malware_dir.name, len(corpus))

    # Train the model explicitly on the corpus
    model.train(corpus, total_examples=len(corpus), epochs=model.epochs)
    logging.info("Training complete for %s.", malware_dir.name)

    return model, filepaths, file_ids


def compute_embeddings(
    model: Doc2Vec, file_ids: List[str], malware_type: str
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Compute the embedding for each file in the corpus.

    Args:
        model (Doc2Vec): Trained Doc2Vec model.
        file_ids (List[str]): List of file identifiers.
        malware_type (str): The malware type (used as a key).

    Returns:
        Dict[Tuple[str, str], np.ndarray]: Dictionary with keys as (malware_type, filename)
            and values as embeddings.
    """
    embeddings: Dict[Tuple[str, str], np.ndarray] = {}
    for file_id in file_ids:
        vector = model.dv[file_id]
        embeddings[(malware_type, file_id)] = vector
    return embeddings


def main() -> None:
    """
    Main function to process malware types, train Doc2Vec models,
    compute embeddings, and save results.
    """
    all_embeddings: Dict[Tuple[str, str], np.ndarray] = {}

    for malware_dir, max_samples in zip(MALWARE_TYPES, MAX_SAMPLES_PER_TYPE):
        malware_type = malware_dir.name
        print(f"Processing malware type: {malware_type:<20}", end="\t", flush=True)

        # Process the current malware folder: build corpus, train model, etc.
        model, filepaths, file_ids = process_malware_type(malware_dir, max_samples)

        # Save the trained Doc2Vec model
        model_filename = SAVED_MODELS_DIR / f"{malware_type}.model"
        model.save(str(model_filename))
        logging.info("Model saved for %s at %s.", malware_type, model_filename)

        # Compute and collect embeddings for each file
        embeddings = compute_embeddings(model, file_ids, malware_type)
        all_embeddings.update(embeddings)
        print("(Done).", flush=True)

    # Save all embeddings to a pickle file for later use
    pickle_filepath = SAVED_MODELS_DIR / "embedding_per_file.pkl"
    with pickle_filepath.open("wb") as f:
        pickle.dump(all_embeddings, f)
    print("Saved embeddings to pickle file.", flush=True)


if __name__ == "__main__":
    main()
