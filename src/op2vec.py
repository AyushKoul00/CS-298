#!/usr/bin/env python3
"""
Train Op2Vec models for malware opcode sequences and compute mean embeddings per file,
handling large files by chunking.
"""

import os
import pickle
from pathlib import Path
from multiprocessing import cpu_count
from typing import Dict, List, Tuple

import numpy as np
from gensim.models import Word2Vec  # Op2Vec uses Word2Vec under the hood

# Constants and configuration
MALWARE_DIR = Path("../dataset/")  # Directory containing malware type folders
SAVED_MODELS_DIR = Path("../saved_models/op2vec/")
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Malware types and sample limits
MALWARE_TYPES = [folder for folder in MALWARE_DIR.iterdir() if folder.is_dir()]
MAX_SAMPLES_PER_TYPE = [-1] * len(MALWARE_TYPES)
EMBEDDING_SIZE = 128  # Dimension of the embeddings (vectors)
NUM_CORES = cpu_count()
CHUNK_SIZE = 1000  # Number of opcodes per chunk


def build_corpus(filepaths: List[Path]) -> List[List[str]]:
    """
    Build a corpus from the given filepaths, handling large files by chunking.
    Each file is read, split into chunks, and converted into a list of opcode lists.

    Args:
        filepaths (List[Path]): List of file paths to process.

    Returns:
        List[List[str]]: A list of opcode lists (chunks).
    """
    corpus = []
    for filepath in filepaths:
        with filepath.open("r") as file:
            opcodes = [line.strip() for line in file if line.strip()]
            for i in range(0, len(opcodes), CHUNK_SIZE):
                chunk = opcodes[i:i + CHUNK_SIZE]
                corpus.append(chunk)  # Each chunk is a separate "sentence"
    return corpus


def process_malware_type(
    malware_dir: Path, max_samples: int
) -> Tuple[Word2Vec, List[Path], List[List[str]]]:
    """
    Process a malware type folder:
        - Read opcode files, chunk them, and build the corpus.
        - Train an Op2Vec model (using Word2Vec).

    Args:
        malware_dir (Path): Path to the malware type folder.
        max_samples (int): Maximum number of samples to process (-1 for all).

    Returns:
        Tuple[Word2Vec, List[Path], List[List[str]]]: The trained Op2Vec model,
        list of processed filepaths, and the corpus.
    """
    filepaths = list(malware_dir.glob("*.txt"))
    if 0 <= max_samples < len(filepaths):
        filepaths = filepaths[:max_samples]

    corpus = build_corpus(filepaths)

    model = Word2Vec(  # Op2Vec is implemented using Word2Vec
        sentences=corpus,
        vector_size=EMBEDDING_SIZE,
        window=5,
        min_count=1,
        workers=NUM_CORES,
        epochs=40  # Increase the number of epochs
    )

    return model, filepaths, corpus


def compute_mean_embeddings(
    model: Word2Vec, filepaths: List[Path], malware_dir: Path  # Add malware_dir as argument
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Compute the mean embedding for each file by averaging chunk embeddings.

    Args:
        model (Word2Vec): Trained Op2Vec model.
        filepaths (List[Path]): List of file paths.
        malware_dir (Path): Path to the current malware type directory. # Added

    Returns:
        Dict[Tuple[str, str], np.ndarray]: Dictionary with keys as (malware_type, filename)
            and values as mean embeddings.
    """
    mean_embeddings = {}
    for filepath in filepaths:
        file_embeddings = []
        opcodes = [line.strip() for line in filepath.open() if line.strip()]
        for i in range(0, len(opcodes), CHUNK_SIZE):
            chunk = opcodes[i:i + CHUNK_SIZE]
            valid_embeddings = [model.wv[opcode] for opcode in chunk if opcode in model.wv]
            if valid_embeddings:
                chunk_embedding = np.mean(valid_embeddings, axis=0)
                file_embeddings.append(chunk_embedding)

        if file_embeddings:
            mean_embedding = np.mean(file_embeddings, axis=0)
            mean_embeddings[(malware_dir.name, filepath.name)] = mean_embedding  # Use malware_dir.name
    return mean_embeddings


def main() -> None:
    all_mean_embeddings: Dict[Tuple[str, str], np.ndarray] = {}

    for malware_dir, max_samples in zip(MALWARE_TYPES, MAX_SAMPLES_PER_TYPE):
        malware_type = malware_dir.name
        print(f"Processing malware type: {malware_type:<20}", end="\t", flush=True)

        model, filepaths, corpus = process_malware_type(malware_dir, max_samples)

        model_filename = SAVED_MODELS_DIR / f"{malware_type}.model"
        model.save(str(model_filename))

        mean_embeddings = compute_mean_embeddings(model, filepaths, malware_dir)  # Pass malware_dir
        all_mean_embeddings.update(mean_embeddings)
        print("(Done).", flush=True)

    pickle_filepath = SAVED_MODELS_DIR / "mean_embedding_per_file.pkl"
    with pickle_filepath.open("wb") as f:
        pickle.dump(all_mean_embeddings, f)
    print("Saved mean embeddings to pickle file.", flush=True)


if __name__ == "__main__":
    main()