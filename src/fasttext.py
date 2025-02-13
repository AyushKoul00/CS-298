import os
import pickle
from pathlib import Path
from multiprocessing import cpu_count
from typing import Dict, List, Tuple

import numpy as np
from gensim.models import FastText

# Constants and configuration
MALWARE_DIR = Path("../dataset/")  # Directory containing malware type folders
SAVED_MODELS_DIR = Path("../saved_models/fasttext/")
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

MALWARE_TYPES = [folder for folder in MALWARE_DIR.iterdir() if folder.is_dir()]
MAX_SAMPLES_PER_TYPE = [-1] * len(MALWARE_TYPES)
EMBEDDING_SIZE = 128  # Dimension of the embeddings (vectors)
NUM_CORES = cpu_count()
CHUNK_SIZE = 5000  # Number of opcodes per chunk


def build_corpus(filepaths: List[Path]) -> List[List[str]]:
    """
    Build a corpus from the given filepaths.
    Reads opcode sequences and processes them in chunks if necessary.

    Args:
        filepaths (List[Path]): List of file paths to process.

    Returns:
        List[List[str]]: A list of opcode sequences.
    """
    corpus = []
    for filepath in filepaths:
        with filepath.open("r") as file:
            opcodes = [l for line in file if (l := line.strip())]
            # Break into chunks if needed
            for i in range(0, len(opcodes), CHUNK_SIZE):
                corpus.append(opcodes[i : i + CHUNK_SIZE])
    return corpus


def process_malware_type(
    malware_dir: Path, max_samples: int
) -> Tuple[FastText, List[Path], List[List[str]]]:
    """
    Process a malware type folder:
        - Read opcode files.
        - Build corpus.
        - Train a FastText model.

    Args:
        malware_dir (Path): Path to the malware type folder.
        max_samples (int): Maximum number of samples to process (-1 for all).

    Returns:
        Tuple[FastText, List[Path], List[List[str]]]:
            The trained FastText model, list of processed filepaths, and the corpus.
    """
    filepaths = list(malware_dir.glob("*.txt"))
    if 0 <= max_samples < len(filepaths):
        filepaths = filepaths[:max_samples]

    corpus = build_corpus(filepaths)

    model = FastText(
        sentences=corpus,
        vector_size=EMBEDDING_SIZE,
        window=5,
        min_count=1,
        workers=NUM_CORES,
    )
    return model, filepaths, corpus


def compute_mean_embeddings(
    model: FastText, filepaths: List[Path], malware_type: str
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Compute the mean embedding for each file.
    If a file is too large, it is split into chunks, and embeddings are averaged.

    Args:
        model (FastText): Trained FastText model.
        filepaths (List[Path]): List of file paths.
        malware_type (str): The malware type (used as a key).

    Returns:
        Dict[Tuple[str, str], np.ndarray]: Dictionary with keys as (malware_type, filename)
            and values as mean embeddings.
    """
    mean_embeddings = {}
    for filepath in filepaths:
        with filepath.open("r") as file:
            opcodes = [l for line in file if (l := line.strip())]
            chunk_embeddings = []
            for i in range(0, len(opcodes), CHUNK_SIZE):
                chunk = opcodes[i : i + CHUNK_SIZE]
                valid_embeddings = [model.wv[opcode] for opcode in chunk if opcode in model.wv]
                if valid_embeddings:
                    chunk_embeddings.append(np.mean(valid_embeddings, axis=0))
            
            if chunk_embeddings:
                mean_embedding = np.mean(chunk_embeddings, axis=0)
                mean_embeddings[(malware_type, filepath.name)] = mean_embedding
    return mean_embeddings


def main() -> None:
    """
    Main function to process malware types, train FastText models,
    compute mean embeddings, and save results.
    """
    all_mean_embeddings: Dict[Tuple[str, str], np.ndarray] = {}

    for malware_dir, max_samples in zip(MALWARE_TYPES, MAX_SAMPLES_PER_TYPE):
        malware_type = malware_dir.name
        print(f"Processing malware type: {malware_type:<20}", end="\t", flush=True)

        # Process the current malware folder
        model, filepaths, _ = process_malware_type(malware_dir, max_samples)

        # Save the FastText model
        model_filename = SAVED_MODELS_DIR / f"{malware_type}.model"
        model.save(str(model_filename))

        # Compute and collect mean embeddings
        mean_embeddings = compute_mean_embeddings(model, filepaths, malware_type)
        all_mean_embeddings.update(mean_embeddings)
        print("(Done).", flush=True)

    # Save the mean embeddings to a pickle file
    pickle_filepath = SAVED_MODELS_DIR / "mean_embedding_per_file.pkl"
    with pickle_filepath.open("wb") as f:
        pickle.dump(all_mean_embeddings, f)
    print("Saved mean embeddings to pickle file.", flush=True)


if __name__ == "__main__":
    main()
