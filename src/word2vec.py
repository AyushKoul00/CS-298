# %%
import os
import numpy as np
from gensim.models import Word2Vec
import pickle
from typing import Any, Dict, Iterator, List, Optional, Tuple
from multiprocessing import cpu_count
from pathlib import Path

# %%
# Define variables
MALWARE_DIR = Path('../malware_data/v077_clean/')  # Directory containing malware type folders
SAVED_MODELS_DIR = Path(f'../saved_models/word2vec/')
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
MALWARE_TYPES = ['Winwebsec', 'Small', 'Zbot']  # Malware type folder names
MAX_SAMPLES_PER_TYPE = [500] * len(MALWARE_TYPES) # Set to -1 to read all files, or set to maximum number of files per folder
EMBEDDING_SIZE = 128        # Dimension of the embeddings (vectors)
NUM_CORES = cpu_count()

# %%
# Dictionary to store embeddings per file
MEAN_EMBEDDING_PER_FILE: Dict[Path, np.ndarray] = {}

# Process each malware type
for malware_type, max_samples in zip(MALWARE_TYPES, MAX_SAMPLES_PER_TYPE):
    curr_dir = MALWARE_DIR / malware_type
    if not curr_dir.is_dir():
        continue  # Skip if the directory doesn't exist

    # Convert to a list so we can reuse it
    filepaths = list(curr_dir.glob('*.txt'))

    # Optionally limit the number of samples
    if max_samples > 0 and max_samples < len(filepaths):
        filepaths = filepaths[:max_samples]

    # Build the corpus (list of opcode lists)
    corpus: List[List[str]] = []
    for filepath in filepaths:
        with filepath.open('r') as f:
            # Collect non-empty lines (opcodes)
            opcodes = [l for line in f if (l := line.strip())]
            corpus.append(opcodes)

    # Train a Word2Vec model for the current malware family
    model = Word2Vec(
        sentences=corpus,
        vector_size=EMBEDDING_SIZE,  # Embedding size
        window=5,                    # Context window size
        min_count=1,                 # Minimum frequency for a word to be included
        workers=NUM_CORES - 1        # Number of worker threads
    )

    # Save the Word2Vec model for the current malware family
    model_filename = SAVED_MODELS_DIR / f'{malware_type}.model'
    model.save(str(model_filename))

    # Compute and store mean embeddings for each file
    for filepath, opcodes in zip(filepaths, corpus):
        # Only take opcodes that are actually in the model vocabulary
        valid_embeddings = [model.wv[opcode] for opcode in opcodes if opcode in model.wv]
        if valid_embeddings:
            mean_embedding = np.mean(valid_embeddings, axis=0)
            MEAN_EMBEDDING_PER_FILE[(malware_type, filepath.name)] = mean_embedding

# Save MEAN_EMBEDDING_PER_FILE to a pickle file
with (SAVED_MODELS_DIR / 'mean_embedding_per_file.pkl').open('wb') as f:
    pickle.dump(MEAN_EMBEDDING_PER_FILE, f)

# %%
def get_embedding(key: Tuple[str, str]) -> Optional[Any]:
    """Return the stored mean embedding for a given file Path."""
    global MEAN_EMBEDDING_PER_FILE
    return MEAN_EMBEDDING_PER_FILE.get(key)

def embeddings() -> Iterator[Any]:
    """Yield embeddings for all (limited) files across malware types."""
    for malware_type, max_samples in zip(MALWARE_TYPES, MAX_SAMPLES_PER_TYPE):
        curr_dir = MALWARE_DIR / malware_type
        if not curr_dir.is_dir():
            continue
        
        filepaths = list(curr_dir.glob('*.txt'))
        if max_samples > 0 and max_samples < len(filepaths):
            filepaths = filepaths[:max_samples]

        for filepath in filepaths:
            yield get_embedding((malware_type, filepath.name))

# %%
# # Example usage: get a specific embedding
# key = (MALWARE_TYPES[0], 'abc.txt')
# embedding = get_embedding(key)
# if embedding is not None:
#     print("Embedding (first 5 elements):", embedding[:5])
# else:
#     print('No embedding found for', key)

# # Example usage: iterate over all embeddings
# for E in embeddings():
#     if E is not None:
#         print("Embedding snippet:", E[:5])
#     else:
#         print("No embedding for this file.")


