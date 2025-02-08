import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pickle
from pathlib import Path
from typing import Dict, Tuple

# Define variables
MALWARE_DIR = Path('../malware_data/v077_clean/')  # Directory containing malware type folders
SAVED_MODELS_DIR = Path('../saved_models/elmo/')
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
MALWARE_TYPES = ['Winwebsec', 'Small', 'Zbot']  # Malware type folder names
MAX_SAMPLES_PER_TYPE = [500] * len(MALWARE_TYPES)  # Set to -1 for all files

# Load ELMo model once (outside processing loop)
ELMO_MODEL = hub.load("https://tfhub.dev/google/elmo/3")

def load_opcodes(file_path: Path) -> list:
    """Load opcodes from a file, one per line"""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def get_elmo_embedding(opcodes: list) -> np.ndarray:
    """Generate single ELMo embedding for an opcode sequence"""
    if not opcodes:
        return None
    
    # Process through ELMo model
    outputs = ELMO_MODEL.signatures["tokens"](
        tokens=tf.constant([opcodes]),  # Batch of 1 sequence
        sequence_len=tf.constant([len(opcodes)])
    )
    
    # Average token embeddings to get sequence representation
    token_embeddings = outputs["elmo"].numpy()[0]
    return np.mean(token_embeddings, axis=0)

# Dictionary to store embeddings {(malware_type, filename): embedding}
embeddings_dict: Dict[Tuple[str, str], np.ndarray] = {}

# Process each malware type
for malware_type, max_samples in zip(MALWARE_TYPES, MAX_SAMPLES_PER_TYPE):
    type_dir = MALWARE_DIR / malware_type
    if not type_dir.exists():
        print(f"Skipping missing directory: {type_dir}")
        continue

    # Get all text files in directory
    file_paths = list(type_dir.glob('*.txt'))
    if max_samples > 0:
        file_paths = file_paths[:max_samples]

    # Process each file in the malware type directory
    for file_path in file_paths:
        try:
            opcodes = load_opcodes(file_path)
            if not opcodes:
                continue
                
            embedding = get_elmo_embedding(opcodes)
            if embedding is not None:
                key = (malware_type, file_path.name)
                embeddings_dict[key] = embedding
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

# Save embeddings to pickle file
with open(SAVED_MODELS_DIR / 'mean_embedding_per_file.pkl', 'wb') as f:
    pickle.dump(embeddings_dict, f)

print(f"Successfully processed {len(embeddings_dict)} files")
print(f"Embeddings saved to: {SAVED_MODELS_DIR / 'mean_embedding_per_file.pkl'}")