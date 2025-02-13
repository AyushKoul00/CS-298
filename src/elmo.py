import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional

# Enable TensorFlow 1.x compatibility mode (Required for ELMo)
tf.compat.v1.disable_eager_execution()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("elmo_embedding.log"),
        logging.StreamHandler()
    ]
)

# Define constants
MALWARE_DIR = Path("../dataset/")
SAVED_MODELS_DIR = Path("../saved_models/elmo/")
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Each subfolder in MALWARE_DIR is a malware type
MALWARE_TYPES = [folder.name for folder in MALWARE_DIR.iterdir() if folder.is_dir()]

# Set to -1 to read all files, or set a maximum number of files per folder
MAX_SAMPLES_PER_TYPE = [-1] * len(MALWARE_TYPES)
CHUNK_SIZE = 1000  # Process opcodes in chunks to avoid truncation issues

# Load ELMo model (TensorFlow 1.x style)
try:
    logging.info("Loading ELMo model...")
    elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=False)
    logging.info("ELMo model loaded successfully.")
except Exception as e:
    logging.exception("Failed to load ELMo model.")
    raise e


def load_opcodes(file_path: Path) -> List[str]:
    """
    Reads a file containing opcode sequences.

    Args:
        file_path (Path): Path to the opcode file.

    Returns:
        List[str]: List of opcodes.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            opcodes = [line.strip() for line in f if line.strip()]
        return opcodes
    except Exception as e:
        logging.exception(f"Error reading file {file_path}")
        return []


def get_elmo_embedding(opcodes: List[str], chunk_size: int = CHUNK_SIZE) -> Optional[np.ndarray]:
    """
    Generates an ELMo embedding for a given list of opcodes using chunking to avoid truncation.

    Args:
        opcodes (List[str]): List of opcode strings.
        chunk_size (int): Maximum number of tokens to process at once.

    Returns:
        Optional[np.ndarray]: Aggregated ELMo embedding or None if failed.
    """
    if not opcodes:
        return None

    try:
        embeddings = []

        # TensorFlow 1.x Session
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            sess.run(tf.compat.v1.tables_initializer())

            # Process opcodes in manageable chunks
            for i in range(0, len(opcodes), chunk_size):
                chunk = opcodes[i: i + chunk_size]

                # Generate embeddings
                elmo_embeddings = elmo(chunk, signature="default", as_dict=True)["elmo"]
                chunk_embedding = sess.run(tf.reduce_mean(elmo_embeddings, axis=1))  # Mean of chunk
                embeddings.append(chunk_embedding)

            # Aggregate embeddings
            final_embedding = np.mean(embeddings, axis=0)  # Mean of all chunks

            return final_embedding

    except Exception as e:
        logging.exception("Error generating ELMo embedding")
        return None


def process_malware_files():
    """
    Processes opcode files for each malware type and generates ELMo embeddings.
    """
    embeddings_dict: Dict[Tuple[str, str], np.ndarray] = {}

    for malware_type, max_samples in zip(MALWARE_TYPES, MAX_SAMPLES_PER_TYPE):
        type_dir = MALWARE_DIR / malware_type

        if not type_dir.exists():
            logging.warning(f"Skipping missing directory: {type_dir}")
            continue

        file_paths = sorted(type_dir.glob("*.txt"))
        if max_samples > 0:
            file_paths = file_paths[:max_samples]

        logging.info(f"Processing {len(file_paths)} files for {malware_type}...")

        for file_path in file_paths:
            try:
                opcodes = load_opcodes(file_path)
                if not opcodes:
                    logging.warning(f"Skipping {file_path.name}: No valid opcodes found.")
                    continue

                embedding = get_elmo_embedding(opcodes)
                if embedding is not None:
                    embeddings_dict[(malware_type, file_path.name)] = embedding
                else:
                    logging.warning(f"Skipping {file_path.name}: Embedding generation failed.")

            except Exception as e:
                logging.exception(f"Unexpected error processing {file_path}")

    # Save embeddings
    save_path = SAVED_MODELS_DIR / "mean_embedding_per_file.pkl"
    try:
        with save_path.open("wb") as f:
            pickle.dump(embeddings_dict, f)
        logging.info(f"Successfully processed {len(embeddings_dict)} files.")
        logging.info(f"Embeddings saved to: {save_path}")

    except Exception as e:
        logging.exception("Failed to save embeddings")


if __name__ == "__main__":
    process_malware_files()
