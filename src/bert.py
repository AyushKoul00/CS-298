# %%
import os
import logging
from pathlib import Path
import pickle
from typing import List, Dict, Tuple

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# %%
# Constants and configuration
BERT_MODEL_TYPE = 'distilbert'
MALWARE_DIR = Path("../dataset/")  # Directory containing malware type folders
SAVED_MODELS_DIR = Path(f"../saved_models/{BERT_MODEL_TYPE}/")
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Each subfolder in MALWARE_DIR is a malware type
MALWARE_TYPES = [folder for folder in MALWARE_DIR.iterdir() if folder.is_dir()]
# Set to -1 to read all files, or set a maximum number of files per folder
MAX_SAMPLES_PER_TYPE = [-1] * len(MALWARE_TYPES)
MAX_CHUNK_LENGTH = 512
CHUNK_OVERLAP_PERCENTAGE = 0.2

# Batch size for processing chunks during inference
BATCH_SIZE = 16

# Configure logging to write to a file with a standard format.
logging.basicConfig(
    filename=SAVED_MODELS_DIR / f"{BERT_MODEL_TYPE}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='w'  # Override log file on each run
)

# %%
def tokenize_and_chunk(
    opcodes: List[str],
    tokenizer,
    max_length: int = 512,
    overlap_percent: float = 0.1,
):
    """
    Tokenize all opcodes into subwords first, then split into chunks with overlap.

    Args:
        opcodes (List[str]): List of opcode strings.
        tokenizer: Hugging Face tokenizer.
        max_length (int): Maximum sequence length.
        overlap_percent (float): Overlap percentage between chunks.

    Returns:
        BatchEncoding: Contains input_ids, attention_mask, etc.
    """
    # Tokenize all opcodes into subwords using list comprehension.
    all_tokens = [token for opcode in opcodes for token in tokenizer.tokenize(opcode)]
    
    # Calculate chunking parameters.
    chunk_size = max_length - 2  # Account for [CLS] and [SEP].
    step = max(1, int(chunk_size * (1 - overlap_percent)))
    
    # Generate overlapping chunks.
    token_chunks = []
    start_idx = 0
    while (current_chunk := all_tokens[start_idx:start_idx + chunk_size]):
        token_chunks.append(current_chunk)
        start_idx += step

    # Convert token chunks to model inputs.
    return tokenizer(
        token_chunks,
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
        add_special_tokens=True
    )

def generate_malware_embeddings(model_name: str = 'bert-base-uncased', overlap_percent: float = 0.1) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Generate embeddings using BERT with overlapping token chunks.

    Args:
        model_name (str): Pre-trained model name.
        overlap_percent (float): Overlap percentage between token chunks.

    Returns:
        Dict[Tuple[str, str], np.ndarray]: Dictionary mapping (malware_type, filename) to its embedding.
    """
    # Set device to CUDA if available, else CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    embeddings = {}

    # Process each malware type.
    for malware_type, max_samples in zip(MALWARE_TYPES, MAX_SAMPLES_PER_TYPE):
        curr_dir = MALWARE_DIR / malware_type
        if not curr_dir.is_dir():
            logging.warning(f"Skipping {curr_dir} because it doesn't exist.")
            continue
        else:
            logging.info(f"Processing malware type in directory: {curr_dir}")

        # Get a list of file paths.
        filepaths = list(curr_dir.glob('*.txt'))

        # Optionally limit the number of samples.
        if max_samples >= 0 and max_samples < len(filepaths):
            filepaths = filepaths[:max_samples]

        for filepath in filepaths:
            # Read opcodes using a context manager.
            with filepath.open('r', encoding='utf-8') as f:
                opcodes = [l for line in f if (l := line.strip())]

            # Tokenize and chunk with overlap.
            encoded_chunks = tokenize_and_chunk(
                opcodes=opcodes,
                tokenizer=tokenizer,
                max_length=MAX_CHUNK_LENGTH,
                overlap_percent=overlap_percent
            )
            
            # Move inputs to the selected device.
            encoded_chunks = {key: val.to(device) for key, val in encoded_chunks.items()}
            input_ids = encoded_chunks['input_ids']
            num_chunks = input_ids.shape[0]
            all_chunk_embeddings = []

            # Process chunks in mini-batches to avoid OOM.
            for i in range(0, num_chunks, BATCH_SIZE):
                # Create mini-batch.
                batch = {key: val[i:i+BATCH_SIZE] for key, val in encoded_chunks.items()}
                with torch.inference_mode():
                    outputs = model(**batch)

                # Extract the CLS token embeddings for each chunk.
                # Even though we are using padding, we use the first token for each sequence.
                batch_embeddings = [
                    outputs.last_hidden_state[j][0].cpu().numpy()
                    for j in range(batch['input_ids'].shape[0])
                ]
                all_chunk_embeddings.extend(batch_embeddings)

            # Average the embeddings across all chunks; if there are no chunks, return a zero vector.
            if all_chunk_embeddings:
                file_embedding = np.mean(all_chunk_embeddings, axis=0)
            else:
                file_embedding = np.zeros(model.config.hidden_size)

            embeddings[(malware_type.name, filepath.name)] = file_embedding
            logging.info(f"Generated embedding for file: {(malware_type.name, filepath.name)}")

            # Optionally clear GPU cache between files.
            torch.cuda.empty_cache()

    return embeddings

# %%
def main() -> None:
    embeddings = generate_malware_embeddings(
        model_name=f'{BERT_MODEL_TYPE}-base-uncased',
        overlap_percent=CHUNK_OVERLAP_PERCENTAGE  # 20% overlap between token chunks.
    )

    logging.info(f"Generated embeddings for {len(embeddings)} files.")

    for key, E in embeddings.items():
        logging.info(f'{key}: {E.shape}')

    # Save the embeddings to a pickle file.
    output_path = SAVED_MODELS_DIR / 'mean_embedding_per_file.pkl'
    with output_path.open('wb') as f:
        pickle.dump(embeddings, f)
    logging.info(f"Embeddings saved to {output_path}")

if __name__ == '__main__':
    main()
