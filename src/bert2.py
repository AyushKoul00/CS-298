import logging
from pathlib import Path
import pickle
from typing import List, Dict, Tuple

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, BatchEncoding

# Constants and configuration
BERT_MODEL_TYPE = 'bert'  
MALWARE_DIR = Path("../dataset/")  # Directory containing malware type folders
SAVED_MODELS_DIR = Path(f"../saved_models/{BERT_MODEL_TYPE}/")
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Malware types (subfolders in MALWARE_DIR)
MALWARE_TYPES = [folder for folder in MALWARE_DIR.iterdir() if folder.is_dir()]
MAX_SAMPLES_PER_TYPE = [-1] * len(MALWARE_TYPES)  # -1 for all files

MAX_CHUNK_LENGTH = 512
CHUNK_OVERLAP_PERCENTAGE = 0.2
BATCH_SIZE = 16

# Logging setup
logging.basicConfig(
    filename=SAVED_MODELS_DIR / f"{BERT_MODEL_TYPE}2.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode='w'  # Overwrite log file on each run
)

def tokenize_and_chunk(
    opcodes: List[str],
    tokenizer,
    max_length: int = 512,
    overlap_percent: float = 0.1,
) -> BatchEncoding:
    """Tokenizes opcodes, chunks them with overlap, and returns a BatchEncoding."""

    all_tokens = [token for opcode in opcodes for token in tokenizer.tokenize(opcode)]
    # print(all_tokens)
    chunk_size = max_length - 2  # Account for [CLS] and [SEP]
    step = max(1, int(chunk_size * (1 - overlap_percent)))

    token_chunks = []
    start_idx = 0
    while start_idx < len(all_tokens):
        chunk = all_tokens[start_idx:min(start_idx + chunk_size, len(all_tokens))]
        token_chunks.append(chunk)
        start_idx += step

    encoded_chunks = tokenizer(
        token_chunks,
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
        add_special_tokens=True,
    )

    # # Decode and print input_ids for debugging
    # for i, input_ids_chunk in enumerate(encoded_chunks['input_ids']):
    #     decoded_text = tokenizer.decode(input_ids_chunk)
    #     print(f"\nChunk {i+1}: {decoded_text}")  # Print decoded text for each chunk

    return encoded_chunks



def generate_malware_embeddings(model_name: str, overlap_percent: float) -> Dict[Tuple[str, str], np.ndarray]:
    """Generates malware embeddings using BERT with overlapping chunks."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    embeddings = {}

    for malware_type, max_samples in zip(MALWARE_TYPES, MAX_SAMPLES_PER_TYPE):
        curr_dir = MALWARE_DIR / malware_type
        if not curr_dir.is_dir():
            logging.warning(f"Skipping {curr_dir} (not a directory).")
            continue

        logging.info(f"Processing malware type: {curr_dir}")
        filepaths = list(curr_dir.glob('*.txt'))

        if max_samples >= 0 and max_samples < len(filepaths):
            filepaths = filepaths[:max_samples]

        for filepath in filepaths:
            try:
                with filepath.open('r', encoding='utf-8') as f:
                    opcodes = [l for line in f if (l := line.strip())]

                encoded_chunks = tokenize_and_chunk(opcodes, tokenizer, MAX_CHUNK_LENGTH, overlap_percent)
                # for k, v in encoded_chunks.items():
                #     print(k, v.shape)
                # exit(0)
                encoded_chunks = {k: v.to(device) for k, v in encoded_chunks.items()}

                input_ids = encoded_chunks['input_ids']
                num_chunks = input_ids.shape[0]
                all_chunk_embeddings = []

                for i in range(0, num_chunks, BATCH_SIZE):
                    batch = {k: v[i:i + BATCH_SIZE] for k, v in encoded_chunks.items()}
                    with torch.inference_mode():
                        outputs = model(**batch)

                    batch_embeddings = [outputs.last_hidden_state[j][0].cpu().numpy() for j in range(batch['input_ids'].shape[0])]
                    all_chunk_embeddings.extend(batch_embeddings)

                if all_chunk_embeddings:
                    file_embedding = np.mean(all_chunk_embeddings, axis=0)
                else:
                    file_embedding = np.zeros(model.config.hidden_size)

                embeddings[(malware_type.name, filepath.name)] = file_embedding
                logging.info(f"Generated embedding for: {(malware_type.name, filepath.name)}")
                torch.cuda.empty_cache()  # Clear cache after each file

            except Exception as e:
                logging.error(f"Error processing {filepath}: {e}") # log errors for each file

    return embeddings


def main() -> None:
    model_name = f'{BERT_MODEL_TYPE}-base-uncased' # or just BERT_MODEL_TYPE
    embeddings = generate_malware_embeddings(model_name, CHUNK_OVERLAP_PERCENTAGE)

    logging.info(f"Generated embeddings for {len(embeddings)} files.")
    for key, embedding in embeddings.items():
        logging.info(f'{key}: {embedding.shape}')

    output_path = SAVED_MODELS_DIR / 'mean_embedding_per_file2.pkl'
    with output_path.open('wb') as f:
        pickle.dump(embeddings, f)
    logging.info(f"Embeddings saved to {output_path}")


if __name__ == '__main__':
    main()