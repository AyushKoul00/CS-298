# %%
import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import pickle

# %%
# Define variables
BERT_MODEL_TYPE = 'distilbert'
MALWARE_DIR = Path('../malware_data/v077_clean/')  # Directory containing malware type folders
SAVED_MODELS_DIR = Path(f'../saved_models/{BERT_MODEL_TYPE}/')
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
MALWARE_TYPES = ['Winwebsec', 'Small', 'Zbot']  # Malware type folder names
MAX_SAMPLES_PER_TYPE = [500] * len(MALWARE_TYPES) # Set to -1 to read all files, or set to maximum number of files per folder
MAX_CHUNK_LENGTH = 512
CHUNK_OVERLAP_PERCENTAGE = 0.2

# %%
def tokenize_and_chunk(opcodes, tokenizer, max_length=512, overlap_percent=0.1):
    """
    Tokenize all opcodes into subwords first, then split into chunks with overlap
    
    Args:
        opcodes (list): List of opcode strings
        tokenizer: Hugging Face tokenizer
        max_length (int): Maximum sequence length
        overlap_percent (float): Overlap percentage between chunks
    
    Returns:
        BatchEncoding: Contains input_ids, attention_mask, etc.
    """
    # Tokenize all opcodes into subwords using list comprehension
    all_tokens = [token for opcode in opcodes for token in tokenizer.tokenize(opcode)]

    # Calculate chunking parameters
    chunk_size = max_length - 2  # Account for [CLS] and [SEP]
    step = max(1, int(chunk_size * (1 - overlap_percent)))
    
    # Generate overlapping chunks using walrus operator
    token_chunks = []
    start_idx = 0
    while (current_chunk := all_tokens[start_idx:start_idx + chunk_size]):
        token_chunks.append(current_chunk)
        start_idx += step

    # Convert token chunks to model inputs
    return tokenizer(
        token_chunks,
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
        add_special_tokens=True
    )

def generate_malware_embeddings(model_name='bert-base-uncased', overlap_percent=0.1):
    """
    Generate embeddings using BERT with overlapping token chunks
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    embeddings = {}

    # Process each malware type
    for malware_type, max_samples in zip(MALWARE_TYPES, MAX_SAMPLES_PER_TYPE):
        curr_dir = MALWARE_DIR / malware_type
        if not curr_dir.is_dir():
            print(f"Skipping {curr_dir} because it doesn't exist")
            continue  # Skip if the directory doesn't exist
        else:
            print(f"Finding embeddings for {curr_dir}")

        # Convert to a list so we can reuse it
        filepaths = list(curr_dir.glob('*.txt'))

        # Optionally limit the number of samples
        if max_samples > 0 and max_samples < len(filepaths):
            filepaths = filepaths[:max_samples]

        for filepath in filepaths:
            # Read opcodes with walrus operator
            with open(filepath, 'r', encoding='utf-8') as f:
                opcodes = [l for line in f if (l := line.strip())]

            # Tokenize and chunk with overlap
            encoded_chunks = tokenize_and_chunk(
                opcodes=opcodes,
                tokenizer=tokenizer,
                max_length=MAX_CHUNK_LENGTH,
                overlap_percent=overlap_percent
            )

            # Process all chunks in batch with inference mode
            with torch.inference_mode():
                outputs = model(**encoded_chunks)

            # Calculate valid token mask
            input_ids = encoded_chunks['input_ids']
            # print(input_ids.shape)

            valid_mask = (
                (input_ids != tokenizer.cls_token_id) &
                (input_ids != tokenizer.sep_token_id) &
                (input_ids != tokenizer.pad_token_id)
            )

            # # Mean Chunk Embeddings
            # chunk_embeddings = [
            #     outputs.last_hidden_state[i][mask].mean(dim=0).numpy()
            #     for i, mask in enumerate(valid_mask)
            #     if mask.any()
            # ]

            # CLS Chunk Embeddings
            chunk_embeddings = [
                outputs.last_hidden_state[i][0].cpu().numpy()  # CLS token
                for i in range(input_ids.shape[0])
                if valid_mask[i].any()  # Still filter empty chunks
            ]

            # Average across chunks (no normalization)
            file_embedding = np.mean(chunk_embeddings, axis=0) if chunk_embeddings \
                else np.zeros(model.config.hidden_size)
            
            embeddings[(malware_type, filepath.name)] = file_embedding

    return embeddings

# %%
embeddings = generate_malware_embeddings(
    model_name=f'{BERT_MODEL_TYPE}-base-uncased',
    overlap_percent=CHUNK_OVERLAP_PERCENTAGE  # 20% overlap between token chunks
)

print(f"Generated embeddings for {len(embeddings)} files")

for key, E in embeddings.items():
    # print(f'{filename}:', E, sep="\n")
    print(f'{key}: {E.shape}')

# %%
with (SAVED_MODELS_DIR / 'mean_embedding_per_file.pkl').open('wb') as f:
    pickle.dump(embeddings, f)


