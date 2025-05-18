import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

import numpy as np

# --- CONFIGURATION ---
MALWARE_DIR = Path("../dataset/")          # root dir with subfolders per malware type
MAX_SAMPLES_PER_TYPE = -1                  # -1 to use _all_ samples per folder
OUTPUT_DIR = Path("../saved_models/")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
PICKLE_PATH = OUTPUT_DIR / "opcode_distribution_embeddings.pkl"


def gather_all_filepaths(
    root: Path, max_samples: int
) -> List[Path]:
    """Collect up to max_samples .txt files from each subfolder."""
    files = []
    for subtype in root.iterdir():
        if not subtype.is_dir():
            continue
        txts = list(subtype.glob("*.txt"))
        if 0 <= max_samples < len(txts):
            txts = txts[:max_samples]
        files.extend(txts)
    return files


def build_corpus(filepaths: List[Path]) -> List[List[str]]:
    """Read each file into a list of opcodes."""
    corpus = []
    for fp in filepaths:
        with fp.open("r") as f:
            ops = [l for line in f if (l:= line.strip())]
        corpus.append(ops)
    return corpus


def build_global_vocabulary(corpus: List[List[str]]) -> List[str]:
    """Return a sorted list of all unique opcodes in the corpus."""
    vocab = set()
    for ops in corpus:
        vocab.update(ops)
    return sorted(vocab)


def compute_distribution_embeddings(
    filepaths: List[Path],
    corpus: List[List[str]],
    vocab: List[str]
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Compute a percentage-distribution vector for each file.
    Keyed by (malware_type, file_id).
    """
    idx = {opc: i for i, opc in enumerate(vocab)}
    embeddings: Dict[Tuple[str, str], np.ndarray] = {}

    for fp, ops in zip(filepaths, corpus):
        cnt = Counter(ops)
        total = len(ops)
        vec = np.zeros(len(vocab), dtype=np.float32)
        if total > 0:
            for opc, c in cnt.items():
                vec[idx[opc]] = c / total
        key = (fp.parent.name, fp.name)
        embeddings[key] = vec

    return embeddings


def main():
    # 1. collect files
    filepaths = gather_all_filepaths(MALWARE_DIR, MAX_SAMPLES_PER_TYPE)
    print(f"Collected {len(filepaths)} samples.")

    # 2. read opcode sequences
    corpus = build_corpus(filepaths)

    # 3. build global opcode vocab
    vocab = build_global_vocabulary(corpus)
    print(f"Vocabulary size: {len(vocab)} opcodes.")

    # 4. compute per-file distribution embeddings
    embeddings = compute_distribution_embeddings(filepaths, corpus, vocab)
    print(f"Computed embeddings for {len(embeddings)} files.")

    # 5. save the dict (you can also pickle vocab separately if needed)
    with PICKLE_PATH.open("wb") as f:
        pickle.dump(embeddings, f)
    print(f"Saved embeddings dict to {PICKLE_PATH!r}")


if __name__ == "__main__":
    main()