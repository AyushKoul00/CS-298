import os
import glob
import matplotlib.pyplot as plt
import numpy as np

def create_grid_for_technique(technique, root_dir="saved_models", output_dir="."):
    """
    Creates a 5x5 grid image for a given technique using matplotlib.

    Expected folder structure:
      saved_models/{embedding}/{clustering}/*3d*{technique}*.png

    - embeddings:  [bert, distilbert, word2vec, doc2vec, fasttext]
    - clusterings: [birch, dbscan, kmeans, hierarchical, em]

    For each (embedding, clustering) cell, the script looks for files matching:
      *3d*{technique}*.png
    and uses the first match. If no file is found, it shows a blank white cell.

    Parameters:
      technique (str): e.g., "pca", "umap", "t-sne".
      root_dir (str): Top-level directory containing model folders.
      output_dir (str): Directory where the final combined image will be saved.
    """
    # Define folder names for embeddings and clustering methods.
    embeddings = ['bert', 'distilbert', 'word2vec', 'doc2vec', 'fasttext']
    clusterings = ['birch', 'dbscan', 'kmeans', 'hierarchical', 'em']

    n_rows = len(embeddings)
    n_cols = len(clusterings)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15), squeeze=False)

    print(f"\n=== Building grid for technique: {technique} ===")
    
    for row_idx, embedding in enumerate(embeddings):
        for col_idx, clustering in enumerate(clusterings):
            ax = axes[row_idx, col_idx]
            
            # Construct search pattern including "3d" before technique.
            pattern = os.path.join(root_dir, embedding, clustering, f"*3d*{technique}*.png")
            file_list = sorted(glob.glob(pattern))
            
            # Debug information
            print(f"[{embedding} | {clustering}] Searching with pattern: {pattern}")
            if file_list:
                print(f"  Found files: {file_list}")
            else:
                print("  No matches found.")
            
            if file_list:
                try:
                    img_path = file_list[0]
                    img = plt.imread(img_path)
                    ax.imshow(img)
                    ax.set_title(f"{embedding}\n{clustering}", fontsize=9)
                except Exception as e:
                    print(f"  Error loading image {file_list[0]}: {e}")
                    _show_blank(ax, title=f"{embedding}\n{clustering}")
            else:
                _show_blank(ax, title=f"{embedding}\n{clustering}")
            
            ax.axis('off')

    fig.suptitle(f"{technique.upper()} Grid", fontsize=16)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"combined_{technique}.png")
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved grid for '{technique}' to {output_path}\n")

def _show_blank(ax, title=""):
    """Helper function to show a blank white cell with an optional title."""
    blank_image = np.ones((100, 100, 3), dtype=np.float32)
    ax.imshow(blank_image)
    ax.set_title(title, fontsize=9)

if __name__ == "__main__":
    # Define the techniques you want grids for.
    techniques = ['pca', 'umap', 't-sne']
    
    # Create grid images for each technique.
    for tech in techniques:
        create_grid_for_technique(tech, root_dir="../saved_models", output_dir=".")
