from pathlib import Path

def count_files_in_subdirectories(dataset_path: str) -> None:
    """
    Iterates over each subdirectory in the given dataset path and prints
    the number of files in each subdirectory.

    Parameters:
        dataset_path (str): Path to the dataset directory.
    """
    base_path = Path(dataset_path)
    
    # Iterate over items in the dataset and filter only for directories.
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            # Count files directly contained in the subdirectory.
            file_count = sum(1 for item in subdir.iterdir() if item.is_file())
            print(f"Directory: {subdir.name}, Files: {file_count}")

if __name__ == "__main__":
    # Replace 'dataset' with the path to your dataset folder if needed.
    count_files_in_subdirectories("../dataset")
