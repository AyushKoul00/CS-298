from pathlib import Path
import random
import shutil

# Define the source directory and the destination directory as constants
SRC_DIR = Path('/home/akoul/project_code/malware_data/v077_clean')
DEST_DIR = Path('/home/akoul/project_code/dataset')

# Ensure the destination directory exists
DEST_DIR.mkdir(parents=True, exist_ok=True)

# List number of files in each folder of DEST_DIR
for f in DEST_DIR.iterdir():
    if not f.is_dir():
        continue
    file_count = sum(1 for item in f.iterdir() if item.is_file())
    print(file_count)
exit(0)


# List all sub-folders (directories) in the source directory
folders = [f for f in SRC_DIR.iterdir() if f.is_dir()]

# Filter folders that have at least 1000 files (non-recursively) in each
qualified_folders = []
for f in folders:
    file_count = sum(1 for item in f.iterdir() if item.is_file())
    if file_count >= 1000:
        qualified_folders.append((f.name, file_count))

# Select 10 random folders from the qualified list (if there are at least 10)
selected_folders = random.sample(qualified_folders, 10) if len(qualified_folders) >= 10 else qualified_folders

print("Selected folders (name, file count):", selected_folders)

# Copy each selected folder's content to the destination directory
for folder_name, _ in selected_folders:
    src_folder = SRC_DIR / folder_name
    dest_folder = DEST_DIR / folder_name

    # Copy the entire folder (including all subdirectories and files)
    # dirs_exist_ok=True allows the destination directory to exist (Python 3.8+)
    shutil.copytree(src_folder, dest_folder, dirs_exist_ok=True)
