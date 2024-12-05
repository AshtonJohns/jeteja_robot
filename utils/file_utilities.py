import os
import glob
from pathlib import Path

def sort_files(files):
    return sorted(files, key=lambda f: os.path.getmtime(f)) # Sort on date created

def get_latest_directory(base_dir):
    """Get the latest directory in the given base directory."""
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not dirs:
        return None  # No directories found
    latest_dir = max(dirs, key=lambda d: os.path.getmtime(os.path.join(base_dir, d)))
    return os.path.join(base_dir, latest_dir)   

def get_files_from_directory(directory):
    files = [file.name for file in Path(directory).iterdir() if file.is_file()]
    return files if files else None
