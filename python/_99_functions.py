# Imports
from pathlib import Path

def get_project_root():
    """Returns the project root folder."""
    # Start from the current file's directory or notebook's location
    current_path = Path().resolve()

    # Loop to go up the directory tree until we reach the "ComputationalTools" folder
    while current_path.name != "ComputationalTools":
        if current_path.parent == current_path:  # We've reached the root directory without finding the folder
            raise FileNotFoundError("Could not find the 'ComputationalTools' folder in the directory hierarchy.")
        current_path = current_path.parent
        
    return current_path
