"""
I/O operations for MultimodalData objects.

This module handles saving and loading MultimodalData instances to/from disk.
"""
import joblib
from .data_structures import MultimodalData


def save_to_file(multimodal_data: MultimodalData, output_dir: str) -> None:
    """
    Save MultimodalData instance to a joblib file.

    Args:
        multimodal_data: The multimodal data instance to save.
        output_dir: Directory path where the file will be saved.

    Returns:
        None: Saves file to {output_dir}/{dyad_id}.joblib
    """
    joblib.dump(multimodal_data, output_dir + f"/{multimodal_data.id}.joblib")


def load_output_data(filename: str) -> MultimodalData | None:
    """
    Load saved MultimodalData from a joblib file.

    Args:
        filename: Path to the joblib file to load.

    Returns:
        MultimodalData or None: The loaded multimodal data instance, or None if file not found.
    """
    try:
        results = joblib.load(filename)
        return results
    except FileNotFoundError:
        print(f"File not found {filename}")
        return None
