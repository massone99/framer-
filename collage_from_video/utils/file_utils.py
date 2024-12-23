# utils/file_utils.py

import os
import json
import shutil
from pathlib import Path

def is_video_file(file_path, video_extensions):
    """
    Check if the file has a video extension.

    Args:
        file_path (Path): Path object of the file.
        video_extensions (set): Set of supported video file extensions.

    Returns:
        bool: True if file has a supported video extension, False otherwise.
    """
    return file_path.suffix.lower() in video_extensions

def get_all_video_files(directory, video_extensions):
    """
    Recursively get all video files in the directory.

    Args:
        directory (Path): Path object of the directory to search.
        video_extensions (set): Set of supported video file extensions.

    Returns:
        list: List of Path objects for each video file found.
    """
    video_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            if is_video_file(file_path, video_extensions):
                video_files.append(file_path)
    return video_files

def find_existing_image(output_folder, image_name, image_extensions):
    """
    Search recursively in the output folder for an image with the specified name.

    Args:
        output_folder (Path): Path object of the output folder.
        image_name (str): Name of the image file without extension.
        image_extensions (set): Set of supported image file extensions.

    Returns:
        Path or None: Path to the existing image if found, None otherwise.
    """
    for root, dirs, files in os.walk(output_folder):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in image_extensions and file_path.stem == image_name:
                return file_path
    return None

def load_processed_videos(json_path):
    """
    Load the set of processed videos from a JSON file.

    Args:
        json_path (Path): Path to the JSON file.

    Returns:
        set: Set of processed video absolute paths.
    """
    if not json_path.exists():
        return set()
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            return set(data)
    except Exception as e:
        print(f"Error loading processed videos from {json_path}: {e}")
        return set()

def save_processed_video(json_path, video_path):
    """
    Save a processed video's absolute path to the JSON file.

    Args:
        json_path (Path): Path to the JSON file.
        video_path (Path): Path object of the processed video.
    """
    processed = load_processed_videos(json_path)
    processed.add(str(video_path.resolve()))
    try:
        with open(json_path, 'w') as f:
            json.dump(list(processed), f, indent=4)
    except Exception as e:
        print(f"Error saving processed video to {json_path}: {e}")

def copy_existing_image(existing_image_path, collage_output_path):
    """
    Copy an existing image to the collage output path.

    Args:
        existing_image_path (Path): Path to the existing image.
        collage_output_path (Path): Destination path for the collage image.
    """
    try:
        shutil.copy2(existing_image_path, collage_output_path)
        print(f"Copied existing image from {existing_image_path} to {collage_output_path}")
    except Exception as e:
        print(f"Error copying existing image: {e}")
