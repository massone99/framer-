# utils/frame_utils.py

import cv2
import random

def get_frame_by_number(cap, frame_number):
    """
    Retrieve a specific frame from the video by frame number.

    Args:
        cap (cv2.VideoCapture): OpenCV VideoCapture object.
        frame_number (int): Frame number to retrieve.

    Returns:
        numpy.ndarray: The frame image.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    if success:
        return frame
    else:
        raise ValueError(f"Unable to retrieve frame number {frame_number}.")

def get_random_frame(cap, total_frames, exclude_frames):
    """
    Retrieve a random frame from the video, excluding specified frames.

    Args:
        cap (cv2.VideoCapture): OpenCV VideoCapture object.
        total_frames (int): Total number of frames in the video.
        exclude_frames (list): List of frame numbers to exclude.

    Returns:
        tuple: (frame_number, frame_image)
    """
    attempts = 0
    max_attempts = 100
    while attempts < max_attempts:
        frame_number = random.randint(0, total_frames - 1)
        if frame_number in exclude_frames:
            attempts += 1
            continue
        frame = get_frame_by_number(cap, frame_number)
        return frame_number, frame
    raise ValueError("Unable to retrieve a unique frame after multiple attempts.")
