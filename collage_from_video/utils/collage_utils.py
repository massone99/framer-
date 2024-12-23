# utils/collage_utils.py

import cv2
import math
import numpy as np

def create_collage(frames, enlarge_frame_index=None, enlarge_factor=1.1):
    """
    Create a collage image by arranging frames.
    
    - If the number of frames is even, arrange them two per row.
    - If the number of frames is odd, arrange them vertically, one per row.
    
    Optionally enlarge a specific frame slightly for emphasis.
    
    Args:
        frames (list): List of frame images (numpy.ndarray).
        enlarge_frame_index (int, optional): Index of the frame to enlarge. Defaults to None.
        enlarge_factor (float, optional): Factor by which to enlarge the selected frame. Defaults to 1.1.
    
    Returns:
        numpy.ndarray: The collage image.
    """
    num_frames = len(frames)
    
    # Determine grid configuration based on frame count
    if num_frames % 2 == 0:
        # Even number of frames: two per row
        cols = 2
        rows = num_frames // 2
    else:
        # Odd number of frames: one per row (vertical arrangement)
        cols = 1
        rows = num_frames
    
    # Optionally enlarge a specific frame
    if enlarge_frame_index is not None and 0 <= enlarge_frame_index < num_frames:
        frames[enlarge_frame_index] = enlarge_frame(frames[enlarge_frame_index], enlarge_factor)
    
    # Determine the maximum width and height for uniform resizing
    frame_heights = [frame.shape[0] for frame in frames]
    frame_widths = [frame.shape[1] for frame in frames]
    max_height = max(frame_heights)
    max_width = max(frame_widths)
    
    # Resize frames to have uniform dimensions
    resized_frames = []
    for idx, frame in enumerate(frames, start=1):
        resized_frame = cv2.resize(frame, (max_width, max_height))
        # Overlay frame numbers
        cv2.putText(
            resized_frame, f"{idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2, cv2.LINE_AA
        )
        resized_frames.append(resized_frame)
    
    # Calculate the size of the collage canvas
    collage_height = rows * max_height
    collage_width = cols * max_width
    
    # Create a black canvas for the collage
    collage = np.zeros((collage_height, collage_width, 3), dtype=np.uint8)
    
    # Place each frame onto the collage canvas
    for idx, frame in enumerate(resized_frames):
        row = idx // cols
        col = idx % cols
        y_start = row * max_height
        y_end = y_start + max_height
        x_start = col * max_width
        x_end = x_start + max_width
        collage[y_start:y_end, x_start:x_end] = frame
    
    # If odd number of frames, pad the remaining space with black (handled inherently by initial canvas)
    # Alternatively, you can add placeholders or leave them as black
    
    # Add information about current settings
    info_text = f"Frames: {num_frames} | Layout: {'Vertical' if num_frames % 2 ==1 else 'Two per Row'}"
    cv2.putText(
        collage, info_text, (10, collage.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2, cv2.LINE_AA
    )
    
    return collage

def enlarge_frame(frame, factor=1.1):
    """
    Enlarge a frame by a specified factor while maintaining aspect ratio.
    
    Args:
        frame (numpy.ndarray): Frame image to enlarge.
        factor (float, optional): Enlargement factor. Defaults to 1.1.
    
    Returns:
        numpy.ndarray: Enlarged frame image.
    """
    height, width = frame.shape[:2]
    new_width = int(width * factor)
    new_height = int(height * factor)
    enlarged_frame = cv2.resize(frame, (new_width, new_height))
    return enlarged_frame

def resize_image_for_display(image, max_width=1280, max_height=720):
    """
    Resize the image to fit within max_width and max_height while maintaining aspect ratio.
    
    Args:
        image (numpy.ndarray): Image to resize.
        max_width (int): Maximum width for display.
        max_height (int): Maximum height for display.
    
    Returns:
        numpy.ndarray: Resized image.
    """
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height, 1)  # Ensure scaling factor <=1
    if scaling_factor < 1:
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return resized_image
    return image
