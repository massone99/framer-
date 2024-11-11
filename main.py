import cv2
import numpy as np
import random
import os
import json
from tkinter import Tk, messagebox
from tkinter import filedialog
from pathlib import Path
import sys

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
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
        if success:
            return frame_number, frame
        attempts += 1
    raise ValueError("Unable to retrieve a unique frame after multiple attempts.")

def create_collage(frames):
    """
    Create a collage image by arranging frames in two columns,
    with frame numbers overlaid.

    Args:
        frames (list): List of frame images (numpy arrays).

    Returns:
        numpy.ndarray: Collage image.
    """
    num_frames = len(frames)
    num_rows = (num_frames + 1) // 2  # Number of rows needed

    # Resize frames to have the same width
    widths = [frame.shape[1] for frame in frames]
    min_width = min(widths)
    resized_frames = [
        cv2.resize(frame, (min_width, int(frame.shape[0] * min_width / frame.shape[1])))
        for frame in frames
    ]

    # Overlay frame numbers
    for idx, frame in enumerate(resized_frames):
        cv2.putText(
            frame, f"{idx+1}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2, cv2.LINE_AA
        )

    # Split frames into two columns
    left_column_frames = []
    right_column_frames = []

    for idx, frame in enumerate(resized_frames):
        if idx % 2 == 0:
            left_column_frames.append(frame)
        else:
            right_column_frames.append(frame)

    # Stack frames in each column vertically
    left_column = cv2.vconcat(left_column_frames)
    right_column = cv2.vconcat(right_column_frames) if right_column_frames else None

    # Make sure both columns have the same height
    if right_column is not None:
        if left_column.shape[0] > right_column.shape[0]:
            # Pad right_column at the bottom
            padding = left_column.shape[0] - right_column.shape[0]
            right_column = cv2.copyMakeBorder(right_column, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif left_column.shape[0] < right_column.shape[0]:
            # Pad left_column at the bottom
            padding = right_column.shape[0] - left_column.shape[0]
            left_column = cv2.copyMakeBorder(left_column, 0, padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Stack the two columns horizontally
    if right_column is not None:
        collage = cv2.hconcat([left_column, right_column])
    else:
        collage = left_column

    # Add information about current settings
    info_text = f"Frames: {len(frames)} | Layout: {num_rows}x2"
    cv2.putText(
        collage, info_text, (10, collage.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 255), 2, cv2.LINE_AA
    )

    return collage

def resize_image_for_display(image, max_width=800, max_height=600):
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

def process_video(video_path, input_folder, output_folder, json_path, video_index, total_videos):
    """
    Process a single video to create a collage.

    Args:
        video_path (Path): Path object of the video file.
        input_folder (Path): Path object of the input folder.
        output_folder (Path): Path object of the output folder.
        json_path (Path): Path object to the JSON file tracking processed videos.
        video_index (int): Current video index for display purposes.
        total_videos (int): Total number of videos to process.

    Returns:
        bool: True to continue processing, False to exit the program.
    """
    processed_videos = load_processed_videos(json_path)
    if str(video_path.resolve()) in processed_videos:
        print(f"\nSkipping already processed video {video_index + 1}/{total_videos}: {video_path}")
        return True  # Continue processing

    print(f"\nProcessing video {video_index + 1}/{total_videos}: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}. Skipping.")
        return True  # Continue processing

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 4:
        print(f"Error: Video {video_path} does not have enough frames. Skipping.")
        cap.release()
        return True  # Continue processing

    # Initialize frame numbers and frames
    frame_numbers = []
    frames = []

    # Initial settings
    min_frames = 4
    max_frames = 8
    current_frames = 4
    swap_mode = False
    swap_first_frame = None

    # Get initial random different frames
    while len(frame_numbers) < current_frames:
        try:
            frame_num, frame = get_random_frame(cap, total_frames, frame_numbers)
            frame_numbers.append(frame_num)
            frames.append(frame)
        except ValueError as ve:
            print(f"Error: {ve}")
            break

    if len(frames) < min_frames:
        print(f"Error: Unable to retrieve the minimum required frames for {video_path}. Skipping.")
        cap.release()
        return True  # Continue processing

    # Create initial collage
    collage = create_collage(frames)
    collage_filename = "collage_temp.jpg"
    cv2.imwrite(collage_filename, collage)

    # Resize collage for display
    display_collage = resize_image_for_display(collage, max_width=800, max_height=600)

    # Display collage using OpenCV
    window_title = f"Collage for {video_path.name} - Press '+/-' to change frames, 'S' to swap frames, '1'-'{current_frames}' to replace frames, '0' to confirm"
    cv2.imshow(window_title, display_collage)

    print("Please check the collage window.")
    print("Press '+' to increase frames by 2, '-' to decrease frames by 2.")
    print("Press 'S' to swap two frames.")
    print(f"Press '1'-'{current_frames}' to replace specific frames if available.")
    print("Press '0' to confirm and save the final image.")

    while True:
        key = cv2.waitKey(100) & 0xFF  # Wait for 100ms

        # Check if the window has been closed
        if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
            print("Collage window closed by user. Exiting program.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)  # Exit the entire program

        if key != 255:  # 255 indicates no key was pressed
            if swap_mode:
                if key in [ord(str(n)) for n in range(1, current_frames +1)]:
                    selected_frame = int(chr(key)) - 1  # Convert key to index (0-based)
                    if swap_first_frame is None:
                        swap_first_frame = selected_frame
                        print(f"Selected first frame to swap: Frame {selected_frame + 1}")
                    else:
                        swap_second_frame = selected_frame
                        print(f"Selected second frame to swap: Frame {swap_second_frame + 1}")
                        # Swap the frames
                        frames[swap_first_frame], frames[swap_second_frame] = frames[swap_second_frame], frames[swap_first_frame]
                        frame_numbers[swap_first_frame], frame_numbers[swap_second_frame] = frame_numbers[swap_second_frame], frame_numbers[swap_first_frame]
                        print(f"Swapped Frame {swap_first_frame + 1} with Frame {swap_second_frame + 1}.")
                        # Update collage
                        collage = create_collage(frames)
                        cv2.imwrite(collage_filename, collage)
                        display_collage = resize_image_for_display(collage, max_width=800, max_height=600)
                        cv2.imshow(window_title, display_collage)
                        # Reset swap mode
                        swap_mode = False
                        swap_first_frame = None
                else:
                    print(f"Invalid key pressed for swapping. Please press '1'-'{current_frames}'.")
                continue  # Continue to next iteration

            if key == ord('0'):
                break  # Confirm and save
            elif key in [ord('+'), ord('=')]:
                if current_frames + 2 <= max_frames:
                    # Add two frames
                    for _ in range(2):
                        try:
                            new_frame_num, new_frame = get_random_frame(cap, total_frames, frame_numbers)
                            frame_numbers.append(new_frame_num)
                            frames.append(new_frame)
                        except ValueError as ve:
                            print(f"Error: {ve}")
                            break
                    current_frames += 2
                    print(f"Increased frame count to {current_frames}.")
                    # Update collage
                    collage = create_collage(frames)
                    cv2.imwrite(collage_filename, collage)
                    display_collage = resize_image_for_display(collage, max_width=800, max_height=600)
                    cv2.imshow(window_title, display_collage)
                else:
                    print(f"Maximum frame count ({max_frames}) reached.")
            elif key in [ord('-'), ord('_')]:
                if current_frames - 2 >= min_frames:
                    # Remove the last two frames
                    for _ in range(2):
                        frame_numbers.pop()
                        frames.pop()
                    current_frames -= 2
                    print(f"Decreased frame count to {current_frames}.")
                    # Update collage
                    collage = create_collage(frames)
                    cv2.imwrite(collage_filename, collage)
                    display_collage = resize_image_for_display(collage, max_width=800, max_height=600)
                    cv2.imshow(window_title, display_collage)
                else:
                    print(f"Minimum frame count ({min_frames}) reached.")
            elif key in [ord('S'), ord('s')]:
                if current_frames >= 2:
                    swap_mode = True
                    swap_first_frame = None
                    print("Swap mode activated. Please press two frame numbers to swap.")
                else:
                    print("Not enough frames to perform a swap. Need at least 2 frames.")
            elif key in [ord(str(n)) for n in range(1, current_frames +1)]:
                frame_to_replace = int(chr(key)) - 1  # Convert key to index (0-based)
                print(f"Replacing frame {frame_to_replace + 1}...")
                # Exclude current frames to ensure uniqueness
                frame_numbers_set = set(frame_numbers)
                frame_numbers_set.discard(frame_numbers[frame_to_replace])
                try:
                    new_frame_num, new_frame = get_random_frame(cap, total_frames, frame_numbers_set)
                    frame_numbers[frame_to_replace] = new_frame_num
                    frames[frame_to_replace] = new_frame
                    # Update collage
                    collage = create_collage(frames)
                    cv2.imwrite(collage_filename, collage)
                    # Resize collage for display
                    display_collage = resize_image_for_display(collage, max_width=800, max_height=600)
                    cv2.imshow(window_title, display_collage)
                    print(f"Frame {frame_to_replace + 1} replaced.")
                except ValueError as ve:
                    print(f"Error: {ve}")
            else:
                print(f"Invalid key pressed. Please use '+', '-', 'S', '1'-'{current_frames}', or '0'.")

    # Determine the relative path of the video with respect to the input folder
    try:
        relative_path = video_path.relative_to(input_folder)
    except ValueError:
        # If video_path is not under input_folder, use the absolute path
        relative_path = video_path.name

    # Define the path for the collage image in the output folder
    collage_output_path = output_folder / relative_path.parent / (video_path.stem + '.jpg')
    
    # Ensure the output directory exists
    collage_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save final image
    try:
        cv2.imwrite(str(collage_output_path), collage)
        print(f"Final image saved as {collage_output_path}")
    except Exception as e:
        print(f"Error saving final image for {video_path}: {e}")

    # Delete the temporary collage image
    try:
        os.remove(collage_filename)
        print(f"Temporary collage image '{collage_filename}' has been deleted.")
    except Exception as e:
        print(f"Could not delete temporary collage image '{collage_filename}'. Please delete it manually.")
        print(f"Error: {e}")

    cap.release()
    cv2.destroyWindow(window_title)

    # Add video to processed list
    save_processed_video(json_path, video_path)

    return True  # Continue processing

def main():
    """
    Main function to execute the video collage creation process.
    """
    # Define supported video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpg', '.mpeg'}

    # Initialize Tkinter root
    root = Tk()
    root.withdraw()  # Hide the root window

    # Prompt user to select the input folder
    input_folder = filedialog.askdirectory(title="Select Folder Containing Videos")
    if not input_folder:
        messagebox.showinfo("No Folder Selected", "No input folder was selected. Exiting the program.")
        return
    input_folder = Path(input_folder)

    # Prompt user to select the output folder
    output_folder = filedialog.askdirectory(title="Select Output Folder for Collage Images")
    if not output_folder:
        messagebox.showinfo("No Folder Selected", "No output folder was selected. Exiting the program.")
        return
    output_folder = Path(output_folder)

    root.destroy()  # Close the Tkinter root window

    # Define path to JSON file for tracking processed videos in the project folder
    try:
        script_directory = Path(__file__).parent
    except NameError:
        # Fallback if __file__ is not defined (e.g., in interactive mode)
        script_directory = Path.cwd()
    json_filename = "processed_videos.json"
    json_path = script_directory / json_filename

    # Load processed videos
    processed_videos = load_processed_videos(json_path)

    # Get all video files in the input directory recursively
    video_files = get_all_video_files(input_folder, video_extensions)

    if not video_files:
        messagebox.showinfo(
            "No Videos Found",
            f"No video files with extensions {sorted(video_extensions)} were found in '{input_folder}'."
        )
        return

    print(f"Found {len(video_files)} video file(s) in '{input_folder}' and its subdirectories.")

    # Process each video file
    total_videos = len(video_files)
    for index, video_file in enumerate(video_files):
        continue_processing = process_video(
            video_file,
            input_folder,
            output_folder,
            json_path,
            index,
            total_videos
        )
        if not continue_processing:
            break  # Exit the loop if processing should stop

    print("\nAll videos have been processed.")
    messagebox.showinfo("Processing Complete", "All videos have been processed successfully.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
