# main.py

import os
import sys
from pathlib import Path
from tkinter import Tk, messagebox, filedialog

import cv2
from collage_from_video.utils.collage_utils import create_collage, resize_image_for_display
from collage_from_video.utils.file_utils import copy_existing_image, find_existing_image, get_all_video_files, \
    load_processed_videos, save_processed_video
from collage_from_video.utils.frame_utils import get_frame_by_number
from collage_from_video.utils.frame_utils import get_random_frame


class CollageManager():
    def __init__(self, videos_to_process, input_folder, output_folder, json_path):
        self.videos_to_process = videos_to_process

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.json_path = json_path

        # Initial settings
        self.min_frames = 2
        self.max_frames = 9
        self.current_frames = 2
        self.swap_mode = False
        self.swap_first_frame = None
        self.copy_mode = False
        self.copy_source_frame = None
        self.replace_mode = False
        self.frame_to_replace = None
        self.skip_video = False  # Flag to determine if the video should be skipped
        self.reprocess_current = False

    def process_videos(self):
        index = 0
        while index < len(self.videos_to_process):
            video = self.videos_to_process[index]
            continue_processing = self.process_video(video, self.input_folder, self.output_folder, 
                                                     self.json_path, index, len(self.videos_to_process))
            
            if self.reprocess_current:
                self.videos_to_process.insert(index+1, video)
                self.reprocess_current = False
            
            if not continue_processing:
                break
            
            index += 1

    def process_video(self, video_path, input_folder, output_folder, json_path, video_index, total_videos):
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
        # Load processed videos (full paths)
        processed_videos = load_processed_videos(json_path)

        print(f"Json path: {json_path}")

        # Map all processed video paths to their filenames
        processed_videos_filenames = {Path(p).name for p in processed_videos}

        # Check if the current video's filename is in the processed set
        if video_path.name in processed_videos_filenames:
            print(f"\nSkipping already processed video {video_index + 1}/{total_videos}: {video_path.name}")
            return True  # Continue processing

        # Determine the relative path of the video with respect to the input folder
        try:
            relative_path = video_path.relative_to(input_folder)
        except ValueError:
            # If video_path is not under input_folder, use the filename
            relative_path = video_path.name

        # Define the path for the collage image in the output folder
        collage_output_path = output_folder / relative_path.parent / (video_path.stem + '.jpg')

        # Ensure the output directory exists
        collage_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if an image with the same name exists in output_folder recursively
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        existing_image_path = find_existing_image(output_folder, video_path.stem, image_extensions)

        # if existing_image_path is not None:
        #     if existing_image_path != collage_output_path:
        #         # Copy the image to the collage_output_path
        #         copy_existing_image(existing_image_path, collage_output_path)
        #     else:
        #         print(f"Image already exists at {collage_output_path}")
        #     # Mark the video as processed using its full path
        #     save_processed_video(json_path, video_path)
        #     return True  # Continue processing

        print(f"\nProcessing video {video_index + 1}/{total_videos}: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}. Skipping.")
            return True  # Continue processing

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 2:
            print(f"Error: Video {video_path} does not have enough frames. Skipping.")
            cap.release()
            return True  # Continue processing

        # Initialize frame numbers and frames
        frame_numbers = []
        frames = []

        # Get initial random different frames
        while len(frame_numbers) < self.current_frames:
            try:
                frame_num, frame = get_random_frame(cap, total_frames, frame_numbers)
                frame_numbers.append(frame_num)
                frames.append(frame)
            except ValueError as ve:
                print(f"Error: {ve}")
                break

        if len(frames) < self.min_frames:
            print(f"Error: Unable to retrieve the minimum required frames for {video_path.name}. Skipping.")
            cap.release()
            return True  # Continue processing

        # Create initial collage
        collage = create_collage(frames)
        collage_filename = "collage_temp.jpg"
        cv2.imwrite(collage_filename, collage)

        # Resize collage for display
        display_collage = resize_image_for_display(collage, max_width=1280, max_height=720)

        # Display collage using OpenCV
        window_title = video_path.name  # Set window title to video name
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)  # Make window resizable
        cv2.resizeWindow(window_title, 1280, 720)  # Set initial window size
        cv2.imshow(window_title, display_collage)

        print("Please check the collage window.")
        print("Press '+' to increase frames by 1, '-' to decrease frames by 1.")
        print("Press 'S' to swap two frames.")
        print("Press 'C' to copy one frame to another.")
        print(f"Press '1'-'{self.current_frames}' to replace specific frames.")
        print("Press 'X' to skip this video.")
        print("Press 'L' to process this video again after current processing.")
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
                if self.swap_mode:
                    if key in [ord(str(n)) for n in range(1, self.current_frames + 1)]:
                        selected_frame = int(chr(key)) - 1  # Convert key to index (0-based)
                        if swap_first_frame is None:
                            swap_first_frame = selected_frame
                            print(f"Selected first frame to swap: Frame {swap_first_frame + 1}")
                        else:
                            swap_second_frame = selected_frame
                            print(f"Selected second frame to swap: Frame {swap_second_frame + 1}")
                            # Swap the frames
                            frames[swap_first_frame], frames[swap_second_frame] = frames[swap_second_frame], frames[
                                swap_first_frame]
                            frame_numbers[swap_first_frame], frame_numbers[swap_second_frame] = frame_numbers[
                                swap_second_frame], frame_numbers[swap_first_frame]
                            print(f"Swapped Frame {swap_first_frame + 1} with Frame {swap_second_frame + 1}.")
                            # Update collage
                            collage = create_collage(frames)
                            cv2.imwrite(collage_filename, collage)
                            display_collage = resize_image_for_display(collage, max_width=1280, max_height=720)
                            cv2.imshow(window_title, display_collage)
                            # Reset swap mode
                            swap_mode = False
                            swap_first_frame = None
                    else:
                        print(f"Invalid key pressed for swapping. Please press '1'-'{self.current_frames}'.")
                    continue  # Continue to next iteration

                if self.copy_mode:
                    if key in [ord(str(n)) for n in range(1, self.current_frames + 1)]:
                        selected_frame = int(chr(key)) - 1  # Convert key to index (0-based)
                        if copy_source_frame is None:
                            copy_source_frame = selected_frame
                            print(f"Selected source frame to copy: Frame {copy_source_frame + 1}")
                        else:
                            copy_target_frame = selected_frame
                            print(f"Selected target frame: Frame {copy_target_frame + 1}")
                            # Copy the frame
                            frames[copy_target_frame] = frames[copy_source_frame].copy()
                            frame_numbers[copy_target_frame] = frame_numbers[copy_source_frame]
                            print(f"Copied Frame {copy_source_frame + 1} to Frame {copy_target_frame + 1}.")
                            # Update collage
                            collage = create_collage(frames)
                            cv2.imwrite(collage_filename, collage)
                            display_collage = resize_image_for_display(collage, max_width=1280, max_height=720)
                            cv2.imshow(window_title, display_collage)
                            # Reset copy mode
                            copy_mode = False
                            copy_source_frame = None
                    else:
                        print(f"Invalid key pressed for copying. Please press '1'-'{self.current_frames}'.")
                    continue  # Continue to next iteration

                if self.replace_mode:
                    # Handle keypresses in replace mode
                    self.enter_key = 13
                    self.r_keys = [ord('R'), ord('r')]
                    self.z_keys = [ord('Z'), ord('z')]
                    if key == 81:  # Left arrow key
                        if frame_numbers[frame_to_replace] > 0:
                            frame_numbers[frame_to_replace] -= 1
                            frames[frame_to_replace] = get_frame_by_number(cap, frame_numbers[frame_to_replace])
                            print(f"Selected previous frame: Frame number {frame_numbers[frame_to_replace]}")
                            # Update collage
                            collage = create_collage(frames)
                            cv2.imwrite(collage_filename, collage)
                            display_collage = resize_image_for_display(collage, max_width=1280, max_height=720)
                            cv2.imshow(window_title, display_collage)
                        else:
                            print("Already at the first frame.")
                    elif key == 83:  # Right arrow key
                        if frame_numbers[frame_to_replace] < total_frames - 1:
                            frame_numbers[frame_to_replace] += 1
                            frames[frame_to_replace] = get_frame_by_number(cap, frame_numbers[frame_to_replace])
                            print(f"Selected next frame: Frame number {frame_numbers[frame_to_replace]}")
                            # Update collage
                            collage = create_collage(frames)
                            cv2.imwrite(collage_filename, collage)
                            display_collage = resize_image_for_display(collage, max_width=1280, max_height=720)
                            cv2.imshow(window_title, display_collage)
                        else:
                            print("Already at the last frame.")
                    elif key in [ord('A'), ord('a')]:
                        # Decrease frame number by 10
                        collage = self.handle_a_key(cap, collage, collage_filename, frame_numbers, frame_to_replace,
                                                    frames, window_title)
                    elif key in [ord('D'), ord('d')]:
                        # Increase frame number by 10
                        collage = self.handle_d_key(cap, collage, collage_filename, frame_numbers, frame_to_replace,
                                                    frames, total_frames, window_title)
                    elif key in [ord('W'), ord('w')]:
                        # Increase frame number by 100
                        collage = self.handle_w_key(cap, collage, collage_filename, frame_numbers, frame_to_replace,
                                                    frames, total_frames, window_title)
                    elif key in [ord('Q'), ord('q')]:
                        # Decrease frame number by 100
                        collage = self.handle_q_key(cap, collage, collage_filename, frame_numbers, frame_to_replace,
                                                    frames, window_title)
                    elif key in [ord('E'), ord('e')]:
                        # Increase frame number by 1000
                        collage = self.handle_e_key(cap, collage, collage_filename, frame_numbers, frame_to_replace,
                                                    frames, total_frames, window_title)

                    elif key in self.z_keys:
                        # Decrease frame number by 1000
                        collage = self.handle_z_key(cap, collage, collage_filename, frame_numbers, frame_to_replace,
                                                    frames, window_title)
                    elif key in self.r_keys:
                        # Replace with a random frame
                        collage = self.handle_r_key(cap, collage, collage_filename, frame_numbers, frame_to_replace,
                                                    frames, total_frames, window_title)
                    elif key == self.enter_key:  # Enter key
                        # Confirm replacement
                        replace_mode = False
                        frame_to_replace = None
                        print("Replacement confirmed.")
                    else:
                        print(
                            "Invalid key in replace mode. Use Left/Right arrows, 'A'/'D', 'W'/'Q', 'Z'/'E', 'R', or 'Enter'.")
                    continue  # Skip the rest of the loop

                # Check for skip key ('X' or 'x')
                if key in [ord('X'), ord('x')]:
                    print("Skip requested. Skipping this video.")
                    skip_video = True
                    break  # Exit the key handling loop to skip the video
                
                if key in [ord('L'), ord('l')]:
                    print("Video will be processed again after current processing.")
                    self.reprocess_current = True
                    continue
                

                if key == ord('0'):
                    break  # Confirm and save
                elif key in [ord('+'), ord('=')]:
                    if self.current_frames + 1 <= self.max_frames:
                        try:
                            new_frame_num, new_frame = get_random_frame(cap, total_frames, frame_numbers)
                            frame_numbers.append(new_frame_num)
                            frames.append(new_frame)
                            self.current_frames += 1
                            print(f"Increased frame count to {self.current_frames}.")
                        except ValueError as ve:
                            print(f"Error: {ve}")
                        collage = create_collage(frames)
                        cv2.imwrite(collage_filename, collage)
                        display_collage = resize_image_for_display(collage, max_width=1280, max_height=720)
                        print(f"Press '1'-'{self.current_frames}' to replace specific frames.")
                        cv2.imshow(window_title, display_collage)
                    else:
                        print(f"Maximum frame count ({self.max_frames}) reached.")
                elif key in [ord('-'), ord('_')]:
                    if self.current_frames - 1 >= self.min_frames:
                        frame_numbers.pop()
                        frames.pop()
                        self.current_frames -= 1
                        print(f"Decreased frame count to {self.current_frames}.")
                        collage = create_collage(frames)
                        cv2.imwrite(collage_filename, collage)
                        display_collage = resize_image_for_display(collage, max_width=1280, max_height=720)
                        print(f"Press '1'-'{self.current_frames}' to replace specific frames.")
                        cv2.imshow(window_title, display_collage)
                    else:
                        print(f"Minimum frame count ({self.min_frames}) reached.")
                elif key in [ord('S'), ord('s')]:
                    if self.current_frames >= 2:
                        swap_mode = True
                        swap_first_frame = None
                        print("Swap mode activated. Please press two frame numbers to swap.")
                    else:
                        print("Not enough frames to perform a swap. Need at least 2 frames.")
                elif key in [ord('C'), ord('c')]:
                    if self.current_frames >= 2:
                        copy_mode = True
                        copy_source_frame = None
                        print("Copy mode activated. Please select source frame number, then target frame number.")
                    else:
                        print("Not enough frames to perform a copy. Need at least 2 frames.")
                elif key in [ord(str(n)) for n in range(1, self.current_frames + 1)]:
                    frame_to_replace = int(chr(key)) - 1  # Convert key to index (0-based)
                    print(f"Replacing frame {frame_to_replace + 1}...")
                    replace_mode = True
                    print(
                        "Use Left/Right arrows to select adjacent frames, 'A'/'D' to jump frames by 10, 'W'/'Q' to jump frames by 100, 'Z'/'E' to jump frames by 1000, 'R' for random frame, 'Enter' to confirm.")  # The actual replacement will happen in the replace_mode
                else:
                    print(
                        f"Invalid key pressed. Please use '+', '-', 'S', 'C', '1'-'{self.current_frames}', 'X', or '0'.")

        # After exiting the key handling loop
        if not self.skip_video and not self.reprocess_current:
            # Save final image
            try:
                cv2.imwrite(str(collage_output_path), collage)
                print(f"Final image saved as {collage_output_path}")
            except Exception as e:
                print(f"Error saving final image for {video_path.name}: {e}")

            # Delete the temporary collage image
            try:
                os.remove(collage_filename)
                print(f"Temporary collage image '{collage_filename}' has been deleted.")
            except Exception as e:
                print(f"Could not delete temporary collage image '{collage_filename}'. Please delete it manually.")
                print(f"Error: {e}")

            cap.release()
            cv2.destroyWindow(window_title)

            # Add video to processed list (saving full path as before)
            save_processed_video(json_path, video_path)
        else:
            # If skipping, perform cleanup without saving
            try:
                os.remove(collage_filename)
                print(f"Temporary collage image '{collage_filename}' has been deleted.")
            except Exception as e:
                print(f"Could not delete temporary collage image '{collage_filename}'. Please delete it manually.")
                print(f"Error: {e}")

            cap.release()
            cv2.destroyWindow(window_title)

        return True  # Continue processing

    def handle_a_key(self, cap, collage, collage_filename, frame_numbers, frame_to_replace, frames, window_title):
        if frame_numbers[frame_to_replace] > 0:
            frame_numbers[frame_to_replace] = max(frame_numbers[frame_to_replace] - 10, 0)
            frames[frame_to_replace] = get_frame_by_number(cap, frame_numbers[frame_to_replace])
            print(f"Jumped backward 10 frames to frame number {frame_numbers[frame_to_replace]}")
            # Update collage
            collage = create_collage(frames)
            cv2.imwrite(collage_filename, collage)
            display_collage = resize_image_for_display(collage, max_width=1280, max_height=720)
            cv2.imshow(window_title, display_collage)
        else:
            print("Already at the first frame.")
        return collage

    def handle_r_key(self, cap, collage, collage_filename, frame_numbers, frame_to_replace, frames, total_frames,
                     window_title):
        frame_numbers_set = set(frame_numbers)
        frame_numbers_set.discard(frame_numbers[frame_to_replace])
        try:
            new_frame_num, new_frame = get_random_frame(cap, total_frames, frame_numbers_set)
            frame_numbers[frame_to_replace] = new_frame_num
            frames[frame_to_replace] = new_frame
            print(f"Replaced with random frame number {new_frame_num}")
            # Update collage
            collage = create_collage(frames)
            cv2.imwrite(collage_filename, collage)
            display_collage = resize_image_for_display(collage, max_width=1280, max_height=720)
            cv2.imshow(window_title, display_collage)
        except ValueError as ve:
            print(f"Error: {ve}")
        return collage

    def handle_z_key(self, cap, collage, collage_filename, frame_numbers, frame_to_replace, frames, window_title):
        if frame_numbers[frame_to_replace] > 0:
            frame_numbers[frame_to_replace] = max(frame_numbers[frame_to_replace] - 1000, 0)
            frames[frame_to_replace] = get_frame_by_number(cap, frame_numbers[frame_to_replace])
            print(f"Jumped backward 1000 frames to frame number {frame_numbers[frame_to_replace]}")
            # Update collage
            collage = create_collage(frames)
            cv2.imwrite(collage_filename, collage)
            display_collage = resize_image_for_display(collage, max_width=1280, max_height=720)
            cv2.imshow(window_title, display_collage)
        else:
            print("Already at the first frame.")
        return collage

    def handle_e_key(self, cap, collage, collage_filename, frame_numbers, frame_to_replace, frames, total_frames,
                     window_title):
        if frame_numbers[frame_to_replace] < total_frames - 1:
            frame_numbers[frame_to_replace] = min(frame_numbers[frame_to_replace] + 1000, total_frames - 1)
            frames[frame_to_replace] = get_frame_by_number(cap, frame_numbers[frame_to_replace])
            print(f"Jumped forward 1000 frames to frame number {frame_numbers[frame_to_replace]}")
            # Update collage
            collage = create_collage(frames)
            cv2.imwrite(collage_filename, collage)
            display_collage = resize_image_for_display(collage, max_width=1280, max_height=720)
            cv2.imshow(window_title, display_collage)
        else:
            print("Already at the last frame.")
        return collage

    def handle_q_key(self, cap, collage, collage_filename, frame_numbers, frame_to_replace, frames, window_title):
        if frame_numbers[frame_to_replace] > 0:
            frame_numbers[frame_to_replace] = max(frame_numbers[frame_to_replace] - 100, 0)
            frames[frame_to_replace] = get_frame_by_number(cap, frame_numbers[frame_to_replace])
            print(f"Jumped backward 100 frames to frame number {frame_numbers[frame_to_replace]}")
            # Update collage
            collage = create_collage(frames)
            cv2.imwrite(collage_filename, collage)
            display_collage = resize_image_for_display(collage, max_width=1280, max_height=720)
            cv2.imshow(window_title, display_collage)
        else:
            print("Already at the first frame.")
        return collage

    def handle_w_key(self, cap, collage, collage_filename, frame_numbers, frame_to_replace, frames, total_frames,
                     window_title):
        if frame_numbers[frame_to_replace] < total_frames - 1:
            frame_numbers[frame_to_replace] = min(frame_numbers[frame_to_replace] + 100, total_frames - 1)
            frames[frame_to_replace] = get_frame_by_number(cap, frame_numbers[frame_to_replace])
            print(f"Jumped forward 100 frames to frame number {frame_numbers[frame_to_replace]}")
            # Update collage
            collage = create_collage(frames)
            cv2.imwrite(collage_filename, collage)
            display_collage = resize_image_for_display(collage, max_width=1280, max_height=720)
            cv2.imshow(window_title, display_collage)
        else:
            print("Already at the last frame.")
        return collage

    def handle_d_key(self, cap, collage, collage_filename, frame_numbers, frame_to_replace, frames, total_frames,
                     window_title):
        if frame_numbers[frame_to_replace] < total_frames - 1:
            frame_numbers[frame_to_replace] = min(frame_numbers[frame_to_replace] + 10, total_frames - 1)
            frames[frame_to_replace] = get_frame_by_number(cap, frame_numbers[frame_to_replace])
            print(f"Jumped forward 10 frames to frame number {frame_numbers[frame_to_replace]}")
            # Update collage
            collage = create_collage(frames)
            cv2.imwrite(collage_filename, collage)
            display_collage = resize_image_for_display(collage, max_width=1280, max_height=720)
            cv2.imshow(window_title, display_collage)
        else:
            print("Already at the last frame.")
        return collage


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


    # Get all video files in the input directory recursively
    video_files = get_all_video_files(input_folder, video_extensions)

    if not video_files:
        messagebox.showinfo("No Videos Found",
            f"No video files with extensions {sorted(video_extensions)} were found in '{input_folder}'.")
        return

    print(f"Found {len(video_files)} video file(s) in '{input_folder}' and its subdirectories.")

    # Process each video file
    total_videos = len(video_files)

    collage_manager = CollageManager(video_files, input_folder, output_folder, json_path)

    collage_manager.process_videos()

    print("\nAll videos have been processed.")
    messagebox.showinfo("Processing Complete", "All videos have been processed successfully.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
