import os
import sys
from pathlib import Path
from tkinter import Tk, messagebox, filedialog

import cv2
from collage_from_video.utils.collage_utils import create_collage, resize_image_for_display
from collage_from_video.utils.file_utils import copy_existing_image, find_existing_image, get_all_video_files, \
    load_processed_videos, save_processed_video
from collage_from_video.utils.frame_utils import get_frame_by_number, get_random_frame


class CollageManager:
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
            continue_processing = self.process_video(
                video, self.input_folder, self.output_folder,
                self.json_path, index, len(self.videos_to_process)
            )

            if self.reprocess_current:
                # Insert the same video again in the list for reprocessing
                self.videos_to_process.insert(index + 1, video)
                self.reprocess_current = False

            if not continue_processing:
                break

            index += 1

    def get_available_filename(self, path: Path) -> Path:
        """
        If 'path' already exists, append an index (_1, _2, etc.) until a free filename is found.
        Otherwise, return 'path' as is.
        """
        if not path.exists():
            return path

        suffix = path.suffix
        stem = path.stem
        parent = path.parent
        counter = 1

        while True:
            candidate = parent / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

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
        # Load processed videos
        processed_videos = load_processed_videos(json_path)
        print(f"Json path: {json_path}")

        # Map all processed video paths to their filenames
        processed_videos_filenames = {Path(p).name for p in processed_videos}

        if video_path.name in processed_videos_filenames:
            print(f"\nSkipping already processed video {video_index + 1}/{total_videos}: {video_path.name}")
            return True  # Continue processing

        # Determine the relative path of the video with respect to the input folder
        try:
            relative_path = video_path.relative_to(input_folder)
        except ValueError:
            relative_path = video_path.name

        # Define the path for the collage image in the output folder
        collage_output_path = output_folder / relative_path.parent / (video_path.stem + '.jpg')

        # Ensure the output directory exists
        collage_output_path.parent.mkdir(parents=True, exist_ok=True)

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
        window_title = video_path.name
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 1280, 720)
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
            key = cv2.waitKey(100) & 0xFF

            # Check if the window has been closed
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                print("Collage window closed by user. Exiting program.")
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)

            if key != 255:  # 255 indicates no keypress
                # --- Swap mode handling ---
                if self.swap_mode:
                    if key in [ord(str(n)) for n in range(1, self.current_frames + 1)]:
                        selected_frame = int(chr(key)) - 1
                        if self.swap_first_frame is None:
                            self.swap_first_frame = selected_frame
                            print(f"Selected first frame to swap: Frame {self.swap_first_frame + 1}")
                        else:
                            swap_second_frame = selected_frame
                            print(f"Selected second frame to swap: Frame {swap_second_frame + 1}")
                            frames[self.swap_first_frame], frames[swap_second_frame] = \
                                frames[swap_second_frame], frames[self.swap_first_frame]
                            frame_numbers[self.swap_first_frame], frame_numbers[swap_second_frame] = \
                                frame_numbers[swap_second_frame], frame_numbers[self.swap_first_frame]
                            print(f"Swapped Frame {self.swap_first_frame + 1} "
                                  f"with Frame {swap_second_frame + 1}.")

                            collage = create_collage(frames)
                            cv2.imwrite(collage_filename, collage)
                            display_collage = resize_image_for_display(collage, max_width=1280, max_height=720)
                            cv2.imshow(window_title, display_collage)

                            self.swap_mode = False
                            self.swap_first_frame = None
                    else:
                        print(f"Invalid key pressed for swapping. Please press '1'-'{self.current_frames}'.")
                    continue

                # --- Copy mode handling ---
                if self.copy_mode:
                    if key in [ord(str(n)) for n in range(1, self.current_frames + 1)]:
                        selected_frame = int(chr(key)) - 1
                        if self.copy_source_frame is None:
                            self.copy_source_frame = selected_frame
                            print(f"Selected source frame to copy: Frame {self.copy_source_frame + 1}")
                        else:
                            copy_target_frame = selected_frame
                            print(f"Selected target frame: Frame {copy_target_frame + 1}")

                            frames[copy_target_frame] = frames[self.copy_source_frame].copy()
                            frame_numbers[copy_target_frame] = frame_numbers[self.copy_source_frame]
                            print(f"Copied Frame {self.copy_source_frame + 1} to Frame {copy_target_frame + 1}.")

                            collage = create_collage(frames)
                            cv2.imwrite(collage_filename, collage)
                            display_collage = resize_image_for_display(collage, max_width=1280, max_height=720)
                            cv2.imshow(window_title, display_collage)

                            self.copy_mode = False
                            self.copy_source_frame = None
                    else:
                        print(f"Invalid key pressed for copying. Please press '1'-'{self.current_frames}'.")
                    continue

                # --- Replace mode handling ---
                if self.replace_mode:
                    self.enter_key = 13
                    self.r_keys = [ord('R'), ord('r')]
                    self.z_keys = [ord('Z'), ord('z')]

                    if key == 81:  # Left arrow
                        if frame_numbers[self.frame_to_replace] > 0:
                            frame_numbers[self.frame_to_replace] -= 1
                            frames[self.frame_to_replace] = get_frame_by_number(
                                cap, frame_numbers[self.frame_to_replace]
                            )
                            print(f"Selected previous frame: {frame_numbers[self.frame_to_replace]}")
                            collage = create_collage(frames)
                            cv2.imwrite(collage_filename, collage)
                            display_collage = resize_image_for_display(collage, max_width=1280, max_height=720)
                            cv2.imshow(window_title, display_collage)
                        else:
                            print("Already at the first frame.")
                    elif key == 83:  # Right arrow
                        if frame_numbers[self.frame_to_replace] < total_frames - 1:
                            frame_numbers[self.frame_to_replace] += 1
                            frames[self.frame_to_replace] = get_frame_by_number(
                                cap, frame_numbers[self.frame_to_replace]
                            )
                            print(f"Selected next frame: {frame_numbers[self.frame_to_replace]}")
                            collage = create_collage(frames)
                            cv2.imwrite(collage_filename, collage)
                            display_collage = resize_image_for_display(collage, max_width=1280, max_height=720)
                            cv2.imshow(window_title, display_collage)
                        else:
                            print("Already at the last frame.")
                    elif key in [ord('A'), ord('a')]:
                        # Decrease frame number by 10
                        collage = self.handle_a_key(
                            cap, collage, collage_filename, frame_numbers, self.frame_to_replace,
                            frames, window_title
                        )
                    elif key in [ord('D'), ord('d')]:
                        # Increase frame number by 10
                        collage = self.handle_d_key(
                            cap, collage, collage_filename, frame_numbers, self.frame_to_replace,
                            frames, total_frames, window_title
                        )
                    elif key in [ord('W'), ord('w')]:
                        # Increase frame number by 100
                        collage = self.handle_w_key(
                            cap, collage, collage_filename, frame_numbers, self.frame_to_replace,
                            frames, total_frames, window_title
                        )
                    elif key in [ord('Q'), ord('q')]:
                        # Decrease frame number by 100
                        collage = self.handle_q_key(
                            cap, collage, collage_filename, frame_numbers, self.frame_to_replace,
                            frames, window_title
                        )
                    elif key in [ord('E'), ord('e')]:
                        # Increase frame number by 1000
                        collage = self.handle_e_key(
                            cap, collage, collage_filename, frame_numbers, self.frame_to_replace,
                            frames, total_frames, window_title
                        )
                    elif key in self.z_keys:
                        # Decrease frame number by 1000
                        collage = self.handle_z_key(
                            cap, collage, collage_filename, frame_numbers, self.frame_to_replace,
                            frames, window_title
                        )
                    elif key in self.r_keys:
                        # Replace with a random frame
                        collage = self.handle_r_key(
                            cap, collage, collage_filename, frame_numbers, self.frame_to_replace,
                            frames, total_frames, window_title
                        )
                    elif key == self.enter_key:  # Enter key
                        self.replace_mode = False
                        self.frame_to_replace = None
                        print("Replacement confirmed.")
                    else:
                        print("Invalid key in replace mode. Use arrows or A/D/W/Q/Z/E/R/Enter.")
                    continue

                # --- Skip video handling ---
                if key in [ord('X'), ord('x')]:
                    print("Skip requested. Skipping this video.")
                    self.skip_video = True
                    break  # Exit the key handling loop to skip the video

                # --- Reprocess video handling ---
                if key in [ord('L'), ord('l')]:
                    print("Video will be processed again after current processing.")
                    self.reprocess_current = True
                    # Do not mark as processed; continue to allow saving
                    continue

                # --- Confirm and save handling ---
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
                        self.swap_mode = True
                        self.swap_first_frame = None
                        print("Swap mode activated. Please press two frame numbers to swap.")
                    else:
                        print("Not enough frames to perform a swap. Need at least 2 frames.")
                elif key in [ord('C'), ord('c')]:
                    if self.current_frames >= 2:
                        self.copy_mode = True
                        self.copy_source_frame = None
                        print("Copy mode activated. Please select source frame number, then target frame number.")
                    else:
                        print("Not enough frames to perform a copy. Need at least 2 frames.")
                elif key in [ord(str(n)) for n in range(1, self.current_frames + 1)]:
                    self.frame_to_replace = int(chr(key)) - 1
                    print(f"Replacing frame {self.frame_to_replace + 1}...")
                    self.replace_mode = True
                    print("Use Left/Right arrows to select adjacent frames, 'A'/'D' to jump by 10, "
                          "'W'/'Q' to jump by 100, 'Z'/'E' to jump by 1000, 'R' for random frame, 'Enter' to confirm.")
                else:
                    print("Invalid key pressed. Please use '+', '-', 'S', 'C', '1'-'{self.current_frames}', 'X', or '0'.")

        # After key loop
        if not self.skip_video:
            # Generate a unique filename if the collage already exists
            collage_final_path = self.get_available_filename(collage_output_path)

            try:
                cv2.imwrite(str(collage_final_path), collage)
                print(f"Final image saved as {collage_final_path}")
            except Exception as e:
                print(f"Error saving final image for {video_path.name}: {e}")

            # Delete temporary collage
            try:
                os.remove(collage_filename)
                print(f"Temporary collage image '{collage_filename}' has been deleted.")
            except Exception as e:
                print(f"Could not delete temporary collage image '{collage_filename}'. Please delete manually.")
                print(f"Error: {e}")

            # Only mark as processed if not flagged for reprocessing
            if not self.reprocess_current:
                save_processed_video(json_path, video_path)
        else:
            # If skipping, perform cleanup without saving
            try:
                os.remove(collage_filename)
                print(f"Temporary collage image '{collage_filename}' has been deleted.")
            except Exception as e:
                print(f"Could not delete temporary collage image '{collage_filename}'. Please delete manually.")
                print(f"Error: {e}")

            # Reset skip_video flag for next video
            self.skip_video = False

        cap.release()
        cv2.destroyWindow(window_title)

        return True  # Continue processing

    def handle_a_key(self, cap, collage, collage_filename, frame_numbers, frame_to_replace, frames, window_title):
        if frame_numbers[frame_to_replace] > 0:
            frame_numbers[frame_to_replace] = max(frame_numbers[frame_to_replace] - 10, 0)
            frames[frame_to_replace] = get_frame_by_number(cap, frame_numbers[frame_to_replace])
            print(f"Jumped backward 10 frames to frame number {frame_numbers[frame_to_replace]}")
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
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.mpg', '.mpeg'}

    root = Tk()
    root.withdraw()

    # Decide whether to pick a single file or an entire folder
    pick_single = messagebox.askyesno(
        "Single or Multiple?",
        "Do you want to select a single video file? Click 'Yes' for a single file or 'No' for an entire folder."
    )

    if pick_single:
        single_file = filedialog.askopenfilename(
            title="Select a Single Video File",
            filetypes=[("Video Files", list(video_extensions))]
        )
        if not single_file:
            messagebox.showinfo("No File Selected", "No video file was selected. Exiting the program.")
            return

        input_folder = Path(single_file).parent
        video_files = [Path(single_file)]
    else:
        input_folder = filedialog.askdirectory(title="Select Folder Containing Videos")
        if not input_folder:
            messagebox.showinfo("No Folder Selected", "No input folder was selected. Exiting the program.")
            return
        input_folder = Path(input_folder)

        video_files = get_all_video_files(input_folder, video_extensions)
        if not video_files:
            messagebox.showinfo(
                "No Videos Found",
                f"No video files with extensions {sorted(video_extensions)} were found in '{input_folder}'."
            )
            return

        print(f"Found {len(video_files)} video file(s) in '{input_folder}' and its subdirectories.")

    output_folder = filedialog.askdirectory(title="Select Output Folder for Collage Images")
    if not output_folder:
        messagebox.showinfo("No Folder Selected", "No output folder was selected. Exiting the program.")
        return
    output_folder = Path(output_folder)

    root.destroy()

    try:
        script_directory = Path(__file__).parent
    except NameError:
        script_directory = Path.cwd()

    json_filename = "processed_videos.json"
    json_path = script_directory / json_filename

    collage_manager = CollageManager(video_files, input_folder, output_folder, json_path)
    collage_manager.process_videos()

    print("\nAll videos have been processed.")
    messagebox.showinfo("Processing Complete", "All videos have been processed successfully.")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
