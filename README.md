# Video Collage Path Handling

## How Paths Are Handled

When selecting input and output folders, the tool maintains the original directory structure by using relative paths. Here's how it works:

### Input/Output Path Resolution

1. User selects:
   - An input folder containing videos
   - An output folder for saving collages

2. For each video file, the tool:
   - Determines the relative path of the video file within the input folder structure
   - Preserves this same structure when saving to the output folder

## Key Features

### Frame Management
- Add/remove frames (2-9 frames per collage)
- Swap frames positions
- Copy frames to other positions
- Replace individual frames

### Navigation Controls
- Arrow keys: Move frame by frame
- A/D: Jump 10 frames
- W/Q: Jump 100 frames
- E/Z: Jump 1000 frames
- R: Random frame
- X: Skip video
- L: Reprocess video
- 0: Save and continue

## Installation

```bash
pip install collage-from-video


