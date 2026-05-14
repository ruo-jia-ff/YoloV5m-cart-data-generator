# YOLOv5m Cart Data Generator

A modified POPS backend for generating YOLOv5m training data from video footage.

## Setup

1. Run the PowerShell setup script to create a local virtual environment (allow a few minutes to complete):
   ```powershell
   ./setup_venv.ps1
   ```

2. Prepare a folder containing the videos you want to use as source data. The script looks for a `sample_videos` folder by default.

3. Activate the virtual environment and run the script:
   ```bash
   python prepare_yolo_dataset.py
   ```

## Parameters

| Parameter | Description |
|---|---|
| `--sample-every n` | Sample one frame every `n` frames | (default: 'n = 5')
| `--conf-thresh x` | Minimum confidence score `x` required to generate a bounding box and label (default: 'x = 0.75') |
| `--data-folder z` | Path to the video folder (default: `sample_videos/`) |

**Example:**
```bash
python prepare_yolo_dataset.py --sample-every 10 --conf-thresh 0.5 --data-folder path/to/videos/
```
