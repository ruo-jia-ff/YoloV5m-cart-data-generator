# YoloV5m-cart-data-generator
Modification of POPS backend to generate training data for the yolov5m model. 

## Instructions
1. Set up local venv with setup_venv.ps1 (powershell script) It will take a few minutes.
2. Make sure you have a video folder for videos that will become data. (Default of script is to look for a **sample_videos** folder)
3. Activate the venv and run python prepare_yolo_dataset.py

## Parameters
a. --sample-every n (n = how many frames)

b. --conf-thresh x (x = how high a threshold to generate a bounding box and label)

c. --data-folder z (z = path/to/videos/)
