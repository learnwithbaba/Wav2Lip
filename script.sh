#!/bin/bash

# Function to prompt user for input
prompt() {
    local PROMPT="$1"
    local VAR
    read -p "$PROMPT" VAR
    echo "$VAR"
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
else
    echo "venv folder already exists, skipping creation..."
fi

# Activate the virtual environment
source venv/bin/activate

# Define Python executable within the virtual environment
PYTHON="venv/bin/python"

echo "venv $PYTHON"
$PYTHON -m pip install -r cuda_requirements.txt
$PYTHON -m pip install -r requirements.txt

$PYTHON download_models.py

# Prompt user for video and audio paths
VIDEO_PATH=$(prompt "Enter the path to your video file (e.g., /path/to/your/video/file.mp4): ")
AUDIO_PATH=$(prompt "Enter the path to your audio file (e.g., /path/to/your/audio/file.wav): ")

echo "Video Path: $VIDEO_PATH"
echo "Audio Path: $AUDIO_PATH"

$PYTHON inference.py --checkpoint_path "checkpoints/wav2lip_gan.pth" --face "$VIDEO_PATH" --audio "$AUDIO_PATH"

echo
echo "Launch unsuccessful. Exiting."
