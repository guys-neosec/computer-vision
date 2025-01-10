# Introduction to Computer Vision - README
## Before Starting
0. Make sure you have ffmpeg on your computer
1. Configure a venv from this project, use python 3.12
2. Run:
    ```bash
    pip3 install -r requirements.txt
    pip3 install pre-commit
    pre-commit install
    ```
   This is going to make sure the code is linted and formatted before committing changes
3. Happy work!
## Commiting Code
1. After making your changes, ruff runs as a pre-commit script to catch static bugs
2. Some of the mistakes are fixable by ruff, so run the commit commands couple of times
## Downloading Videos
### Commands
```bash
# Download the video from 3124 seconds (52:04) to 1 minutes later seconds
yt-dlp https://www.youtube.com/watch\?v\=6RPU08WoaxE  --download-sections "*135-200" -f 399  -o "input.mp4"

# Night Video
yt-dlp https://www.youtube.com/watch\?v\=gAejektGusM  --download-sections "*1995-2047" -f 399  -o "input_night.mp4"

```
