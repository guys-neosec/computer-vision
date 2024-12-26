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
# Download the video from 1434 seconds (23:54) to 1550 seconds
yt-dlp https://www.youtube.com/watch?v=eoXguTDnnHM --download-sections "*1434-1550" -f best -o "input.mp4"
```
### Parameters Explanation
1. `-ss` 1434: Specifies the start time in seconds (1434 seconds = 23 minutes 54 seconds).
2. `-to` 1550: Specifies the end time in seconds (1550 seconds = 25 minutes 50 seconds).
3. `-i`: Input file or URL.
4. `$(youtube-dl -f best -g URL)`: Fetches the direct video URL using youtube-dl. Replace URL with the YouTube video link.
5. `-c:v libx264`: Encodes the video using the H.264 codec.
6. `-c:a mp3`: Encodes the audio using MP3 codec (to ensure AVI compatibility).
7. `output.avi`: The output file name in AVI format.
