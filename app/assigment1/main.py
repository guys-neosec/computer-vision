from pathlib import Path

from loguru import logger

from app.assigment1.loader import load_video
from app.assigment1.pipeline import detect_edges
from app.assigment1.player import play_video

PATH = "/Users/gstrauss/Reichman_University/computer-vision"
INPUT_VIDEO = Path(PATH) / "input.mp4"
OUTPUT_VIDEO = Path(PATH) / "output.mp4"


def main() -> None:
    logger.info("Starting")
    video = load_video(INPUT_VIDEO)
    if video is None:
        return

    detect_edges(video, OUTPUT_VIDEO)
    video = load_video(OUTPUT_VIDEO)
    if video is None:
        return

    play_video(video)


if __name__ == "__main__":
    main()
