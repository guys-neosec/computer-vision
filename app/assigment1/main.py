from pathlib import Path

from loguru import logger

from app.assigment1.loader import load_video
from app.assigment1.player import play_video
from app.assigment1.utilities import extract_metadata

PATH = "/Users/gstrauss/Reichman_University/computer-vision/output.mp4"


def main() -> None:
    logger.info("Starting")
    video_path = Path(PATH)
    video = load_video(video_path)
    if video is None:
        return
    metadata = extract_metadata(video)
    logger.debug(metadata)
    play_video(video)


if __name__ == "__main__":
    main()
