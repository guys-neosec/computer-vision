from pathlib import Path

import cv2
from loguru import logger

MP4_EXTENSION = ".mp4"


def load_video(path: Path) -> None | cv2.VideoCapture:
    logger.debug(f"Loading video from {path}")
    file_extension = path.suffix
    if file_extension != MP4_EXTENSION:
        logger.error(f"File {path} is not {MP4_EXTENSION}")
        return None

    if not path.exists():
        logger.error(f"Video file {path} not found")
        return None
    return cv2.VideoCapture(str(path))
