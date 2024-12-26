"""
Algorithm Pipeline
"""

from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from numpy import ndarray

from app.assigment1.utilities import extract_metadata


def detect_edges(video: cv2.VideoCapture, path: Path) -> Path:
    metadata = extract_metadata(video)
    logger.debug(metadata)
    width, height, fps = metadata["width"], metadata["height"], metadata["fps"]
    output_video = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    while (read := video.read())[0]:
        (_, frame) = read
        new_frame = handle_frame(frame)
        output_video.write(new_frame)

    video.release()
    output_video.release()
    return path


def handle_frame(frame: ndarray) -> ndarray:
    return _mask_frame(frame)


def _mask_frame(frame: ndarray) -> ndarray:
    height, width, _ = frame.shape
    vertices = np.array(
        [[0, height], [width / 2, height / 2], [width, height]],
    )
    mask = np.zeros_like(frame)
    # Fill inside the polygon
    cv2.fillPoly(mask, np.int32([vertices]), (255, 255, 255))
    # Returning the image only where mask pixels match
    return cv2.bitwise_and(frame, mask)
