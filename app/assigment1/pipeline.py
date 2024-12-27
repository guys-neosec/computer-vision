"""
Algorithm Pipeline
"""

from pathlib import Path
from typing import Annotated, Literal, TypeVar

import cv2
import numpy as np
import numpy.typing as npt
from loguru import logger

from app.assigment1.utilities import extract_metadata

DType = TypeVar("DType", bound=np.generic)
RGBArray = Annotated[npt.NDArray[DType], Literal["H", "W", 3]]
GrayScaleArray = Annotated[npt.NDArray[DType], Literal["H", "W"]]


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


def handle_frame(frame: RGBArray) -> RGBArray:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cannyed_framed = cv2.Canny(gray_frame, 100, 200)
    return _mask_frame(cannyed_framed)


def _mask_frame(frame: GrayScaleArray) -> GrayScaleArray:
    (
        height,
        width,
    ) = frame.shape
    vertices = np.array(
        [
            [0, height],
            [width * 3 / 8, height / 2],
            [width * 5 / 8, height / 2],
            [width, height],
        ],
    )
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, np.int32([vertices]), (255, 255, 255))
    return cv2.bitwise_and(frame, mask)
