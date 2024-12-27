"""
Algorithm Pipeline
"""

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from app.assigment1.custom_types import ThreeChannelArray
from app.assigment1.pipeline.color_masking import mask_lanes_colors
from app.assigment1.pipeline.line_drawer import draw_lines
from app.assigment1.pipeline.polygon_masking import mask_polygon
from app.assigment1.utilities import extract_metadata, progressbar

OUTPUT_FORMAT = "mp4v"


def detect_edges(video: cv2.VideoCapture, path: Path) -> Path:
    metadata = extract_metadata(video)
    logger.debug(metadata)
    width, height, fps = metadata.width, metadata.height, metadata.fps
    output_video = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*OUTPUT_FORMAT),
        fps,
        (width, height),
    )
    progress_bar = progressbar(metadata.frame_count)
    for frame in get_frames(video):
        lanes_frame = extract_lanes_feature(frame)
        lines = cv2.HoughLines(
            lanes_frame,
            rho=1,
            theta=np.pi / 180,
            threshold=55,
            srn=0,
            stn=0,
        )
        new_frame = draw_lines(frame, lines)
        output_video.write(cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR))
        progress_bar()

    output_video.release()
    return path


def get_frames(video: cv2.VideoCapture) -> Iterable[ThreeChannelArray]:
    logger.debug("Iterating through video frames")
    while (read := video.read())[0]:
        (_, frame) = read
        frame: ThreeChannelArray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield frame
    video.release()
    return


def extract_lanes_feature(frame: ThreeChannelArray) -> ThreeChannelArray:
    bilateral_filtered_frame = cv2.bilateralFilter(frame, 15, 75, 75)
    masked_colors_frame = mask_lanes_colors(bilateral_filtered_frame)
    gray_frame = cv2.cvtColor(masked_colors_frame, cv2.COLOR_RGB2GRAY)
    cannyed_framed = cv2.Canny(gray_frame, 50, 300)
    kernel = np.ones((3, 3), np.uint8)

    processed_frame = cv2.morphologyEx(cannyed_framed, cv2.MORPH_DILATE, kernel)
    return mask_polygon(processed_frame)
