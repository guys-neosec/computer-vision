"""
Algorithm Pipeline
"""

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
    while (read := video.read())[0]:
        (_, frame) = read
        frame: ThreeChannelArray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        modified_frame = transform_frame(frame)
        lines = cv2.HoughLinesP(
            modified_frame,
            rho=6,
            theta=np.pi / 60,
            threshold=160,
            lines=np.array([]),
            minLineLength=40,
            maxLineGap=25,
        )
        new_frame = draw_lines(frame, lines)
        output_video.write(cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR))
        progress_bar()

    video.release()
    output_video.release()
    return path


def transform_frame(frame: ThreeChannelArray) -> ThreeChannelArray:
    masked_colors_frame = mask_lanes_colors(frame)
    gray_frame = cv2.cvtColor(masked_colors_frame, cv2.COLOR_RGB2GRAY)
    cannyed_framed = cv2.Canny(gray_frame, 100, 200)
    return mask_polygon(cannyed_framed)
