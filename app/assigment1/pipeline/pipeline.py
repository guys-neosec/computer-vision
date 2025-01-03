"""
Algorithm Pipeline
"""

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel

from app.assigment1.custom_types import GrayScaleFrame, RBGFrame
from app.assigment1.loader import load_video
from app.assigment1.utilities import extract_metadata, progressbar

OUTPUT_FORMAT = "mp4v"


class LanesHistory(BaseModel):
    left: list = []
    right: list = []


class Pipeline:
    def __init__(self, input_video: Path) -> None:
        self.video = load_video(input_video)
        if not self.video:
            raise FileNotFoundError
        metadata = extract_metadata(self.video)
        self.height = metadata.height
        self.width = metadata.width
        self.frame_count = metadata.frame_count
        self.fps = metadata.fps
        self.lanes_history = LanesHistory()

    def process(self, output_video: Path) -> None:
        output_video = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*OUTPUT_FORMAT),
            self.fps,
            (self.width, self.height),
        )
        progress_bar = progressbar(self.frame_count)
        # mask = self._area_of_interest_mask()
        for frame in self._get_frames():
            output_video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            progress_bar()

    def _get_frames(self) -> Iterable[RBGFrame]:
        logger.debug("Iterating through video frames")
        while (read := self.video.read())[0]:
            (_, frame) = read
            frame: RBGFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame
        self.video.release()
        return

    def _area_of_interest_mask(self) -> GrayScaleFrame:
        height, width = self.height, self.width
        mask = np.zeros((height, width), dtype=np.uint8)

        polygon = np.array(
            [
                [
                    (int(width * 0.1), height),  # Bottom-left
                    (int(width * 0.45), int(height * 0.6)),  # Top-left
                    (int(width * 0.55), int(height * 0.6)),  # Top-right
                    (int(width * 0.9), height),  # Bottom-right
                ],
            ],
            dtype=np.int32,
        )

        # Fill polygon with white
        cv2.fillPoly(mask, polygon, [255])

        return mask
