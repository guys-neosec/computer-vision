from typing import Any

import cv2
import numpy as np

from app.assigment1.custom_types import ThreeChannelArray

RGB_CHANNELS = 3
RED = (255, 0, 0)


def draw_lines(
    frame: ThreeChannelArray,
    lines: Any,
    color: tuple[int, int, int] = RED,
    thickness: int = 3,
) -> ThreeChannelArray:
    if lines is None or len(lines) == 0:
        return frame
    copied_frame = np.copy(frame)
    (
        height,
        width,
        _,
    ) = frame.shape
    line_img = np.zeros(
        (
            height,
            width,
            RGB_CHANNELS,
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return cv2.addWeighted(copied_frame, 0.8, line_img, 1.0, 0.0)
