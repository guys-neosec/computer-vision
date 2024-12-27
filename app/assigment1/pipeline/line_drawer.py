import math

import cv2
import numpy as np

from app.assigment1.custom_types import ThreeChannelArray

RGB_CHANNELS = 3
RED = (255, 0, 0)


def draw_lines(
    frame: ThreeChannelArray,
    lines: list,
    color: tuple[int, int, int] = RED,
    thickness: int = 3,
) -> ThreeChannelArray:
    if lines is None or len(lines) == 0:
        return frame
    # height, width = frame.shape[:2]
    # grouped_lines = _group_lines((height, width), lines)
    copied_frame = np.copy(frame)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(copied_frame, (x1, y1), (x2, y2), color, thickness)
    return copied_frame


def _group_lines(dimensions: tuple[int, int], lines: list) -> list:
    threshold = 0.5
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if math.fabs(slope) < threshold:
                continue
            if slope <= 0:
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
    min_y = dimensions[0] * (3 / 5)
    max_y = dimensions[0]
    poly_left = np.poly1d(
        np.polyfit(
            left_line_y,
            left_line_x,
            deg=1,
        ),
    )
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
    poly_right = np.poly1d(
        np.polyfit(
            right_line_y,
            right_line_x,
            deg=1,
        ),
    )
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))
    return [
        [
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y],
        ],
    ]
