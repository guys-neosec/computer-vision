import cv2
import numpy as np

from app.assigment1.custom_types import GrayScaleArray


def mask_polygon(frame: GrayScaleArray) -> GrayScaleArray:
    (
        height,
        width,
    ) = frame.shape
    # Trapezoid
    vertices = np.array(
        [
            [0, height],
            [width * 3 / 8, height / 2],
            [width * 5 / 8, height / 2],
            [width, height],
        ],
    )
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, np.int32([vertices]), [255])
    return cv2.bitwise_and(frame, mask)
