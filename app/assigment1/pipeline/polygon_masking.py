import cv2
import numpy as np

from app.assigment1.custom_types import GrayScaleFrame


def mask_polygon(frame: GrayScaleFrame) -> GrayScaleFrame:
    (
        height,
        width,
    ) = frame.shape
    # Trapezoid
    polygons = np.array(
        [
            [
                (0, height),
                (width, height),
                (int(width * 0.6), int(height * 0.6)),
                (int(width * 0.4), int(height * 0.6)),
            ],
        ],
        dtype=np.int32,
    )
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygons, [255])
    return cv2.bitwise_and(frame, mask)
