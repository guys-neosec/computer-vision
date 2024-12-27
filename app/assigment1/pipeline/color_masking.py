import cv2
import numpy as np

from app.assigment1.custom_types import ThreeChannelArray


def mask_lanes_colors(image: ThreeChannelArray) -> ThreeChannelArray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([180, 25, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_yellow = np.array([15, 30, 100], dtype=np.uint8)
    upper_yellow = np.array([35, 204, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    combined_mask = cv2.bitwise_or(mask_white, mask_yellow)

    return cv2.bitwise_and(image, image, mask=combined_mask)
