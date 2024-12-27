import math

import cv2
import numpy as np

from app.assigment1.custom_types import ThreeChannelArray

RGB_CHANNELS = 3
RED = (255, 0, 0)
THICKNESS = 3


def draw_lines(
    frame: ThreeChannelArray,
    lines: list,
    color: tuple[int, int, int] = RED,
    rho_threshold: int = 300,
    theta_threshold: float = np.pi / 16,
) -> ThreeChannelArray:
    # TODO(): This is voodoo, extract to other function and rewrite
    if lines is None or len(lines) == 0:
        return frame
    copied_frame = np.copy(frame)
    line_groups = []
    for line in lines:
        rho, theta = line[0]
        grouped = False

        # Try to find a group for the current line
        for group in line_groups:
            avg_rho, avg_theta, count = group
            if (
                abs(rho - avg_rho) < rho_threshold
                and abs(theta - avg_theta) < theta_threshold
            ):
                # Update the group with the new line
                group[0] = (avg_rho * count + rho) / (count + 1)
                group[1] = (avg_theta * count + theta) / (count + 1)
                group[2] += 1
                grouped = True
                break

        # If no group found, create a new one
        if not grouped:
            line_groups.append([rho, theta, 1])

    # Draw the averaged lines
    for avg_rho, avg_theta, _ in line_groups:
        if avg_theta > np.pi * 3 / 4:
            continue
        a = math.cos(avg_theta)
        b = math.sin(avg_theta)
        x0 = a * avg_rho
        y0 = b * avg_rho
        pt1 = (int(x0 + 2000 * (-b)), int(y0 + 2000 * a))
        pt2 = (int(x0 - 2000 * (-b)), int(y0 - 2000 * a))
        cv2.line(copied_frame, pt1, pt2, color, THICKNESS)
    return copied_frame
