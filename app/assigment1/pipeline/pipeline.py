"""
Algorithm Pipeline
"""

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel

from app.assigment1.custom_types import GrayScaleFrame, Line, RBGFrame
from app.assigment1.loader import load_video
from app.assigment1.pipeline.my_pipeline import draw_lanes_on_frame
from app.assigment1.utilities import extract_metadata, progressbar

OUTPUT_FORMAT = "mp4v"
TRESHOLD = 0.3


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
        self.left_valid_line = None
        self.right_valid_line = None

    def process(self, output_video: Path) -> None:
        output_video = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*OUTPUT_FORMAT),
            self.fps,
            (self.width, self.height),
        )
        progress_bar = progressbar(self.frame_count)
        mask = self._area_of_interest_mask()
        for frame in self._get_frames():
            isolated_lane_colors = self._isolate_color_lanes(frame)
            blur_gray = self._process_for_canny(isolated_lane_colors)
            edges = cv2.Canny(blur_gray, 50, 150)
            roi_edges = cv2.bitwise_and(edges, mask)
            lines = self._hough_transform(roi_edges)

            if lines is None:
                output_video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                continue

            left_lines, right_lines = self._separate_lines(lines)
            left_slope = self._calculate_average_slope(left_lines)
            if not left_slope:
                left_slope = -1.1
            right_slope = self._calculate_average_slope(right_lines)
            if not right_slope:
                right_slope = 0.6

            avg_left_line = self._average_lines(left_lines)
            avg_right_line = self._average_lines(right_lines)
            # print(avg_left_line)
            # print(avg_right_line)
            if abs(right_slope - 0.6) >= TRESHOLD and abs(left_slope + 0.9) < TRESHOLD:
                avg_right_line = self.right_valid_line
            else:
                self.right_valid_line = avg_right_line
            if abs(right_slope - 0.6) < TRESHOLD and abs(left_slope + 0.9) >= TRESHOLD:
                avg_left_line = self.left_valid_line
            else:
                self.left_valid_line = avg_left_line

            frame_height = frame.shape[0]
            roi_top = 435  # Example top limit for ROI
            extended_left_line = self._extend_line_to_full_height(
                avg_left_line,
                roi_top,
                frame_height,
            )
            extended_right_line = self._extend_line_to_full_height(
                avg_right_line,
                roi_top,
                frame_height,
            )

            smoothed_left_line = self._smooth_line(
                extended_left_line,
                self.lanes_history.left,
            )
            smoothed_right_line = self._smooth_line(
                extended_right_line,
                self.lanes_history.right,
            )

            # =============== STEP 6: Draw Final Lanes ===============
            annotated_frame = draw_lanes_on_frame(
                frame,
                smoothed_left_line,
                smoothed_right_line,
                fill_color=(0, 255, 0),  # Optional fill color for area between lanes
                draw_lane_area=True,
            )
            debug_text = (
                f"Left Slope: {left_slope:.2f}"
                if left_slope is not None
                else "Left Slope: None"
            )
            cv2.putText(
                annotated_frame,
                debug_text,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            debug_text = (
                f"Right Slope: {right_slope:.2f}"
                if right_slope is not None
                else "Right Slope: None"
            )
            cv2.putText(
                annotated_frame,
                debug_text,
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            debug_text = (
                "move lane"
                if abs(right_slope - 0.6) >= TRESHOLD
                and abs(left_slope + 0.9) >= TRESHOLD
                else "same lane"
            )
            cv2.putText(
                annotated_frame,
                debug_text,
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            debug_text = (
                "right error"
                if abs(right_slope - 0.6) >= TRESHOLD
                and abs(left_slope + 0.9) < TRESHOLD
                else "right okay"
            )
            cv2.putText(
                annotated_frame,
                debug_text,
                (50, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            debug_text = (
                "left error"
                if abs(right_slope - 0.6) < TRESHOLD
                and abs(left_slope + 0.9) >= TRESHOLD
                else "left okay"
            )
            cv2.putText(
                annotated_frame,
                debug_text,
                (50, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            output_video.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            progress_bar()

    def _get_frames(self) -> Iterable[RBGFrame]:
        logger.debug("Iterating through video frames")
        while (read := self.video.read())[0]:
            (_, frame) = read
            frame: RBGFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame
        self.video.release()
        return

    def _separate_lines(self, lines: list[Line]) -> tuple[list[Line], list[Line]]:
        left_lines = []
        right_lines = []

        # We assume the middle of the image is the dividing line
        # (everything with negative slope or left side is 'left'
        #  positive slope or right side is 'right').
        # Tweak slope thresholds as needed.
        img_center = self.width / 2
        slope_threshold = 0.5

        for line in lines:
            for x1, y1, x2, y2 in line:
                # Avoid dividing by zero
                if (x2 - x1) == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                # Filter out almost-horizontal lines
                if abs(slope) < slope_threshold:
                    continue

                # Decide left vs right by slope sign and x-position
                if slope < 0 and max(x1, x2) < img_center:
                    left_lines.append((x1, y1, x2, y2))
                elif slope > 0 and min(x1, x2) > img_center:
                    right_lines.append((x1, y1, x2, y2))

        return left_lines, right_lines

    @staticmethod
    def _smooth_line(
        current_line: Line,
        history: list[Line],
        max_history: int = 10,
    ) -> None | Line:
        """
        Smooth the line over time using a small history buffer.
        current_line: (x1, y1, x2, y2)
        history_dict: dictionary storing the lines for left/right across frames
        side: 'left' or 'right'
        max_history: how many past lines to store
        """
        if current_line is not None:
            # Store current line
            history.append(current_line)

        # Keep history up to 'max_history' length
        if len(history) > max_history:
            history.pop(0)

        # If we have no lines in history, return None
        if len(history) == 0:
            return None

        # Average over all lines in history
        arr = np.array(history, dtype=np.float32)
        x1_avg = int(np.mean(arr[:, 0]))
        y1_avg = int(np.mean(arr[:, 1]))
        x2_avg = int(np.mean(arr[:, 2]))
        y2_avg = int(np.mean(arr[:, 3]))

        return (x1_avg, y1_avg, x2_avg, y2_avg)

    @staticmethod
    def _extend_line_to_full_height(
        line: None | tuple[int, int, int, int],
        top_limit: int,
        bottom_limit: int,
    ) -> None | tuple[int, int, int, int]:
        if line is None:
            return None

        x1, y1, x2, y2 = line
        if x2 == x1:  # Vertical line
            return (x1, top_limit, x2, bottom_limit)

        slope = (y2 - y1) / (x2 - x1)
        b = y1 - slope * x1  # y = mx + b

        x_top = (top_limit - b) / slope
        x_bottom = (bottom_limit - b) / slope

        return int(x_top), int(top_limit), int(x_bottom), int(bottom_limit)

    @staticmethod
    def _average_lines(lines: list) -> tuple[int, int, int, int]:
        if len(lines) == 0:
            return None

        # Convert to numpy for easier math
        lines_arr = np.array(lines, dtype=np.float32)
        x1_mean = np.mean(lines_arr[:, 0])
        y1_mean = np.mean(lines_arr[:, 1])
        x2_mean = np.mean(lines_arr[:, 2])
        y2_mean = np.mean(lines_arr[:, 3])

        return (int(x1_mean), int(y1_mean), int(x2_mean), int(y2_mean))

    @staticmethod
    def _calculate_average_slope(lines: list) -> float:
        """
        Calculate the average slope of a set of lines.
        """
        if not lines:
            return None
        slopes = []
        for line in lines:
            for x1, y1, x2, y2 in [line]:
                if (x2 - x1) != 0:  # Avoid vertical lines
                    slope = (y2 - y1) / (x2 - x1)
                    slopes.append(slope)

        return np.mean(slopes) if slopes else None

    @staticmethod
    def _isolate_color_lanes(frame: RBGFrame) -> RBGFrame:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Define ranges for yellow and white
        lower_yellow = np.array([20, 80, 80])
        upper_yellow = np.array([40, 150, 150])
        lower_white = np.array([0, 0, 120])
        upper_white = np.array([255, 30, 150])

        # Create masks for yellow and white
        yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
        white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)

        # Combine masks and apply
        combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
        color_isolated = cv2.bitwise_and(hsv_frame, hsv_frame, mask=combined_mask)

        return cv2.cvtColor(color_isolated, cv2.COLOR_HSV2RGB)

    @staticmethod
    def _process_for_canny(frame: RBGFrame) -> GrayScaleFrame:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Define a kernel size for Gaussian smoothing
        kernel_size = 5

        return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    @staticmethod
    def _hough_transform(frame: GrayScaleFrame) -> list:
        # Define Hough Transform parameters
        rho = 2  # Distance resolution in pixels
        theta = np.pi / 180  # Angular resolution in radians
        threshold = 30  # Minimum number of votes
        min_line_length = 20  # Minimum number of pixels making up a line
        max_line_gap = 15  # Maximum gap in pixels between lines

        # Detect lines using Hough Transform
        return cv2.HoughLinesP(
            frame,
            rho,
            theta,
            threshold,
            np.array([]),
            min_line_length,
            max_line_gap,
        )

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
