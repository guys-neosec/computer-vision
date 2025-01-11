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
from app.assigment1.pipeline.crosswalk_detection import crosswalk
from app.assigment1.utilities import extract_metadata, progressbar

OUTPUT_FORMAT = "mp4v"
TRESHOLD = 0.7


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
        self.i = 0
        self.move_lane_cooldown = 0
        self.move_lane = True
        self.right_slope_his = []
        self.left_slope_his = []
        self.inter_x = []
        self.inter_y = []
        self.right_slope_avg = 0.6
        self.left_slope_avg = -0.9
        self.b_history_left_lane = []

    def process(self, output_video: Path) -> None:
        output_video = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*OUTPUT_FORMAT),
            self.fps,
            (self.width, self.height),
        )
        progress_bar = progressbar(self.frame_count)
        for frame in self._get_frames():
            corrected_frame = frame.copy()
            if is_night := self._is_night_frame(frame):
                corrected_frame = self._night_correction(frame)
            mask = self._area_of_interest_mask(is_night)
            isolated_lane_colors = self._isolate_color_lanes(corrected_frame, is_night)
            binary_frame = self._process_for_canny(isolated_lane_colors)
            edges = cv2.Canny(binary_frame, 50, 100)
            roi_binary_frame = cv2.bitwise_and(edges, mask)
            lines = self._hough_transform(roi_binary_frame)

            if lines is None:
                output_video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                continue

            left_lines, right_lines = self._separate_lines(lines)
            left_slope = self._calculate_average_slope(left_lines)
            if not left_slope:
                left_slope = -1.8 if is_night else -1.1
            right_slope = self._calculate_average_slope(right_lines)
            if not right_slope:
                right_slope = 0.8 if is_night else 1.3
            right_slope = self._average_history(right_slope, self.right_slope_his)
            left_slope = self._average_history(left_slope, self.left_slope_his)
            avg_left_line = self._average_lines(left_lines)
            avg_right_line = self._average_lines(right_lines)
            frame_height = frame.shape[0]
            roi_top = 800 if is_night else 750
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
            # self.b_history_left_lane.append(self.get_b(smoothed_left_line))
            if len(self.b_history_left_lane) > 20:
                self.b_history_left_lane.pop(0)
            smoothed_right_line = self._smooth_line(
                extended_right_line,
                self.lanes_history.right,
            )

            # =============== STEP 6: Draw Final Lanes ===============
            if self.move_lane_cooldown > 0:
                self.move_lane_cooldown -= 1
                x, y = self.calculate_intersection(
                    smoothed_left_line,
                    smoothed_right_line,
                )
                x = self._average_history(x, self.inter_x)
                hold_x.append(x)
                if len(hold_x) == 20:
                    # Different approach because we need tighter bound for night
                    if is_night:
                        txt = None
                        max_index = np.argmax(self.b_history_left_lane)
                        min_index = np.argmin(self.b_history_left_lane)
                        threshold = (
                                            self.b_history_left_lane[max_index]
                                            - self.b_history_left_lane[min_index]
                                    ) > 200
                        if threshold and min_index < max_index:
                            txt = "moving left"
                        elif threshold:
                            txt = "moving right"

                    else:
                        count = 0
                        for i in range(len(hold_x) - 1):
                            if hold_x[i] > hold_x[i + 1]:
                                count += 1
                            else:
                                count -= 1
                        if max(hold_x) - min(hold_x) < 40:
                            txt = None
                        if count > 0:
                            txt = "moving left"
                        else:
                            txt = "moving right"
            else:
                threshold = TRESHOLD if not is_night else 1
                num = 15 if not is_night else 6
                self.move_lane = (
                    True
                    if (
                               abs(right_slope - self.right_slope_avg) >= threshold
                               and abs(left_slope + self.left_slope_avg) >= threshold
                       )
                       or abs(right_slope - self.right_slope_avg) > num
                       or abs(left_slope + self.left_slope_avg) > num
                    else False
                )

                if self.move_lane:
                    txt = None
                    self.move_lane_cooldown = 70
                    self.inter_x = []
                    self.inter_y = []
                    self.count_frames_since_change = 0
                    hold_x = []
                    self.b_history_left_lane = []

            if self.move_lane and txt:
                annotated_frame = frame.copy()
                cv2.putText(
                    annotated_frame,
                    txt,
                    (int(self.width / 4), int(self.height / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3.0,
                    (0, 0, 255),
                    5,
                    cv2.LINE_AA,
                )
            else:
                # Draw the lane lines
                annotated_frame = self.draw_lanes_on_frame(
                    frame,
                    smoothed_left_line,
                    smoothed_right_line,
                    fill_color=(0, 255, 0),
                    draw_lane_area=True,
                )

            crosswalk_rectangle = crosswalk(frame)
            if crosswalk_rectangle is not None:
                x1, y1, x2, y2 = crosswalk_rectangle
                overlay = frame.copy()
                color = (255, 0, 255)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                annotated_frame = cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0)
            output_video.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
            progress_bar()
        output_video.release()

    def _get_frames(self) -> Iterable[RBGFrame]:
        logger.debug("Iterating through video frames")
        while (read := self.video.read())[0]:
            (_, frame) = read
            frame: RBGFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame
        self.video.release()
        return

    @staticmethod
    def _night_correction(frame: RBGFrame, gamma=2.5) -> RBGFrame:
        inv_gamma = 1.0 / gamma
        lookup_table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)],
        ).astype("uint8")

        # Apply gamma correction using the lookup table
        corrected_image = cv2.LUT(frame, lookup_table)
        kernel_size = 9

        # Smoothing image, noise from correcting
        return cv2.GaussianBlur(corrected_image, (kernel_size, kernel_size), 0)

        # Display the corrected image

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
    def _is_night_frame(frame: RBGFrame, threshold=80) -> bool:
        # Convert to grayscale
        gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Calculate the average pixel intensity
        avg_intensity = np.mean(gray_image)

        # Check if the average intensity is below the threshold
        return avg_intensity < threshold

    @staticmethod
    def _isolate_color_lanes(frame: RBGFrame, is_night=False) -> RBGFrame:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # White lane HSV range
        lower_white = np.array([0, 0, 160])
        upper_white = np.array([200, 50 if not is_night else 65, 255])

        # Yellow lane HSV range
        lower_yellow = np.array([20, 80, 100])
        upper_yellow = np.array([40, 255, 255])

        # Create masks for white and yellow
        white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)
        yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

        # Combine both masks
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # Apply the combined mask to the original image
        color_isolated = cv2.bitwise_and(frame, frame, mask=combined_mask)

        return color_isolated

    @staticmethod
    def _process_for_canny(frame: RBGFrame) -> GrayScaleFrame:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Define a kernel size for Gaussian smoothing
        kernel_size = 9

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

    def _area_of_interest_mask(self, is_night=False) -> GrayScaleFrame:
        height, width = self.height, self.width
        mask = np.zeros((height, width), dtype=np.uint8)
        # Amit's Video
        # polygon = np.array(
        #     [
        #         [
        #             (int(width * 0.2), height * 0.8),  # Bottom-left
        #             (int(width * 0.35), int(height * 0.55)),  # Top-left
        #             (int(width * 0.55), int(height * 0.55)),  # Top-right
        #             (int(width * 0.8), height * 0.8),  # Bottom-right
        #         ],
        #     ],
        #     dtype=np.int32,
        # )
        # Guys
        polygon = np.array(
            [
                [
                    (int(width * 0.2), height),  # Bottom-left
                    (int(width * 0.5), int(height * 0.6)),  # Top-left
                    (int(width * 0.6), int(height * 0.6)),  # Top-right
                    (int(width * 0.8), height),  # Bottom-right
                ],
            ],
            dtype=np.int32,
        )

        if is_night:
            # Different Video, adjusting ROI
            polygon = np.array(
                [
                    [
                        (int(width * 0.3), height),  # Bottom-left
                        (int(width * 0.3), int(height * 0.75)),  # Top-left
                        (int(width * 0.5), int(height * 0.7)),  # Top-left
                        (int(width * 0.8), height),  # Bottom-right
                    ],
                ],
                dtype=np.int32,
            )

        # Fill polygon with white
        cv2.fillPoly(mask, polygon, [255])

        return mask

    @staticmethod
    def _average_history(
            current_slope: float,
            history: list[float],
            max_history: int = 10,
    ) -> None | float:
        history.append(current_slope)

        if len(history) > max_history:
            history.pop(0)

        # If we have no lines in history, return None
        if len(history) == 0:
            return None

        return np.mean(history)

    def calculate_intersection(
            self,
            left_line: tuple,
            right_line: tuple,
    ) -> tuple[float, float]:
        # Unpack the lines
        x1_left, y1_left, x2_left, y2_left = left_line
        x1_right, y1_right, x2_right, y2_right = right_line

        # Calculate the slopes (m1 and m2)
        m1 = (y2_left - y1_left) / (x2_left - x1_left) if x2_left != x1_left else None
        m2 = (
            (y2_right - y1_right) / (x2_right - x1_right)
            if x2_right != x1_right
            else None
        )

        # Handle vertical lines
        if m1 is None:  # Left line is vertical
            x = x1_left
            y = m2 * x + (y1_right - m2 * x1_right)
        elif m2 is None:  # Right line is vertical
            x = x1_right
            y = m1 * x + (y1_left - m1 * x1_left)
        else:
            # Calculate the y-intercepts (b1 and b2)
            b1 = y1_left - m1 * x1_left
            b2 = y1_right - m2 * x1_right

            # Calculate the intersection point
            x = (b2 - b1) / (m1 - m2)
            y = m1 * x + b1

        return x, y

    @staticmethod
    def get_b(line: Line) -> float:
        x1, y1, x2, y2 = line
        m = (y2 - y1) / (x2 - x1)
        # Calculate y-intercept (b)
        b = y1 - m * x1
        return b

    def draw_lanes_on_frame(
            self,
            frame,
            left_lane,
            right_lane,
            color=(255, 0, 0),
            thickness=8,
            fill_color=None,
            draw_lane_area=False,
    ):
        """
        Draw left and right lanes onto the frame.
        Optionally fill the area in between lanes.
        """
        overlay = frame.copy()

        # Draw the lane lines
        if left_lane is not None:
            cv2.line(
                overlay,
                (left_lane[0], left_lane[1]),
                (left_lane[2], left_lane[3]),
                color,
                thickness,
            )
        if right_lane is not None:
            cv2.line(
                overlay,
                (right_lane[0], right_lane[1]),
                (right_lane[2], right_lane[3]),
                color,
                thickness,
            )

        # Optionally fill the area between the two lanes
        if draw_lane_area and (left_lane is not None) and (right_lane is not None):
            pts = np.array(
                [
                    [left_lane[0], left_lane[1]],
                    [left_lane[2], left_lane[3]],
                    [right_lane[2], right_lane[3]],
                    [right_lane[0], right_lane[1]],
                ],
                dtype=np.int32,
            )

            cv2.fillPoly(overlay, [pts], fill_color)

        # Combine overlay with original using some transparency
        alpha = 0.4
        frame_with_lanes = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return frame_with_lanes
