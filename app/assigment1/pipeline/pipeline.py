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
from app.assigment1.utilities import extract_metadata, progressbar
from app.assigment1.pipeline.my_pipeline import draw_lanes_on_frame

OUTPUT_FORMAT = "mp4v"
TRESHOLD = 0.4


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
        self.i = 0
        self.move_lane_cooldown = 0
        self.move_lane = True

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
            if self._is_night_frame(frame):
                corrected_frame = self._night_correction(frame)

            mask = self._area_of_interest_mask()
            isolated_lane_colors = self._isolate_color_lanes(corrected_frame)
            binary_frame = self._process_for_canny(isolated_lane_colors)
            roi_binary_frame = cv2.bitwise_and(binary_frame, mask)
            lines = self._hough_transform(roi_binary_frame)

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
            if self.move_lane_cooldown > 0 :
                self.move_lane_cooldown -=1
            else:
                self.move_lane  =  True if (abs(right_slope - 0.6) >= TRESHOLD \
                    and abs(left_slope + 0.9) >= TRESHOLD) or \
                    abs(right_slope - 0.6) > 2 or abs(left_slope +0.9) >2 else False
                if self.move_lane:
                    self.move_lane_cooldown = 30
                    print(left_slope)
                    print(right_slope)
                    cv2.imshow("Corrected Frame", frame)
                    cv2.waitKey(0)  # Wait for a key press
                    cv2.destroyAllWindows()

            if self.move_lane:
                # Annotate the frame with "Moving Lane" in big red text
                annotated_frame = frame.copy()  # Copy the frame to annotate
                cv2.putText(
                    annotated_frame,
                    "MOVING LANE",  # Text to display
                    (int(self.width / 4), int(self.height / 2)),  # Position (centered horizontally)
                    cv2.FONT_HERSHEY_SIMPLEX,  # Font style
                    2.0,  # Font size
                    (0, 0, 255),  # Color (red in BGR)
                    5,  # Thickness
                    cv2.LINE_AA,  # Line type
                )
            else:
                # Draw the lane lines
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
            continue
            # if self.i % 300 == 0:
            #     cv2.imshow("Corrected Frame", corrected_frame)
            #     cv2.imshow("Isolated Lane Colors", isolated_lane_colors)
            #     cv2.imshow("Binary Frame (Canny)", binary_frame)
            #     cv2.imshow("ROI Binary Frame", roi_binary_frame)
            #     cv2.waitKey(0)  # Wait for a key press
            #     cv2.destroyAllWindows()
            # self.i +=1
            # Fit polynomials to detected lanes
            left_fit, right_fit = self._fit_polynomial(roi_binary_frame)

            # Draw the lanes on the original frame
            annotated_frame = self._draw_lane_overlay(frame, left_fit, right_fit)

            # Add debugging text (optional)
            if left_fit is not None:
                cv2.putText(
                    annotated_frame,
                    f"Left Curve: {left_fit[0]:.4f}x^2 + {left_fit[1]:.4f}x + {left_fit[2]:.4f}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
            if right_fit is not None:
                cv2.putText(
                    annotated_frame,
                    f"Right Curve: {right_fit[0]:.4f}x^2 + {right_fit[1]:.4f}x + {right_fit[2]:.4f}",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

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
    def _night_correction(frame: RBGFrame, gamma=2) -> RBGFrame:
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
    def _is_night_frame(frame: RBGFrame, threshold=90) -> bool:
        # Convert to grayscale
        gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Calculate the average pixel intensity
        avg_intensity = np.mean(gray_image)

        # Check if the average intensity is below the threshold
        return avg_intensity < threshold

    @staticmethod
    def _isolate_color_lanes(frame: RBGFrame, is_night=False) -> RBGFrame:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # Define ranges for yellow and white
        # lower_yellow = np.array([20, 80, 80])
        # upper_yellow = np.array([40, 150, 150])
        lower_white = np.array([0, 0, 175])
        upper_white = np.array([200, 150 if not is_night else 65, 255])

        # Create masks for yellow and white
        # yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
        white_mask = cv2.inRange(hsv_frame, lower_white, upper_white)

        # Combine masks and apply
        # combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
        color_isolated = cv2.bitwise_and(hsv_frame, hsv_frame, mask=white_mask)

        return cv2.cvtColor(color_isolated, cv2.COLOR_HSV2RGB)

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

        polygon = np.array(
            [
                [
                    (int(width * 0.2), height*0.8),  # Bottom-left
                    (int(width * 0.43), int(height * 0.55)),  # Top-left
                    (int(width * 0.47), int(height * 0.55)),  # Top-right
                    (int(width * 0.8), height*0.8),  # Bottom-right
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
                        (int(width * 0.45), int(height * 0.6)),  # Top-left
                        (int(width * 0.8), height),  # Bottom-right
                    ],
                ],
                dtype=np.int32,
            )

        # Fill polygon with white
        cv2.fillPoly(mask, polygon, [255])

        return mask

    @staticmethod
    def _draw_lanes_on_frame(
        frame: RBGFrame,
        left_lane: Line,
        right_lane: Line,
        thickness: int = 8,
        fill_color: list | None = None,
    ) -> RBGFrame:
        """
        Draw left and right lanes onto the frame.
        Optionally fill the area in between lanes.
        """
        overlay = frame.copy()
        color = (255, 0, 0)

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

        if (left_lane is not None) and (right_lane is not None):
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

        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    def _find_lane_pixels(self, binary_frame: np.ndarray):
        """
        Detect lane pixels using a sliding window approach.
        Returns the pixel positions for left and right lanes.
        """
        histogram = np.sum(binary_frame[binary_frame.shape[0] // 2:, :], axis=0)

        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        margin = 100
        minpix = 50

        window_height = int(binary_frame.shape[0] / nwindows)
        nonzero = binary_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            win_y_low = binary_frame.shape[0] - (window + 1) * window_height
            win_y_high = binary_frame.shape[0] - window * window_height

            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) &
                            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
            good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) &
                            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def _fit_polynomial(self, binary_frame: np.ndarray):
        """
        Fit a second-degree polynomial to the detected lane pixels.
        """
        leftx, lefty, rightx, righty = self._find_lane_pixels(binary_frame)

        left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 0 else None
        right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 0 else None

        return left_fit, right_fit


    def _draw_lane_overlay(self, frame: np.ndarray, left_fit, right_fit):
        """
        Draw lane overlay using polynomial fits.
        """
        ploty = np.linspace(0, frame.shape[0] - 1, frame.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2] if left_fit is not None else None
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2] if right_fit is not None else None

        color_warp = np.zeros_like(frame).astype(np.uint8)

        if left_fitx is not None and right_fitx is not None:
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            cv2.fillPoly(color_warp, [np.int32(pts)], (0, 255, 0))

        return cv2.addWeighted(frame, 1, color_warp, 0.3, 0)
