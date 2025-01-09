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
            right_slope = self._average_slope(right_slope,self.right_slope_his)
            left_slope = self._average_slope(left_slope,self.left_slope_his)
            avg_left_line = self._average_lines(left_lines)
            avg_right_line = self._average_lines(right_lines)
            frame_height = frame.shape[0]
            roi_top = 435
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
                x,y = self.calculate_intersection(smoothed_left_line,smoothed_right_line)
                x = self._average_slope(x,self.inter_x)
                hold_x.append(x)

                if len(hold_x) == 35:
                    count =0
                    print(hold_x)
                    for i in range(len(hold_x)-1):
                        if hold_x[i]>hold_x[i+1]:
                            count+=1
                        else:
                            count-=1
                    if max(hold_x)-min(hold_x) < 40:
                        txt = None
                    if count >0:
                        txt = "left"
                    else:
                        txt = "right"
            else:
                self.move_lane  =  True if (abs(right_slope - self.right_slope_avg) >= TRESHOLD \
                    and abs(left_slope + self.left_slope_avg) >= TRESHOLD) or \
                    abs(right_slope - self.right_slope_avg) > 4 or abs(left_slope + self.left_slope_avg) > 4 else False

                if self.move_lane:
                    print((abs(right_slope - self.right_slope_avg) >= TRESHOLD \
                    and abs(left_slope + self.left_slope_avg) >= TRESHOLD))
                    print((abs(right_slope - self.right_slope_avg), abs(left_slope + self.left_slope_avg) ))
                    print(abs(right_slope - self.right_slope_avg) > 4)
                    print(abs(right_slope - self.right_slope_avg) )
                    print( abs(left_slope + self.left_slope_avg) > 4)
                    print( abs(left_slope + self.left_slope_avg) )
                    txt = None
                    self.move_lane_cooldown = 70
                    self.inter_x = []
                    self.inter_y = []
                    self.count_frames_since_change = 0
                    hold_x = []

            if self.move_lane and txt:
                # Annotate the frame with "Moving Lane" in big red text
                annotated_frame = frame.copy()  # Copy the frame to annotate
                cv2.putText(
                    annotated_frame,
                    txt, # Text to display
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
                    (int(width * 0.35), int(height * 0.55)),  # Top-left
                    (int(width * 0.55), int(height * 0.55)),  # Top-right
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
    def _average_slope(
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

    def calculate_intersection(self, left_line: tuple, right_line: tuple) -> tuple[float, float]:
        # Unpack the lines
        x1_left, y1_left, x2_left, y2_left = left_line
        x1_right, y1_right, x2_right, y2_right = right_line

        # Calculate the slopes (m1 and m2)
        m1 = (y2_left - y1_left) / (x2_left - x1_left) if x2_left != x1_left else None
        m2 = (y2_right - y1_right) / (x2_right - x1_right) if x2_right != x1_right else None

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

    def perspective_warp(self, img,
                dst_size=(1280,720),
                src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
        img_size = np.float32([(img.shape[1],img.shape[0])])
        src = src* img_size
        dst = dst * np.float32(dst_size)
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, dst_size)
        return warped

    def inv_perspective_warp(self, img,
                     dst_size=(1280,720),
                     src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                     dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])):
        img_size = np.float32([(img.shape[1],img.shape[0])])
        src = src* img_size
        # For destination points, I'm arbitrarily choosing some points to be
        # a nice fit for displaying our warped result
        # again, not exact, but close enough for our purposes
        dst = dst * np.float32(dst_size)
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, dst_size)
        return warped

    def get_hist(self, img):
        hist = np.sum(img[img.shape[0]//2:,:], axis=0)
        return hist