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
        self.left_a = []
        self.left_b = []
        self.left_c= []
        self.right_a = []
        self.right_b = []
        self.right_c= []

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

            img = corrected_frame
            corrected_frame
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

            src = np.float32([
                [600, 700],   # Bottom-left
                [1100, 700],  # Bottom-right
                [1100, 600],  # Top-right
                [600, 600]    # Top-left
            ])

            # Destination points: Map to a rectangle
            dst = np.float32([
                [0, 720],         # Bottom-left
                [1280, 720],      # Bottom-right
                [1280, 0],        # Top-right
                [0, 0]            # Top-left
            ])
            img_ = self.curve_pipeline(img,src,dst)
            height, width = img.shape[:2]
            img_ = self.perspective_warp(img_)

            out_img, curves, lanes, ploty = self.sliding_window(img_, draw_windows=False)

            curverad = self.get_curve(img, curves[0], curves[1])
            lane_curve = np.mean([curverad[0], curverad[1]])
            img = self.draw_lanes(img, curves[0], curves[1])
            cv2.imshow("curve iamge", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


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
                     src=np.float32([(0.5,0.6),(0.58,0.65),(0.1,1),(1,1)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
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

    # def perspective_warp(self, img, src, dst, dst_size=(1280, 720)):
    #     """
    #     Applies a perspective warp to the given image.

    #     Args:
    #         img (numpy.ndarray): Input binary image.
    #         src (numpy.ndarray): Source points for the transformation.
    #         dst (numpy.ndarray): Destination points for the transformation.
    #         dst_size (tuple): Desired output image size (width, height).

    #     Returns:
    #         numpy.ndarray: Warped image.
    #     """
    #     # Compute the transformation matrix
    #     M = cv2.getPerspectiveTransform(src, dst)

    #     # Warp the image
    #     warped = cv2.warpPerspective(img, M, dst_size)
    #     return warped

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


    def sliding_window(self, img, nwindows=9, margin=150, minpix = 1, draw_windows=True):
        left_fit_= np.empty(3)
        right_fit_ = np.empty(3)
        out_img = np.dstack((img, img, img))*255

        histogram = self.get_hist(img)
        # find peaks of left and right halves
        midpoint = int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint


        # Set height of windows
        window_height = int(img.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base


        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            if draw_windows == True:
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
                (100,255,255), 3)
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
                (100,255,255), 3)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))


    #        if len(good_right_inds) > minpix:
    #            rightx_current = np.int(np.mean([leftx_current +900, np.mean(nonzerox[good_right_inds])]))
    #        elif len(good_left_inds) > minpix:
    #            rightx_current = np.int(np.mean([np.mean(nonzerox[good_left_inds]) +900, rightx_current]))
    #        if len(good_left_inds) > minpix:
    #            leftx_current = np.int(np.mean([rightx_current -900, np.mean(nonzerox[good_left_inds])]))
    #        elif len(good_right_inds) > minpix:
    #            leftx_current = np.int(np.mean([np.mean(nonzerox[good_right_inds]) -900, leftx_current]))


        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        self.left_a.append(left_fit[0])
        self.left_b.append(left_fit[1])
        self.left_c.append(left_fit[2])

        self.right_a.append(right_fit[0])
        self.right_b.append(right_fit[1])
        self.right_c.append(right_fit[2])

        left_fit_[0] = np.mean(self.left_a[-10:])
        left_fit_[1] = np.mean(self.left_b[-10:])
        left_fit_[2] = np.mean(self.left_c[-10:])

        right_fit_[0] = np.mean(self.right_a[-10:])
        right_fit_[1] = np.mean(self.right_b[-10:])
        right_fit_[2] = np.mean(self.right_c[-10:])

        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
        right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]

        return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty


    def get_curve(self, img, leftx, rightx):
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        y_eval = np.max(ploty)
        ym_per_pix = 30.5/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/720 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        car_pos = img.shape[1]/2
        l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
        r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) /2
        center = (car_pos - lane_center_position) * xm_per_pix / 10
        # Now our radius of curvature is in meters
        return (left_curverad, right_curverad, center)

    def draw_lanes(self, img, left_fit, right_fit):
        # Generate y-coordinates
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])

        # Create an empty color image
        color_img = np.zeros_like(img)

        # Generate points for left and right lanes
        left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
        points = np.hstack((left, right))

        # Draw the lane polygon
        cv2.fillPoly(color_img, np.int_(points), (0, 200, 255))

        # Apply inverse perspective warp
        inv_perspective = self.inv_perspective_warp(color_img)

        # Resize the inverse perspective image to match the original image
        inv_perspective = cv2.resize(inv_perspective, (img.shape[1], img.shape[0]))

        print(f"Original image shape: {img.shape}")
        print(f"Inverse perspective shape: {inv_perspective.shape}")

    # Ensure both images have the same number of channels
        if len(img.shape) == 2:  # If original image is grayscale, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            print(f"Converted original image shape: {img.shape}")

        if len(inv_perspective.shape) == 2:  # If inverse perspective is grayscale, convert to RGB
            inv_perspective = cv2.cvtColor(inv_perspective, cv2.COLOR_GRAY2BGR)
            print(f"Converted inverse perspective shape: {inv_perspective.shape}")

        # Blend the original image and the lane polygon
        blended_img = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)

        return blended_img

    def curve_pipeline(self,img,src, dst):

        s_thresh = (100, 255)  # Loosen thresholds for testing
        sx_thresh = (15, 255)

        # Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(float)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)  # Gradient
        abs_sobelx = np.absolute(sobelx)
        if np.max(abs_sobelx) != 0:
            scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        else:
            scaled_sobel = np.zeros_like(abs_sobelx)

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold S channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # Combine binary images
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        return combined_binary