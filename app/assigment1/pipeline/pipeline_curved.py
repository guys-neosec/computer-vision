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

OUTPUT_FORMAT = "mp4v"
TRESHOLD = 0.7



class line():
    def __init__(self):
        self.first_frame=False
        self.curvature=0

        self.right_fit=[np.array([False])]
        self.left_fit=[np.array([True])]
        self.max_tolerance=0.01

        self.img=None
        self.y_eval=700
        self.mid_x=640
        self.ym_per_pix=3.0/72.0
        self.xm_per_pix=3.7/650.0 #HardCoded

    def update_fit(self,left_fit,right_fit):
        if self.first_frame:
            error_left=((self.left_fit[0]-left_fit[0])**2).mean(axis=None)
            error_right=((self.right_fit[0]-right_fit[0])**2).mean(axis=None)
            if error_left<self.max_tolerance:
                self.left_fit=0.75*self.left_fit+0.25*left_fit
            if error_right<self.max_tolerance:
                self.right_fit=0.75*self.right_fit+0.25*right_fit

        else:
            self.right_fit=right_fit
            self.left_fit=left_fit

        self.update_curvature(self.right_fit)

    def update_curvature(self,fit):

        c1=(2*fit[0]*self.y_eval+fit[1])*self.xm_per_pix/self.ym_per_pix
        c2=2*fit[0]*self.xm_per_pix/(self.ym_per_pix**2)

        curvature=((1+c1*c1)**1.5)/(np.absolute(c2))

        if self.first_frame:
            self.curvature=curvature

        elif np.absolute(curvature-self.curvature)<500:
            self.curvature=0.75*self.curvature + 0.25* curvature

    def vehicle_position(self):
        left_pos=(self.left_fit[0]*(self.y_eval**2))+(self.left_fit[1]*self.y_eval)+self.left_fit[2]
        right_pos=(self.right_fit[0]*(self.y_eval**2))+(self.right_fit[1]*self.y_eval)+self.right_fit[2]

        return ((left_pos+right_pos)/2.0 - self.mid_x)*self.xm_per_pix




class LanesHistory(BaseModel):
    left: list = []
    right: list = []


class Pipeline:
    line=None
    M=None
    Minv=None
    camera_mtx=None
    dist=None

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

        l = line()
        for frame in self._get_frames():
            corrected_frame = frame.copy()
            corrected_frame
            if self._is_night_frame(frame):
                corrected_frame = self._night_correction(frame)

            mask = self._area_of_interest_mask()
            isolated_lane_colors = self.gradient_color_thresh(corrected_frame)
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

                if len(hold_x) == 15:
                    count =0
                    for i in range(len(hold_x)-1):
                        if hold_x[i]>hold_x[i+1]:
                            count+=1
                        else:
                            count-=1
                    if count >0:
                        txt = "left"
                    else:
                        txt = "right"
            else:
                l = line()
                self.move_lane  =  True if (abs(right_slope - self.right_slope_avg) >= TRESHOLD \
                    and abs(left_slope + self.left_slope_avg) >= TRESHOLD) or \
                    abs(right_slope - self.right_slope_avg) > 4 or abs(left_slope + self.left_slope_avg) > 4 else False

                if self.move_lane:
                    txt = None
                    self.move_lane_cooldown = 50
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
                # Draw the lane
                image =np.copy(corrected_frame)
                if not l.first_frame:
                    thresh_image=self.gradient_color_thresh(image)
                    binary_warped,Minv,M=self.perspective_transform(thresh_image)
                annotated_frame=self.curve_pipeline(image,l,M,Minv)
            output_video.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)) # DO NOT TOUCH
            progress_bar() # DO NOT TOUCH
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
    def _process_for_canny(frame: RBGFrame) -> GrayScaleFrame:
        # Define a kernel size for Gaussian smoothing
        kernel_size = 9

        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

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


    def abs_sobel_thresh(self, image,orient,  thresh=(20, 100)):
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

        if orient=='x':x, y=1,0
        else: x, y=0,1

        sobel=cv2.Sobel(gray,cv2.CV_64F,x,y)
        sobel=np.absolute(sobel)
        scaled_sobel=np.uint8(255*sobel/np.max(sobel))

        sx_binary=np.zeros_like(scaled_sobel)
        sx_binary[(scaled_sobel>=thresh[0]) & (scaled_sobel<=thresh[1])]=1
        binary_output=np.copy(sx_binary)
        return binary_output


    def mag_thresh(self, img, mag_thresh=(20,150)):

        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0)
        sobely=cv2.Sobel(gray,cv2.CV_64F,0,1)
        sobel=np.sqrt(np.square(sobelx)+np.square(sobely))
        scaled_sobel=np.uint8(255*sobel/np.max(sobel))

    #     t=sum((i > 150) &(i<200)  for i in scaled_sobel)
        binary_sobel=np.zeros_like(scaled_sobel)
        binary_sobel[(scaled_sobel>=mag_thresh[0]) & (scaled_sobel<=mag_thresh[1])]=1
        return binary_sobel

    def dir_threshold(self, img,  thresh=(0.7,1.3)):
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        sobelx=np.absolute(cv2.Sobel(gray,cv2.CV_64F,1,0))
        sobely=np.absolute(cv2.Sobel(gray,cv2.CV_64F,0,1))

        dir_=np.arctan2(sobely,sobelx)

        sx_binary = np.zeros_like(gray)
        sx_binary[(dir_>=thresh[0]) &(dir_<=thresh[1])]=1
        binary_output=sx_binary
        return binary_output


    def color_space(self, image,thresh=(170,255)):
        hls=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
        gray_image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        l_channel=hls[:,:,1]
        s_channel=hls[:,:,2]
        s_binary=np.zeros_like(s_channel)

        _, gray_binary = cv2.threshold(gray_image.astype('uint8'), 150, 255, cv2.THRESH_BINARY)
        s_binary[(s_channel>=thresh[0]) & (s_channel<=thresh[1])&(l_channel>=80)]=1
        color_output=np.copy(s_binary)
        return color_output

    def segregate_white_line(self, image,thresh=(200,255)):
        hls=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
        l_channel=hls[:,:,1]
        l_binary=np.zeros_like(l_channel)
        l_binary[((l_channel>=200)&(l_channel<=255))]=1
        return l_binary

    def gradient_color_thresh(self, image):
        ksize=3
        gradx = self.abs_sobel_thresh(image, orient='x', thresh=(20, 200))
        grady = self.abs_sobel_thresh(image, orient='y', thresh=(20, 200))

        mag_binary = self.mag_thresh(image, mag_thresh=(20, 200))

        dir_binary = self.dir_threshold(image, thresh=(0.7, 1.3))

        color_binary=self.color_space(image,thresh=(100,255))

        combined = np.zeros_like(dir_binary)
        combined[(color_binary==1)|((gradx == 1)& (grady == 1)) |(mag_binary==1) &(dir_binary==1)] = 1

        kernel = np.ones((3,3),np.uint8)
        morph_image=combined[600:,:950]
        morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel)
        combined[600:,:950]=morph_image

        white_line=self.segregate_white_line(image,thresh=(200,255))
        combined=(combined)|(white_line)

        return combined

    def perspective_transform(self, image):
        src = np.float32([[350, 880], [1480, 880], [1050, 600], [800, 600]])
        dst = np.float32([
                [400, 1080],   # Bottom-left
                [1520, 1080],  # Bottom-right
                [1520, 0],     # Top-right
                [400, 0]       # Top-left
            ])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        img_size=(image.shape[1],image.shape[0])
        warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
        return warped,Minv,M

    def curve_pipeline(self,img, line: Line, M , Minv):
        img_size=(img.shape[1],img.shape[0])
        undist_img=np.copy(img)
        thresh_image=self.gradient_color_thresh(undist_img)
        binary_warped = cv2.warpPerspective(thresh_image, M, img_size, flags=cv2.INTER_LINEAR)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        nwindows = 9
        window_height = int(binary_warped.shape[0]/nwindows)
        margin = 100
        minpix=50

        if not line.first_frame:
            histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
            # Create an output image to draw on and  visualize the result
            midpoint = int(histogram.shape[0]/2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint

            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = binary_warped.nonzero()
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
                win_y_low = binary_warped.shape[0] - (window+1)*window_height
                win_y_high = binary_warped.shape[0] - window*window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                # Draw the windows on the visualization image
                cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
                (0,255,0), 2)
                cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
                (0,255,0), 2)
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
            prev_left_fit = left_fit
            prev_right_fit = right_fit

            line.update_fit(left_fit,right_fit)
            line.first_frame=True

        else:
            nonzero = binary_warped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            left_fit=line.left_fit
            right_fit=line.right_fit
            left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
            left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
            left_fit[1]*nonzeroy + left_fit[2] + margin)))

            right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
            right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
            right_fit[1]*nonzeroy + right_fit[2] + margin)))

            # Again, extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            line.update_fit(left_fit,right_fit)
            left_fit=line.left_fit
            right_fit=line.right_fit




        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        cv2.fillPoly(color_warp, [pts.astype(np.int32)], (0,255, 0))
        newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
        result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)

        return result




