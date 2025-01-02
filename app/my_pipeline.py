import cv2
import numpy as np
from pathlib import Path
from moviepy.editor import VideoFileClip, ImageSequenceClip

TRESHOLD = 0.3
LEFT_VALID_LINE = None
RIGHT_VALID_LINE = None


def isolate_lane_colors(image: np.ndarray) -> np.ndarray:
    """
    Isolate lane colors (yellow and white) in an image using HSV colorspace.
    """
    # Convert image from BGR to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define ranges for yellow and white
    lower_yellow = np.array([20, 80, 80])
    upper_yellow = np.array([40, 150, 150])
    lower_white = np.array([0, 0, 120])
    upper_white = np.array([255, 30, 150])

    # Create masks for yellow and white
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)

    # Combine masks and apply
    combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
    color_isolated = cv2.bitwise_and(image, image, mask=combined_mask)

    # Debug outputs
    # cv2.imshow("Yellow Mask", yellow_mask)
    # cv2.imshow("White Mask", white_mask)
    # cv2.imshow("Combined Mask", combined_mask)

    cv2.waitKey(0)

    return color_isolated, combined_mask


def preprocess_image_for_edges(color_isolated: np.ndarray) -> np.ndarray:
    """
    Convert the image to grayscale and apply Gaussian blur to prepare for edge detection.

    Args:
        color_isolated (np.ndarray): The color-isolated image (output from HSV masking).

    Returns:
        np.ndarray: The blurred grayscale image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(color_isolated, cv2.COLOR_BGR2GRAY)

    # Define a kernel size for Gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    return blur_gray

def define_roi_mask(image: np.ndarray) -> np.ndarray:
    """
    Define a region-of-interest mask for the image but do not apply it.

    Args:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: A binary mask where the ROI is white, and other areas are black.
    """
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    # Define a polygon that roughly corresponds to the road area
    polygon = np.array([[
        (int(width * 0.1), height),  # Bottom-left
        (int(width * 0.45), int(height * 0.6)),  # Top-left
        (int(width * 0.55), int(height * 0.6)),  # Top-right
        (int(width * 0.9), height)  # Bottom-right
    ]], dtype=np.int32)

    # Fill polygon with white
    cv2.fillPoly(mask, polygon, 255)

    return mask

def apply_roi_on_edges(edges: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """
    Apply the ROI mask on the edges image.

    Args:
        edges (np.ndarray): The edges image from Canny edge detection.
        roi_mask (np.ndarray): The binary mask defining the ROI.

    Returns:
        np.ndarray: The edges image with the ROI applied.
    """
    return cv2.bitwise_and(edges, roi_mask)


def hough_transform(masked_edges: np.ndarray) -> np.ndarray:
    """
    Apply Hough Transform to find lines in the edges image.

    Args:
        masked_edges (np.ndarray): The edges image with ROI applied.

    Returns:
        list: Detected lines represented as (x1, y1, x2, y2).
    """
    # Define Hough Transform parameters
    rho = 2  # Distance resolution in pixels
    theta = np.pi / 180  # Angular resolution in radians
    threshold = 30  # Minimum number of votes
    min_line_length = 20  # Minimum number of pixels making up a line
    max_line_gap = 15  # Maximum gap in pixels between lines

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(
        masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap
    )
    return lines

def hough_transform_standard(masked_edges: np.ndarray):
    """
    Apply the standard Hough Transform to find lines in the edges image.
    Returns (rho, theta).
    """
    rho = 1          # Distance resolution in pixels
    theta = np.pi/180  # Angular resolution in radians
    threshold = 50     # Minimum number of votes

    # The return shape from cv2.HoughLines is (N,1,2), each line is [[rho, theta]].
    lines = cv2.HoughLines(masked_edges, rho, theta, threshold)
    return lines

def separate_lines(lines, img_shape):
    """
    Separate the Hough lines into left and right lines based on slope and position.
    """
    left_lines = []
    right_lines = []

    # We assume the middle of the image is the dividing line
    # (everything with negative slope or left side is 'left', positive slope or right side is 'right').
    # Tweak slope thresholds as needed.
    img_center = img_shape[1] / 2
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


def average_lines(lines):
    """
    Given a list of lines (x1, y1, x2, y2), average them
    to produce a single representative line for that group.
    Returns (x1, y1, x2, y2) or None if no lines.
    """
    if len(lines) == 0:
        return None

    # Convert to numpy for easier math
    lines_arr = np.array(lines, dtype=np.float32)
    x1_mean = np.mean(lines_arr[:, 0])
    y1_mean = np.mean(lines_arr[:, 1])
    x2_mean = np.mean(lines_arr[:, 2])
    y2_mean = np.mean(lines_arr[:, 3])

    return (int(x1_mean), int(y1_mean), int(x2_mean), int(y2_mean))


def smooth_line(current_line, history_dict, side='left', max_history=10):
    """
    Smooth the line over time using a small history buffer.
    current_line: (x1, y1, x2, y2)
    history_dict: dictionary storing the lines for left/right across frames
    side: 'left' or 'right'
    max_history: how many past lines to store
    """
    if current_line is not None:
        # Store current line
        history_dict[side].append(current_line)

    # Keep history up to 'max_history' length
    if len(history_dict[side]) > max_history:
        history_dict[side].pop(0)

    # If we have no lines in history, return None
    if len(history_dict[side]) == 0:
        return None

    # Average over all lines in history
    arr = np.array(history_dict[side], dtype=np.float32)
    x1_avg = int(np.mean(arr[:, 0]))
    y1_avg = int(np.mean(arr[:, 1]))
    x2_avg = int(np.mean(arr[:, 2]))
    y2_avg = int(np.mean(arr[:, 3]))

    return (x1_avg, y1_avg, x2_avg, y2_avg)


def draw_lanes_on_frame(
    frame,
    left_lane,
    right_lane,
    color=(255, 0, 0),
    thickness=8,
    fill_color=None,
    draw_lane_area=False
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
            thickness
        )
    if right_lane is not None:
        cv2.line(
            overlay,
            (right_lane[0], right_lane[1]),
            (right_lane[2], right_lane[3]),
            color,
            thickness
        )

    # Optionally fill the area between the two lanes
    if draw_lane_area and (left_lane is not None) and (right_lane is not None):
        pts = np.array([
            [left_lane[0], left_lane[1]],
            [left_lane[2], left_lane[3]],
            [right_lane[2], right_lane[3]],
            [right_lane[0], right_lane[1]]
        ], dtype=np.int32)

        cv2.fillPoly(overlay, [pts], fill_color)

    # Combine overlay with original using some transparency
    alpha = 0.4
    frame_with_lanes = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    return frame_with_lanes

def extend_line_to_full_height(line, top_limit, bottom_limit):
    """
    Extend a line (x1, y1, x2, y2) so that it spans from top_limit to bottom_limit.
    """
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

def calculate_average_slope(lines):
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



def process_video(input_filename: str, output_filename: str):
    """
    Process the video to detect and draw smoothed lanes,
    and optionally fill the area between them.

    Args:
        input_filename (str): Path to the input video file.
        output_filename (str): Path to save the processed video.
    """
    # Example history for line smoothing:
    # Store (x1, y1, x2, y2) for left and right lanes separately
    history = {"left": [], "right": []}

    # Read video using MoviePy
    base_clip = VideoFileClip(input_filename)
    def process_frame(frame):
        """
        This sub-function processes a single frame and returns the annotated frame.
        """
        # =============== STEP 1: Preprocessing ===============
        roi_mask = define_roi_mask(frame)
        color_isolated, combined_mask = isolate_lane_colors(frame)
        blur_gray = preprocess_image_for_edges(color_isolated)
        edges = cv2.Canny(blur_gray, 50, 150)
        roi_edges = apply_roi_on_edges(edges, roi_mask)

        # =============== STEP 2: Hough Lines ===============
        lines = hough_transform(roi_edges)

        # =============== STEP 3: Filter & Separate Lines ===============
        # lines might be None if no Hough lines found
        if lines is None:
            return frame

        left_lines, right_lines = separate_lines(lines, frame.shape)

        # =============== STEP 4: Average/Extrapolate Each Side ===============


        left_slope = calculate_average_slope(left_lines)
        if not left_slope:
            left_slope = -1.1
        right_slope = calculate_average_slope(right_lines)
        if not right_slope:
            right_slope = 0.6
        global RIGHT_VALID_LINE,LEFT_VALID_LINE
        avg_left_line = average_lines(left_lines)
        avg_right_line = average_lines(right_lines)
        # print(avg_left_line)
        # print(avg_right_line)
        if abs(right_slope -0.6) >= TRESHOLD and abs(left_slope+0.9) < TRESHOLD:
            avg_right_line = RIGHT_VALID_LINE
        else:
            RIGHT_VALID_LINE = avg_right_line
        if abs(right_slope -0.6) < TRESHOLD and abs(left_slope+0.9) >= TRESHOLD:
            avg_left_line = LEFT_VALID_LINE
        else:
            LEFT_VALID_LINE = avg_left_line
        # =============== STEP 5: Smoothing Over Time ===============
        # Store lines in history, average them if enough frames are in the buffer


        frame_height = frame.shape[0]
        roi_top = 435  # Example top limit for ROI
        extended_left_line = extend_line_to_full_height(avg_left_line, roi_top, frame_height)
        extended_right_line = extend_line_to_full_height(avg_right_line, roi_top, frame_height)

        smoothed_left_line = smooth_line(extended_left_line, history, 'left')
        smoothed_right_line = smooth_line(extended_right_line, history, 'right')

        # =============== STEP 6: Draw Final Lanes ===============
        annotated_frame = draw_lanes_on_frame(
            frame,
            smoothed_left_line,
            smoothed_right_line,
            fill_color=(0, 255, 0),  # Optional fill color for area between lanes
            draw_lane_area=True
        )
        debug_text = f"Left Slope: {left_slope:.2f}" if left_slope is not None else "Left Slope: None"
        cv2.putText(annotated_frame, debug_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        debug_text = f"Right Slope: {right_slope:.2f}" if right_slope is not None else "Right Slope: None"
        cv2.putText(annotated_frame, debug_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        debug_text = f"move lane" if abs(right_slope -0.6) >= TRESHOLD and abs(left_slope+0.9) >= TRESHOLD else "same lane"
        cv2.putText(annotated_frame, debug_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        debug_text = f"right error" if abs(right_slope -0.6) >= TRESHOLD and abs(left_slope+0.9) <TRESHOLD else "right okay"
        cv2.putText(annotated_frame, debug_text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        debug_text = f"left error" if abs(right_slope -0.6) < TRESHOLD and abs(left_slope+0.9) >= TRESHOLD else "left okay"
        cv2.putText(annotated_frame, debug_text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return annotated_frame

    # Apply the process_frame function to every video frame
    output_clip = base_clip.fl_image(process_frame)
    # Write out the processed video
    output_clip.write_videofile(output_filename, audio=False)






def main():
    # 1. Load your test image (BGR format by default with cv2.imread)
    image_path = Path("/home/aweinsto/projects/computer-vision/image.png")
    input_filename = "/home/aweinsto/projects/computer-vision/short_input.mp4"
    output_filename = "/home/aweinsto/projects/computer-vision/output.mp4"
    process_video(input_filename, output_filename)


if __name__ == "__main__":
    main()
