import cv2
import numpy as np

from app.assigment1.custom_types import GrayScaleFrame, RBGFrame


def detect_proximity(frame: RBGFrame):
    # frame = cv2.imread("/Users/gstrauss/Reichman_University/computer-vision/app/assigment1/pipeline/img.png")
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    annotated_frame = frame.copy()
    mask = area_of_interest_mask(frame)
    masked_frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)
    # Define HSV range for white color (tolerant range to capture white lines)
    threshold_value = 100  # Pixels brighter than this value are considered "white"
    _, bright_mask = cv2.threshold(
        masked_frame_gray,
        threshold_value,
        255,
        cv2.THRESH_BINARY,
    )
    bright_colors_frame = cv2.bitwise_and(frame, frame, mask=bright_mask)
    frame_hsv = cv2.cvtColor(bright_colors_frame, cv2.COLOR_RGB2HSV)

    # Define HSV range for yellow color
    lower_yellow = np.array([20, 100, 100])  # Lower bound for yellow
    upper_yellow = np.array([30, 255, 255])  # Upper bound for yellow

    # Create a mask to isolate yellow regions
    yellow_mask = cv2.inRange(frame_hsv, lower_yellow, upper_yellow)
    lower_green = np.array([30, 50, 50])  # Lower bound for green
    upper_green = np.array([90, 255, 255])  # Upper bound for green

    # Create the mask for green regions
    green_mask = cv2.inRange(frame_hsv, lower_green, upper_green)
    # Invert the mask to remove yellow regions
    non_yellow_mask = cv2.bitwise_not(yellow_mask)
    dilation_kernel = np.ones((5, 5), np.uint8)
    eroded_non_yellow_mask = cv2.erode(non_yellow_mask, dilation_kernel, iterations=2)

    # Apply the non-yellow mask to the frame
    filtered_frame = cv2.bitwise_and(
        bright_colors_frame,
        bright_colors_frame,
        mask=eroded_non_yellow_mask,
    )

    gray_filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
    gray_filtered_frame = cv2.morphologyEx(
        gray_filtered_frame,
        cv2.MORPH_OPEN,
        dilation_kernel,
    )
    height, width = gray_filtered_frame.shape
    src_points = np.float32(
        [
            [0, height],  # Bottom-left
            [width, height],  # Bottom-right
            [int(0.35 * width), int(0.65 * height)],  # Top-left (near vanishing point)
            [int(0.65 * width), int(0.65 * height)],  # Top-right (near vanishing point)
        ],
    )

    # Desired rectangle for the top-down (2D) view
    dst_points = np.float32(
        [
            [0, height],  # Bottom-left
            [width, height],  # Bottom-right
            [0, 0],  # Top-left
            [width, 0],  # Top-right
        ],
    )

    # Compute the perspective transformation matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    M_inv = cv2.getPerspectiveTransform(dst_points, src_points)

    # Apply perspective warp
    warped_image = cv2.warpPerspective(
        gray_filtered_frame,
        perspective_matrix,
        (width, height),
    )

    edges = cv2.Canny(warped_image, 50, 180)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=20,
        maxLineGap=5,
    )

    # Hough Line Transform to detect lines

    results = annotated_frame.copy()
    possible_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(warped_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            points_array = np.array([(x1, y1, x2, y2)], dtype="float32").reshape(
                -1,
                1,
                2,
            )
            transformed_points = cv2.perspectiveTransform(points_array, M_inv)
            start_point, end_point = transformed_points.reshape(-1, 2)
            x1, y1 = int(start_point[0]), int(start_point[1])
            x2, y2 = int(end_point[0]), int(end_point[1])
            line_distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if line_distance > 30:
                continue
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            slope = (y2 - y1) / (x2 - x1 + 1e-10)
            # Only draw horizontal lines (close to 0° or 180°)
            if -0.5 < slope < 1:
                possible_lines.append((x1, y1, x2, y2))
    for x1, y1, x2, y2 in possible_lines:
        if abs(y1 - y2) > 10:
            continue
        if abs(x1 - x2) < 10:
            continue
        cv2.line(results, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # cntrs = cv2.findContours(gray_filtered_frame, cv2.RETR_EXTERNAL,
    #                          cv2.CHAIN_APPROX_SIMPLE)
    # cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
    # # filter on area
    # good_contours = []
    # for c in cntrs:
    #     x, y, w, h = cv2.boundingRect(c)
    #     area = cv2.contourArea(c)
    #     aspect_ratio = w / h
    #     if 20 < area < 1500 and 1.2 < aspect_ratio < 4:
    #         cv2.drawContours(results, [c], -1, (0, 255, 0), 1)
    #         good_contours.append(c)
    # horizontal_distance_threshold = 50
    #
    # # Extract the center (x, y) points of each rectangle
    # rectangle_centers = [(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] // 2,
    #                       cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] // 2) for c in
    #                      good_contours]
    #
    # # Sort rectangles by their Y-coordinate (vertical position)
    # rectangle_centers.sort(key=lambda p: p[1])
    #
    # # Group rectangles by rows (lines)
    # lines = []  # To store groups of aligned rectangles
    # current_line = [rectangle_centers[0]]
    #
    # for i in range(1, len(rectangle_centers)):
    #     prev_x, prev_y = current_line[-1]
    #     curr_x, curr_y = rectangle_centers[i]
    #
    #     # If the current rectangle is aligned (same row) within a certain distance
    #     if abs(curr_y - prev_y) < 20 and abs(
    #             curr_x - prev_x) < horizontal_distance_threshold:
    #         current_line.append((curr_x, curr_y))
    #     else:
    #         if len(current_line) >= 5:  # If a line of 3 or more rectangles is found
    #             lines.append(current_line)
    #         current_line = [(curr_x, curr_y)]
    #
    # # Add the last line if valid
    # if len(current_line) >= 5:
    #     lines.append(current_line)
    #
    # # Annotate the detected line of rectangles
    # for line in lines:
    #     for (x, y) in line:
    #         cv2.circle(results, (x, y), 5, (255, 0, 0), -1)
    #
    # if len(lines) > 0:
    #     cv2.putText(results, "Crosswalk Line Detected!", (50, 100),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return results


def area_of_interest_mask(frame: RBGFrame) -> GrayScaleFrame:
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    polygon = np.array(
        [
            [
                (int(width * 0.2), height),  # Bottom-left
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
