import cv2
import numpy as np

from app.assigment1.custom_types import GrayScaleFrame, RBGFrame

HISTORY = []
FRAME_COUNT = 0


def crosswalk(frame: RBGFrame):
    global HISTORY, FRAME_COUNT
    # frame = cv2.imread("/Users/gstrauss/Reichman_University/computer-vision/app/assigment1/pipeline/img.png")
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mask = area_of_interest_mask(frame)
    masked_frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)

    for p in [
        "/Users/gstrauss/Reichman_University/computer-vision/app/assigment1/pipeline/car.png",
        "/Users/gstrauss/Reichman_University/computer-vision/app/assigment1/pipeline/car2.png",
        "/Users/gstrauss/Reichman_University/computer-vision/app/assigment1/pipeline/rear.png",
    ]:
        template = cv2.imread(p)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(frame_gray, gray_template, cv2.TM_CCOEFF_NORMED)
        _, _, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc
        h, w = gray_template.shape
        cv2.rectangle(
            masked_frame_gray,
            top_left,
            (top_left[0] + w, top_left[1] + h),
            (0, 0, 0),
            -1,
        )
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
        threshold=45,
        minLineLength=15,
        maxLineGap=5,
    )

    # Hough Line Transform to detect lines

    candidate_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
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
            slope = (y2 - y1) / (x2 - x1 + 1e-10)
            if abs(y1 - y2) > 10:
                continue
            if abs(x1 - x2) < 10:
                continue
            if -0.5 < slope < 1:
                candidate_lines.append((x1, y1, x2, y2))

    if len(candidate_lines) < 5:
        return None

    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    y_sum = 0

    for _, y1, _, y2 in candidate_lines:
        y_sum += y1
        y_sum += y2

    y_avg = y_sum / len(candidate_lines * 2)
    crosswalk_line = []
    for x1, y1, x2, y2 in candidate_lines:
        if abs(y_avg - y1) > 0.06 * height or abs(y_avg - y2) > 0.1 * height:
            continue
        crosswalk_line.append((x1, y1, x2, y2))

    for x1, y1, x2, y2 in crosswalk_line:
        min_x = min(min_x, x1, x2)
        min_y = min(min_y, y1, y2)
        max_x = max(max_x, x1, x2)
        max_y = max(max_y, y1, y2)
        # cv2.line(results, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if 1000 < abs(min_x - max_y) * abs(min_y - max_y) < 7000:
        FRAME_COUNT = 0
        if len(HISTORY) > 6:
            HISTORY.pop(0)
        total = 0
        history_min_x, history_min_y = 0, 0
        history_max_x, history_max_y = 0, 0
        for index, history in enumerate(HISTORY, start=1):
            total += index
            history_min_x += index * history[0]
            history_min_y += index * history[1]
            history_max_x += index * history[2]
            history_max_y += index * history[3]
        if len(HISTORY) == 0:
            history_min_x, history_min_y = min_x, min_y
            history_max_x, history_max_y = max_x, max_y
        else:
            history_min_x /= total
            history_min_y /= total
            history_max_x /= total
            history_max_y /= total
        HISTORY.append((min_x, min_y, max_x, max_y))

        return (
            (min_x + int(history_min_x)) // 2,
            (min_y + int(history_min_y)) // 2,
            (max_x + int(history_max_x)) // 2,
            (int(history_max_y) + max_y) // 2,
        )
    return None


def area_of_interest_mask(frame: RBGFrame) -> GrayScaleFrame:
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    polygon = np.array(
        [
            [
                (int(width * 0.2), height),  # Bottom-left
                (int(width * 0.5), int(height * 0.6)),  # Top-left
                (int(width * 0.55), int(height * 0.6)),  # Top-right
                (int(width * 0.7), height),  # Bottom-right
            ],
        ],
        dtype=np.int32,
    )

    # Fill polygon with white
    cv2.fillPoly(mask, polygon, [255])

    return mask
#
# frame = cv2.imread("/Users/gstrauss/Reichman_University/computer-vision/app/assigment1/pipeline/img_4.png")
# crosswalk_rectangle = crosswalk(frame)
# if crosswalk_rectangle is not None:
#     x1, y1, x2, y2 = crosswalk_rectangle
#     overlay = frame.copy()
#     color = (255, 0, 255)
#     cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
#     annotated_frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
# print(annotated_frame)