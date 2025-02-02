import cv2
import numpy as np
from loguru import logger

feature_extractor = cv2.SIFT_create()

# === template image keypoint and descriptors
template_image = cv2.imread(
    "/Users/gstrauss/Reichman_University/computer-vision/app/assigment2/images/template.png",
)
wrapped_template = cv2.imread(
    "/Users/gstrauss/Reichman_University/computer-vision/app/assigment2/images/warped_template.png",
)
wrapped_template = cv2.resize(
    wrapped_template,
    (template_image.shape[1], template_image.shape[0]),
)
template_grey = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
video_path = "/Users/gstrauss/Downloads/IMG_4642.MOV"
# ===== video input, output and metadata
input_video = cv2.VideoCapture(video_path)
width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = input_video.get(cv2.CAP_PROP_FPS)
output_video = cv2.VideoWriter(
    "/Users/gstrauss/Reichman_University/computer-vision/app/assigment2/movie/output_wrap.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    120,
    (width, height),
)
# ========== run on all frames
index = 0
while True:
    ok, frame = input_video.read()
    if not ok:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_template, desc_template = feature_extractor.detectAndCompute(template_grey, None)
    kp_frame, desc_frame = feature_extractor.detectAndCompute(frame_gray, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_template, desc_frame, k=2)

    good_features = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_features.append([m, n])
    good_match_arr = np.asarray(good_features)[:, 0]

    im_matches = cv2.drawMatchesKnn(
        template_image,
        kp_template,
        frame,
        kp_frame,
        good_features,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    # ======== find homography
    good_kp_template = np.array([kp_template[m.queryIdx].pt for m in good_match_arr])
    good_kp_frame = np.array([kp_frame[m.trainIdx].pt for m in good_match_arr])
    H, masked = cv2.findHomography(good_kp_template, good_kp_frame, cv2.RANSAC, 5.0)

    # ++++++++ do warping of another image on template image
    # we saw this in SIFT notebook
    height, width, channels = frame.shape
    dst = cv2.warpPerspective(wrapped_template, H, (width, height))
    mask = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    mask_inv = cv2.bitwise_not(binary_mask)

    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

    template_fg = cv2.bitwise_and(dst, dst, mask=binary_mask)

    combined_frame = cv2.add(frame_bg, template_fg)

    output_video.write(combined_frame)

    output_video.write(combined_frame)
    logger.debug("Here")

output_video.release()
input_video.release()
logger.success("Done")
