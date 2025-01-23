# ======= imports
from glob import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ======= constants
square_size = 2.65
img_mask = "/home/aweinsto/projects/computer-vision/app/assigment2/calibration/*.jpeg"
pattern_size = (9, 6)

figsize = (20, 20)
# === template image keypoint and descriptors

template_image = cv2.imread(
    "/home/aweinsto/projects/computer-vision/app/assigment2/images/template.png",
)
template_grey = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

img_names = glob(img_mask)
num_images = len(img_names)

pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

obj_points = []
img_points = []
h, w = cv2.imread(img_names[0]).shape[:2]

plt.figure(figsize=figsize)

for i, fn in enumerate(img_names):
    print("processing %s... " % fn)
    imgBGR = cv2.imread(fn)

    if imgBGR is None:
        print("Failed to load", fn)
        continue
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)

    if not (w == img.shape[1] and h == img.shape[0]):
        print(f"assertion failed, skip image {i}")
        continue
    found, corners = cv2.findChessboardCorners(img, pattern_size)
    if not found:
        print("chessboard not found")
        continue
    print(f"{fn}... OK")
    img_points.append(corners.reshape(-1, 2))
    obj_points.append(pattern_points)

rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

video_path = "/home/aweinsto/projects/computer-vision/app/assigment2/movie/origin.mp4"

# ===== video input, output and metadata
input_video = cv2.VideoCapture(video_path)
width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = input_video.get(cv2.CAP_PROP_FPS)
output_video = cv2.VideoWriter(
    "/home/aweinsto/projects/computer-vision/app/assigment2/movie/output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    60,
    (width, height),
)
feature_extractor = cv2.SIFT_create()


def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img

# ========== run on all frames
index = 0
kp_template, desc_template = feature_extractor.detectAndCompute(template_grey, None)

while True:
    ok, frame = input_video.read()
    print(index)
    if not ok:
        break

    index+=1
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    H, mask = cv2.findHomography(good_kp_template, good_kp_frame, cv2.RANSAC, 5.0)
    mask_inliers = mask.ravel().astype(bool)
    # Subselect only those keypoints that obey the homography
    good_kp_template_inliers = good_kp_template[mask_inliers]
    good_kp_frame_inliers    = good_kp_frame[mask_inliers]
    template_h_pix, template_w_pix = template_grey.shape[:2]
    W_real = 25.5   # cm
    H_real = 36  # cm
    template_points_3D = np.hstack([
        good_kp_template_inliers * [W_real / template_w_pix, H_real / template_h_pix],
        np.zeros((len(good_kp_template_inliers), 1))
    ]).astype(np.float32)
    frame_points_2D = good_kp_frame_inliers.reshape(-1, 1, 2).astype(np.float32)
    _, r_vec, t_vec = cv2.solvePnP(template_points_3D, frame_points_2D, camera_matrix, dist_coefs)
    cube_3D = np.float32([
        [0, 0, 0],   [3, 0, 0],   [3, 3, 0],   [0, 3, 0],
        [0, 0, -3],  [3, 0, -3],  [3, 3, -3],  [0, 3, -3]
    ])
    cube_2D, _ = cv2.projectPoints(cube_3D, r_vec, t_vec, camera_matrix, dist_coefs)

    # draw the cube
    frame = draw(frame, cube_2D)

    # 5) Write out to video
    output_video.write(frame)

    # ++++++++ solve PnP to get cam pose (r_vec and t_vec)
    # `cv2.solvePnP` is a function that receives:
    # - xyz of the template in centimeter in camera world (x,3)
    # - uv coordinates (x,2) of frame that corresponds to the xyz triplets
    # - camera K
    # - camera dist_coeffs
    # and outputs the camera pose (r_vec and t_vec) such that the uv is aligned with the xyz.
    #
    # NOTICE: the first input to `cv2.solvePnP` is (x,3) vector of xyz in centimeter- but we have the template keypoints in uv
    # because they are all on the same plane we can assume z=0 and simply rescale each keypoint to the ACTUAL WORLD SIZE IN CM.
    # For this we just need the template width and height in cm.
    #
    # this part is 2 rows

    # ++++++ draw object with r_vec and t_vec on top of rgb frame
    # We saw how to draw cubes in camera calibration. (copy paste)
    # after this works you can replace this with the draw function from the renderer class renderer.draw() (1 line)

output_video.release()
input_video.release()