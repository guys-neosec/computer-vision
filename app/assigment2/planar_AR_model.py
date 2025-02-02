from glob import glob

import cv2
import trimesh
import pyrender
import numpy as np
from scipy.spatial.transform import Rotation as R

# ======= constants
np.infty = np.inf
square_size = 2.6
img_mask = (
    "/Users/gstrauss/Reichman_University/computer-vision/app/assigment2/calibration_2/*"
)
pattern_size = (9, 6)

# === template image keypoints and descriptors
template_image = cv2.imread(
    "/Users/gstrauss/Reichman_University/computer-vision/app/assigment2/images/template.png",
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

for i, fn in enumerate(img_names):
    print("processing %s... " % fn)
    imgBGR = cv2.imread(fn)

    if imgBGR is None:
        print("Failed to load", fn)
        continue

    img = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)

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

# Calibrate the camera
rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(
    obj_points, img_points, (w, h), None, None
)


def validate_calibration(img_names, obj_points, img_points, camera_matrix, dist_coefs):
    for i, fn in enumerate(img_names):
        img = cv2.imread(fn)
        if img is None:
            print(f"Failed to load {fn}")
            continue

        # Reproject the calibration points
        reprojected_points, _ = cv2.projectPoints(
            obj_points[i], _rvecs[i], _tvecs[i], camera_matrix, dist_coefs
        )
        reprojected_points = reprojected_points.reshape(-1, 2)

        # Draw the original and reprojected points
        for (x, y), (rx, ry) in zip(img_points[i], reprojected_points, strict=False):
            cv2.circle(
                img, (int(x), int(y)), 5, (0, 255, 0), -1
            )  # Original points in green
            cv2.circle(
                img, (int(rx), int(ry)), 5, (0, 0, 255), -1
            )  # Reprojected points in red

        # Calculate reprojection error
        error = np.linalg.norm(img_points[i] - reprojected_points, axis=1).mean()
        print(f"Reprojection error for {fn}: {error:.2f} pixels")

        # Display the image with reprojected points
        # cv2.imshow("Reprojection", img)
        # cv2.waitKey(500)  # Display each image for 500 ms
    #
    # cv2.destroyAllWindows()


# Call the validation function
validate_calibration(img_names, obj_points, img_points, camera_matrix, dist_coefs)


obj_path = "/Users/gstrauss/Downloads/Personal/cyborg_cowboy_potato_animated/scene.gltf"

mesh = trimesh.load(obj_path, skip_materials=False)
mesh = mesh.to_mesh()
scale_factor = 4
mesh.apply_scale(scale_factor)
renderer = None
scene = pyrender.Scene(bg_color=(0, 0, 0, 0), ambient_light=(0.2, 0.2, 0.2, 1.0))


def render_3d_object(frame, r_vec, t_vec, camera_matrix, dist_coeffs):
    if mesh.is_empty:
        print("Mesh is empty, please check the file.")
        return frame

    # Scale the model appropriately

    # Convert to PyRender mesh
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)

    # Scene setup
    fx, fy, cx, cy = camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], \
        camera_matrix[1, 2]
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=0.1, zfar=1000.0)
    R_mat, _ = cv2.Rodrigues(r_vec)


    fixed_rotation = R.from_euler('x', -90, degrees=True).as_matrix()
    object_pose = np.eye(4)
    object_pose[:3, :3] = fixed_rotation
    object_pose[:3, 3] = np.array([0,0,0])
    scene.add(pyrender_mesh, pose=object_pose)
    camera_pose = np.eye(4)
    res_R, _ = cv2.Rodrigues(r_vec)
    camera_pose[0:3, 0:3] = res_R.T
    camera_pose[0:3, 3] = (-res_R.T @ t_vec).flatten()
    camera_pose = camera_pose @ np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=5)
    scene.add(light, pose=camera_pose)
    scene.add(camera, pose=camera_pose)


    # Render the scene
    global renderer
    if renderer is None:
        renderer = pyrender.OffscreenRenderer(frame.shape[1], frame.shape[0])
    render, mask = renderer.render(scene)
    render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
    scene.clear()
    mask = np.any(render > [10, 10, 10], axis=-1).astype(np.uint8) * 255

    # Overlay the object on the frame
    blended_frame = frame.copy()
    for c in range(3):
        blended_frame[:, :, c] = np.where(mask > 0, render[:, :, c], frame[:, :, c])


    cv2.imshow("f", blended_frame)
    cv2.waitKey(20)

    return blended_frame



# ===== video input, output and metadata
video_path = "/Users/gstrauss/Downloads/IMG_4643.MOV"
input_video = cv2.VideoCapture(video_path)
width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = input_video.get(cv2.CAP_PROP_FPS)
output_video = cv2.VideoWriter(
    "/Users/gstrauss/Reichman_University/computer-vision/app/assigment2/movie/output_model_2.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

# Initialize SIFT feature extractor
feature_extractor = cv2.SIFT_create()

# Smoothing factor for pose estimation
alpha = 0.95
r_vec_smoothed = None
t_vec_smoothed = None


# Function to draw the cube
def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)

    # Draw pillars in blue
    for i, j in zip(range(4), range(4, 8), strict=False):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)

    # Draw top layer in red
    img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


# Detect keypoints and descriptors in the template image
kp_template, desc_template = feature_extractor.detectAndCompute(template_grey, None)

# ========== Process all frames
index = 0
while True:
    ok, frame = input_video.read()
    if not ok:
        break

    print(f"Processing frame {index}")
    index += 1

    # Convert frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors in the current frame
    kp_frame, desc_frame = feature_extractor.detectAndCompute(frame_gray, None)

    # Match descriptors using BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_template, desc_frame, k=2)

    # Apply ratio test to filter good matches
    good_features = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_features.append(m)

    if len(good_features) < 10:  # Skip frame if not enough good matches
        print("Not enough good matches, skipping frame")
        continue

    # Extract matched keypoints
    good_kp_template = np.array([kp_template[m.queryIdx].pt for m in good_features])
    good_kp_frame = np.array([kp_frame[m.trainIdx].pt for m in good_features])

    # Find homography using RANSAC
    H, mask = cv2.findHomography(good_kp_template, good_kp_frame, cv2.RANSAC, 3.0)
    mask_inliers = mask.ravel().astype(bool)

    # Subselect only inliers
    good_kp_template_inliers = good_kp_template[mask_inliers]
    good_kp_frame_inliers = good_kp_frame[mask_inliers]

    # Define 3D points of the template in real-world coordinates (centimeters)
    template_h_pix, template_w_pix = template_grey.shape[:2]
    # W_real = 16.8  # Width of the template in cm
    # W_real = 17  # Width of the template in cm
    W_real = 28

    # H_real = 22.4  # Height of the template in cm
    # H_real = 22.9  # Height of the template in cm
    H_real = 36
    template_points_3D = np.hstack(
        [
            good_kp_template_inliers
            * [W_real / template_w_pix, H_real / template_h_pix],
            np.zeros((len(good_kp_template_inliers), 1)),
        ]
    ).astype(np.float32)

    # Reshape for solvePnP
    frame_points_2D = good_kp_frame_inliers.reshape(-1, 1, 2).astype(np.float32)

    # Solve PnP to estimate pose
    if r_vec_smoothed is None:
        _, r_vec, t_vec = cv2.solvePnP(
            template_points_3D,
            frame_points_2D,
            camera_matrix,
            dist_coefs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        r_vec_smoothed = r_vec.copy()
        t_vec_smoothed = t_vec.copy()
    else:
        _, r_vec, t_vec = cv2.solvePnP(
            template_points_3D,
            frame_points_2D,
            camera_matrix,
            dist_coefs,
            r_vec_smoothed,
            t_vec_smoothed,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        # Apply exponential smoothing
        r_vec_smoothed = alpha * r_vec_smoothed + (1.0 - alpha) * r_vec
        t_vec_smoothed = alpha * t_vec_smoothed + (1.0 - alpha) * t_vec

    # Define 3D points of the cube
    # cube_3D = np.float32(
    #     [
    #         [0, 0, 0],
    #         [3, 0, 0],
    #         [3, 3, 0],
    #         [0, 3, 0],
    #         [0, 0, -3],
    #         [3, 0, -3],
    #         [3, 3, -3],
    #         [0, 3, -3],
    #     ]
    # )
    #
    # # Project cube points into the frame
    # cube_2D, _ = cv2.projectPoints(
    #     cube_3D, r_vec_smoothed, t_vec_smoothed, camera_matrix, dist_coefs
    # )

    # Draw the cube on the frame
    frame = render_3d_object(frame, r_vec_smoothed, t_vec_smoothed, camera_matrix,
                             dist_coefs)
    # Write the frame to the output video
    output_video.write(frame)

# Release video objects
output_video.release()
input_video.release()
