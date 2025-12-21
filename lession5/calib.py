#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Calibration using a Checkerboard.
Input: calib_1.jpg, calib_2.jpg, ...
Output: camera_matrix.npy, dist_coeffs.npy, calibration_result.txt, undistorted_example.jpg
"""

import cv2
import numpy as np
import glob
import os

# ------------------------------
# Parameters
# ------------------------------
# 注意：这里的CHECKERBOARD参数可能需要根据你的棋盘格图片调整
# 检查你的棋盘格内角点数量，不是方格数量
# 比如如果是8x6的棋盘格（8列7行方格），内角点就是7x5
CHECKERBOARD = (7, 5)  # 尝试这个值，根据你的棋盘格调整
SQUARE_SIZE = 25.0  # 方格的尺寸，单位毫米
IMAGES_PATH = "calib_*.jpg"  # 改为当前目录
# Termination criteria for cornerSubPix
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ------------------------------
# Prepare object points (3D points in world coordinate)
# ------------------------------
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# ------------------------------
# Find corners in all images
# ------------------------------
images = glob.glob(IMAGES_PATH)
# 也尝试其他可能的命名格式
if len(images) == 0:
    images = glob.glob("*.jpg")
    images = [img for img in images if "calib" in img.lower()]

print(f"Searching for calibration images...")
print(f"Found {len(images)} calibration images.")
for img in images[:5]:  # 显示前5个找到的文件
    print(f"  - {os.path.basename(img)}")
if len(images) > 5:
    print(f"  ... and {len(images) - 5} more")

if len(images) < 3:
    print("Error: Need at least 3 images for calibration. Exiting.")
    exit()

# Variables to store image size
img_size = None
first_valid_image = None

# 先显示一张图片看看棋盘格
if len(images) > 0:
    test_img = cv2.imread(images[0])
    if test_img is not None:
        print(f"\nFirst image size: {test_img.shape}")
        print("Displaying first image for reference...")
        cv2.imshow("First calibration image (press any key)", test_img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

for i, fname in enumerate(images):
    print(f"\n[{i + 1}/{len(images)}] Processing: {os.path.basename(fname)}")
    img = cv2.imread(fname)
    if img is None:
        print(f"  ✗ Could not read image. Skipping.")
        continue

    # 缩小图片以加快处理速度（如果图片太大）
    scale_factor = 0.5
    if img.shape[0] > 2000 or img.shape[1] > 2000:
        print(f"  Resizing large image from {img.shape}...")
        img = cv2.resize(
            img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA
        )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Store image size from first valid image
    if img_size is None:
        img_size = gray.shape[::-1]  # (width, height)
        first_valid_image = img

    # Find chessboard corners
    print(f"  Looking for {CHECKERBOARD} corners...")
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK,
    )

    if ret:
        objpoints.append(objp)
        # Refine corner locations
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        imgpoints.append(corners_refined)

        # Draw and display corners
        img_with_corners = img.copy()
        cv2.drawChessboardCorners(img_with_corners, CHECKERBOARD, corners_refined, ret)
        # 在角点上显示坐标
        cv2.putText(
            img_with_corners,
            f"Found {len(corners)} corners",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Chessboard Corners", img_with_corners)
        cv2.waitKey(300)  # display for 300 ms
        print(f"  ✓ Found {len(corners)} corners.")
    else:
        print(f"  ✗ Chessboard corners not found.")
        # 尝试不同的棋盘格尺寸
        print(f"  Trying alternative checkerboard sizes...")
        alt_sizes = [(6, 4), (8, 6), (9, 6), (7, 7)]
        for alt_size in alt_sizes:
            ret, corners = cv2.findChessboardCorners(
                gray,
                alt_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if ret:
                print(f"  ✓ Found corners with size {alt_size}")
                # 需要重新创建objp
                temp_objp = np.zeros((alt_size[0] * alt_size[1], 3), np.float32)
                temp_objp[:, :2] = np.mgrid[0 : alt_size[0], 0 : alt_size[1]].T.reshape(
                    -1, 2
                )
                temp_objp *= SQUARE_SIZE
                objpoints.append(temp_objp)

                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), CRITERIA
                )
                imgpoints.append(corners_refined)

                img_with_corners = img.copy()
                cv2.drawChessboardCorners(
                    img_with_corners, alt_size, corners_refined, ret
                )
                cv2.putText(
                    img_with_corners,
                    f"Size {alt_size}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Chessboard Corners", img_with_corners)
                cv2.waitKey(300)
                break

cv2.destroyAllWindows()

# ------------------------------
# Check if we have enough calibration data
# ------------------------------
if len(objpoints) < 3:
    print(f"\nError: Only found {len(objpoints)} valid images with chessboard corners.")
    print("Need at least 3 images for calibration.")
    print("\nPossible issues:")
    print("1. Wrong CHECKERBOARD parameter (currently set to {CHECKERBOARD})")
    print("2. Chessboard not fully visible in images")
    print("3. Poor lighting or focus")
    print("\nPlease check your calibration images and adjust CHECKERBOARD parameter.")
    exit()

if img_size is None:
    print("\nError: No valid images found. Exiting.")
    exit()

print(f"\n--- Starting calibration with {len(objpoints)} valid images ---")
print(f"Image size: {img_size}")

# ------------------------------
# Camera Calibration
# ------------------------------
print("\nCalibrating camera...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_size, None, None
)
print("Calibration finished.")
print(f"\nCamera matrix:\n{mtx}")
print(f"\nDistortion coefficients:\n{dist}")
print(f"\nCalibration error (RMS): {ret:.4f} pixels")

# ------------------------------
# Save calibration results
# ------------------------------
np.save("camera_matrix.npy", mtx)
np.save("dist_coeffs.npy", dist)
with open("calibration_result.txt", "w") as f:
    f.write(f"CALIBRATION RESULTS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Number of images used: {len(objpoints)}\n")
    f.write(f"Image size: {img_size}\n")
    f.write(f"Checkerboard pattern: {CHECKERBOARD}\n")
    f.write(f"Square size: {SQUARE_SIZE} mm\n\n")
    f.write(f"Camera Matrix:\n{mtx}\n\n")
    f.write(f"Distortion Coefficients:\n{dist}\n\n")
    f.write(f"Reprojection Error (RMS): {ret:.4f} pixels\n")
print("\nCalibration results saved to files.")

# ------------------------------
# Undistort an example image
# ------------------------------
if first_valid_image is not None:
    test_img = first_valid_image.copy()
    h, w = test_img.shape[:2]

    # Get optimal new camera matrix
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Undistort
    undistorted = cv2.undistort(test_img, mtx, dist, None, new_mtx)

    # Crop the image
    x, y, uw, uh = roi
    if uw > 0 and uh > 0:  # 确保ROI有效
        undistorted = undistorted[y : y + uh, x : x + uw]

    # Save and show
    cv2.imwrite("undistorted_example.jpg", undistorted)

    # Create comparison image
    # 调整大小以便并排显示
    max_height = 600
    if h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        new_h = max_height
        test_img_resized = cv2.resize(test_img, (new_w, new_h))
        undistorted_resized = cv2.resize(undistorted, (new_w, new_h))
    else:
        test_img_resized = test_img
        undistorted_resized = undistorted

    comparison = np.hstack([test_img_resized, undistorted_resized])

    # 添加标签
    cv2.putText(
        comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    cv2.putText(
        comparison,
        "Undistorted",
        (w // 2 + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imwrite("comparison_original_vs_undistorted.jpg", comparison)

    cv2.imshow("Original vs Undistorted (press any key)", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Comparison image saved as 'comparison_original_vs_undistorted.jpg'.")

# ------------------------------
# Compute reprojection error for each image
# ------------------------------
print("\n--- Reprojection errors per image ---")
total_error = 0.0
errors = []
for i in range(len(objpoints)):
    imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
    total_error += error
    errors.append(error)
    print(f"Image {i + 1}: {error:.4f} pixels")

mean_error = total_error / len(objpoints)
print(f"\nMean reprojection error: {mean_error:.4f} pixels")
print(f"Min error: {min(errors):.4f} pixels")
print(f"Max error: {max(errors):.4f} pixels")

# ------------------------------
# Display calibration summary
# ------------------------------
print("\n" + "=" * 50)
print("CALIBRATION SUMMARY")
print("=" * 50)
print(f"Number of images processed: {len(images)}")
print(f"Number of valid images used: {len(objpoints)}")
print(f"Image resolution: {img_size}")
print(f"Checkerboard pattern: {CHECKERBOARD}")
print(f"Square size: {SQUARE_SIZE} mm")
print(f"Overall RMS error: {ret:.4f} pixels")
print(f"Mean per-image error: {mean_error:.4f} pixels")

if mean_error < 1.0:
    print("\n✅ Excellent calibration! (Error < 1.0 pixel)")
elif mean_error < 2.0:
    print("\n✓ Good calibration! (Error < 2.0 pixels)")
elif mean_error < 3.0:
    print("\n⚠ Acceptable calibration (Error < 3.0 pixels)")
else:
    print("\n⚠ High error, consider recalibrating")

print("\nFiles saved:")
print("  - camera_matrix.npy")
print("  - dist_coeffs.npy")
print("  - calibration_result.txt")
print("  - undistorted_example.jpg")
print("  - comparison_original_vs_undistorted.jpg")
print("=" * 50)
