#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Least Squares Line Fitting using RANSAC.
Input: line_image.jpg
Output: fitted_line_result.jpg
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor


def extract_edge_points(image_path, canny_low=50, canny_high=150):
    """Extracts edge points from an image using Canny detector.
    Args:
        image_path (str): Path to the input image.
        canny_low (int): Canny low threshold.
        canny_high (int): Canny high threshold.
    Returns:
        tuple: (x_coords, y_coords, gray_image)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    edges = cv2.Canny(img, canny_low, canny_high)
    points = np.column_stack(np.where(edges > 0))
    if len(points) == 0:
        raise ValueError("No edge points detected.")
    x = points[:, 1]
    y = points[:, 0]
    return x, y, img


def robust_fit_line(x, y, residual_thresh=2.0, max_trials=100):
    """Fits a line using RANSAC.
    Args:
        x (ndarray): x-coordinates of points.
        y (ndarray): y-coordinates of points.
        residual_thresh (float): RANSAC residual threshold.
        max_trials (int): Maximum RANSAC iterations.
    Returns:
        tuple: (slope, intercept, inlier_mask)
    """
    X = x.reshape(-1, 1)
    ransac = RANSACRegressor(
        residual_threshold=residual_thresh, max_trials=max_trials, random_state=42
    )
    ransac.fit(X, y)
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_
    inlier_mask = ransac.inlier_mask_
    return slope, intercept, inlier_mask


def visualize_fit(orig_img, x, y, slope, intercept, inliers, title, save_path):
    """Visualizes the fitting result.
    Args:
        orig_img: Original grayscale image.
        x, y: Point coordinates.
        slope, intercept: Line parameters.
        inliers: Boolean mask for inliers.
        title (str): Plot title.
        save_path (str): Path to save the result image.
    """
    # Convert to BGR for color drawing
    result_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
    h, w = orig_img.shape
    # Draw points
    for i in range(len(x)):
        color = (255, 0, 0) if inliers[i] else (0, 0, 255)  # blue: inlier, red: outlier
        cv2.circle(result_img, (int(x[i]), int(y[i])), 2, color, -1)
    # Draw fitted line
    x_line = np.array([0, w - 1])
    y_line = slope * x_line + intercept
    # Clip line to image bounds
    y_line = np.clip(y_line, 0, h - 1)
    cv2.line(
        result_img,
        (int(x_line[0]), int(y_line[0])),
        (int(x_line[1]), int(y_line[1])),
        (0, 255, 0),
        2,
    )  # green line
    # Save
    cv2.imwrite(save_path, result_img)
    # Show
    plt.figure()
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()
    print(f"Saved: {save_path}")
    print(f"Line: y = {slope:.3f}x + {intercept:.3f}")
    print(f"Inlier count: {np.sum(inliers)} / {len(x)}")


if __name__ == "__main__":
    IMAGE_PATH = "line_image.jpg"
    OUTPUT_PREFIX = "fitted_line"
    PARAM_SETS = [
        {"residual_threshold": 2.0, "max_trials": 100},
        {"residual_threshold": 5.0, "max_trials": 100},
        {"residual_threshold": 2.0, "max_trials": 200},
        {"residual_threshold": 5.0, "max_trials": 200},
    ]
    try:
        x_pts, y_pts, gray_img = extract_edge_points(IMAGE_PATH)
        for idx, params in enumerate(PARAM_SETS):
            print(f"\n--- Parameter Set {idx + 1}: {params} ---")
            s, i, mask = robust_fit_line(
                x_pts, y_pts, params["residual_threshold"], params["max_trials"]
            )
            out_name = f"{OUTPUT_PREFIX}_set{idx + 1}.jpg"
            title = f"res_thresh={params['residual_threshold']}, max_trials={params['max_trials']}"
            visualize_fit(gray_img, x_pts, y_pts, s, i, mask, title, out_name)
    except Exception as e:
        print(f"Error: {e}")
