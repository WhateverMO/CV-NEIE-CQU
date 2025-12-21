#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RANSAC Line Fitting (Custom Implementation).
Input: line_image.jpg
Output: ransac_result.jpg
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def ransac_line_fit(points, max_iter=100, dist_thresh=2.0, min_inliers=10):
    """Custom RANSAC for line fitting.
    Args:
        points (ndarray): Nx2 array of (x, y) points.
        max_iter (int): Maximum RANSAC iterations.
        dist_thresh (float): Distance threshold for inliers.
        min_inliers (int): Minimum inliers to accept a model.
    Returns:
        tuple: (best_a, best_b, best_c, best_inlier_mask)
                Line: a*x + b*y + c = 0
    """
    best_model = None
    best_inlier_mask = None
    best_inlier_count = 0
    num_points = points.shape[0]
    for _ in range(max_iter):
        # 1. Randomly sample 2 points
        idx = np.random.choice(num_points, 2, replace=False)
        p1, p2 = points[idx]
        # 2. Compute line parameters: a*x + b*y + c = 0
        a = p1[1] - p2[1]
        b = p2[0] - p1[0]
        c = p1[0] * p2[1] - p2[0] * p1[1]
        norm = np.sqrt(a * a + b * b) + 1e-6
        a, b, c = a / norm, b / norm, c / norm
        # 3. Compute distances
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c)
        # 4. Find inliers
        inlier_mask = distances < dist_thresh
        inlier_count = np.sum(inlier_mask)
        # 5. Update best model
        if inlier_count > best_inlier_count and inlier_count >= min_inliers:
            best_inlier_count = inlier_count
            best_inlier_mask = inlier_mask.copy()
            # Store normalized parameters
            best_model = (a, b, c)
    if best_model is None:
        raise RuntimeError("RANSAC failed to find a good model.")
    # 6. Refit using all inliers (least squares)
    a, b, c = best_model
    inlier_pts = points[best_inlier_mask]
    # Solve for y = slope*x + intercept form
    X = inlier_pts[:, 0].reshape(-1, 1)
    y = inlier_pts[:, 1]
    # Use normal equation for stability
    X_design = np.hstack([X, np.ones((len(X), 1))])
    coeffs = np.linalg.lstsq(X_design, y, rcond=None)[0]
    slope = coeffs[0]
    intercept = coeffs[1]
    # Convert back to a,b,c form
    a_new = slope
    b_new = -1.0
    c_new = intercept
    norm_new = np.sqrt(a_new * a_new + b_new * b_new)
    a_new, b_new, c_new = a_new / norm_new, b_new / norm_new, c_new / norm_new
    return a_new, b_new, c_new, best_inlier_mask


def visualize_ransac(orig_img, points, model, inlier_mask, title, save_path):
    """Visualizes RANSAC result.
    Args:
        orig_img: Original image.
        points: All edge points (Nx2).
        model: (a, b, c) line parameters.
        inlier_mask: Boolean mask for inliers.
        title (str): Plot title.
        save_path (str): Path to save image.
    """
    a, b, c = model
    result_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
    h, w = orig_img.shape
    # Draw points
    outliers = points[~inlier_mask]
    inliers = points[inlier_mask]
    for pt in outliers:
        cv2.circle(result_img, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)  # red
    for pt in inliers:
        cv2.circle(result_img, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)  # blue
    # Draw fitted line (convert to slope-intercept for drawing)
    slope = -a / b
    intercept = -c / b
    x_line = np.array([0, w - 1])
    y_line = slope * x_line + intercept
    y_line = np.clip(y_line, 0, h - 1)
    cv2.line(
        result_img,
        (int(x_line[0]), int(y_line[0])),
        (int(x_line[1]), int(y_line[1])),
        (0, 255, 0),
        2,
    )  # green
    # Save and show
    cv2.imwrite(save_path, result_img)
    plt.figure()
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()
    print(f"Saved: {save_path}")
    print(f"Line: {a:.3f}*x + {b:.3f}*y + {c:.3f} = 0")
    print(f"Inliers: {np.sum(inlier_mask)} / {len(points)}")


if __name__ == "__main__":
    IMAGE_PATH = "line_image.jpg"
    # Get edge points
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 50, 150)
    pts = np.column_stack(np.where(edges > 0))
    pts[:, [0, 1]] = pts[:, [1, 0]]  # (row, col) -> (x, y)
    if len(pts) < 10:
        raise ValueError("Too few edge points.")
    PARAM_SETS = [
        {"max_iterations": 100, "distance_threshold": 2.0},
        {"max_iterations": 500, "distance_threshold": 1.5},
        {"max_iterations": 1000, "distance_threshold": 1.0},
    ]
    for idx, param in enumerate(PARAM_SETS):
        print(f"\n--- Testing Set {idx + 1}: {param} ---")
        a, b, c, mask = ransac_line_fit(
            pts,
            max_iter=param["max_iterations"],
            dist_thresh=param["distance_threshold"],
        )
        out_name = f"ransac_result_set{idx + 1}.jpg"
        title = f"max_iter={param['max_iterations']}, dist_thresh={param['distance_threshold']}"
        visualize_ransac(img, pts, (a, b, c), mask, title, out_name)
