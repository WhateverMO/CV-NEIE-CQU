#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hough Circle Detection.
Input: circle_image.jpg
Output: circle_detection_with_gradient.jpg, circle_detection_without_gradient.jpg
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_circles(image_path, param2_value, use_gradient_label="Default"):
    """Detects circles using HoughCircles.
    Args:
        image_path (str): Path to input image.
        param2_value (float): Accumulator threshold for circle centers.
        use_gradient_label (str): Label for display.
    Returns:
        tuple: (output_image, circles_detected)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Reduce noise
    gray_blur = cv2.medianBlur(gray, 5)
    # Detect circles
    circles = cv2.HoughCircles(
        gray_blur,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,  # adjust based on image
        param1=100,  # upper threshold for Canny
        param2=param2_value,
        minRadius=20,
        maxRadius=100,
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            # Draw circle outline
            cv2.circle(output, center, radius, (0, 255, 0), 2)
            # Draw circle center
            cv2.circle(output, center, 2, (0, 0, 255), 3)
        print(f"[{use_gradient_label}] Detected {len(circles[0])} circles.")
    else:
        print(f"[{use_gradient_label}] No circles detected.")
    return output, circles


if __name__ == "__main__":
    IMAGE_PATH = "circle_image.jpg"
    # Case 1: With gradient (stricter, higher param2)
    result_with, circles_with = detect_circles(
        IMAGE_PATH, param2_value=30, use_gradient_label="With Gradient"
    )
    # Case 2: Without gradient (looser, lower param2)
    result_without, circles_without = detect_circles(
        IMAGE_PATH, param2_value=20, use_gradient_label="Without Gradient"
    )
    # Save results
    cv2.imwrite("circle_detection_with_gradient.jpg", result_with)
    cv2.imwrite("circle_detection_without_gradient.jpg", result_without)
    # Display
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(result_with, cv2.COLOR_BGR2RGB))
    plt.title("With Gradient (param2=30)")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result_without, cv2.COLOR_BGR2RGB))
    plt.title("Without Gradient (param2=20)")
    plt.axis("off")
    plt.show()
