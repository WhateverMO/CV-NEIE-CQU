import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_images_and_match(img1_path, img2_path):
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Error: 找不到图片 {img1_path} 或 {img2_path}")
        return None, None, None, None

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    return img1, img2, pts1, pts2


def draw_matches_custom(img1, img2, pts1, pts2, mask, title):
    """绘制匹配连线"""
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    vis = np.zeros((max(h1, h2), w1 + w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1 : w1 + w2] = img2
    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        if mask[i]:
            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = (int(pt2[0] + w1), int(pt2[1]))
            cv2.line(vis, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(vis, pt1, 3, (0, 0, 255), -1)
            cv2.circle(vis, pt2, 3, (0, 0, 255), -1)

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


def run_task1():
    img1_path = "sixth/image1.jpg"
    img2_path = "sixth/image2.jpg"

    img1, img2, pts1, pts2 = load_images_and_match(img1_path, img2_path)
    if img1 is None:
        return

    # 测试不同的阈值
    thresholds = [0.5, 1.0, 5.0]

    for thresh in thresholds:
        print(f"\n--- Testing RANSAC Threshold: {thresh} ---")
        # 计算基础矩阵
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, thresh, 0.99)

        inliers_cnt = np.sum(mask)
        print(f"Inliers found: {inliers_cnt} / {len(pts1)}")
        print(f"Fundamental Matrix:\n{F}")

        # 可视化
        draw_matches_custom(
            img1,
            img2,
            pts1,
            pts2,
            mask.ravel(),
            f"RANSAC Threshold = {thresh} (Inliers: {inliers_cnt})",
        )


if __name__ == "__main__":
    run_task1()
