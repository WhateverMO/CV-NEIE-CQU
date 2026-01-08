import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 设置相机参数 (根据你的作业数据)
K = np.array([[2961.68, 0, 2054.53], [0, 2962.39, 1466.07], [0, 0, 1]])
Dist = np.array([0.2857, -2.8065, -0.0059, 0.0058, 7.2826])


def undistort_img(img):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, Dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, K, Dist, None, newcameramtx)
    return dst


def run_task2():
    # 读取并校正图像
    img1 = cv2.imread("sixth/image1.jpg", 0)
    img2 = cv2.imread("sixth/image2.jpg", 0)

    img1 = undistort_img(img1)
    img2 = undistort_img(img2)

    # 特征提取
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
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
    print(f"Matched Points: {len(pts1)}")

    # 2. 计算本质矩阵 E
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    print("Essential Matrix Calculated.")

    # 3. 恢复位姿 (R, t)
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)
    print(f"Rotation:\n{R}\nTranslation:\n{t}")

    # 4. 三角化
    # 投影矩阵 P1 = K[I|0], P2 = K[R|t]
    P1 = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(K, np.hstack((R, t)))

    # 将点转换为 (2, N) 格式
    pts1_T = pts1[mask_pose.ravel() > 0].T
    pts2_T = pts2[mask_pose.ravel() > 0].T

    # OpenCV 三角化
    points_4d = cv2.triangulatePoints(P1, P2, pts1_T, pts2_T)
    points_3d = points_4d[:3] / points_4d[3]

    # 5. 可视化
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 画点云
    ax.scatter(points_3d[0], points_3d[1], points_3d[2], c="g", s=2, label="3D Points")

    # 画相机位置 (原点 和 t)
    ax.scatter(0, 0, 0, c="r", marker="^", s=100, label="Cam 1")
    ax.scatter(t[0], t[1], t[2], c="b", marker="^", s=100, label="Cam 2")

    # 设置轴标签
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title("Two-View 3D Reconstruction")
    plt.show()


if __name__ == "__main__":
    run_task2()
