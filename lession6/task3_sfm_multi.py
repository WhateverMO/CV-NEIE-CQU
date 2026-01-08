import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
import glob

# 相机内参 (同任务2)
K = np.array([[2961.68, 0, 2054.53], [0, 2962.39, 1466.07], [0, 0, 1]])


def get_matches(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return np.array([]), np.array([])

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    pts1 = []
    pts2 = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    return np.float32(pts1), np.float32(pts2)


def run_task3():
    # 读取序列图片
    img_dir = "sequence_data"  # 请确保此处有 >10 张图片
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))

    if len(img_files) < 2:
        print("图片数量不足，请在 sequence_data 文件夹放入图片序列")
        return

    # 全局变量
    all_points_3d = []
    camera_centers = [np.array([0, 0, 0])]  # 第一个相机在原点

    # 当前全局位姿 (R_global, t_global)
    R_curr = np.eye(3)
    t_curr = np.zeros((3, 1))

    print(f"Found {len(img_files)} images. Starting sequential reconstruction...")

    for i in range(len(img_files) - 1):
        print(f"Processing Pair: {i} -> {i + 1}")
        img1 = cv2.imread(img_files[i], 0)
        img2 = cv2.imread(img_files[i + 1], 0)

        # 1. 匹配
        pts1, pts2 = get_matches(img1, img2)
        if len(pts1) < 8:
            continue

        # 2. 估计相对运动 (Relative Motion)
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

        # 3. 三角化 (在相对坐标系中)
        P1_rel = np.dot(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
        P2_rel = np.dot(K, np.hstack((R_rel, t_rel)))

        pts1_valid = pts1[mask_pose.ravel() > 0].T
        pts2_valid = pts2[mask_pose.ravel() > 0].T

        points_4d = cv2.triangulatePoints(P1_rel, P2_rel, pts1_valid, pts2_valid)
        points_3d_local = (points_4d[:3] / points_4d[3]).T  # Shape: (N, 3)

        # 4. 更新全局位姿
        # T_global = T_current * T_relative
        # t_new = t_curr + R_curr * t_rel
        # R_new = R_curr * R_rel
        t_curr = t_curr + R_curr.dot(t_rel)
        R_curr = R_curr.dot(R_rel)

        camera_centers.append(t_curr.ravel())

        # 5. 将局部3D点转换到全局坐标系
        # X_global = R_curr * X_local + t_curr (近似处理，简化版)
        # 注意：这里简化了，严谨做法需要维护全局点云地图
        # 为了可视化效果，我们简单地将这一对图产生的点变换到全局
        # 实际 SfM 中点是共享的，这里作为演示只做累积

        # 变换: X_global = R_current_frame_base * X_local + t_current_frame_base
        # 这里的 R_curr 和 t_curr 已经是第 i+1 帧的姿态了，
        # 而 points_3d_local 是基于第 i 帧为原点算的。
        # 所以变换矩阵应该是第 i 帧的全局位姿 (R_prev, t_prev)。
        # 为了简化作业，我们直接存入列表用于展示结构密度。

        # 简单可视化策略：只可视化每一段相对重建的点云（会断开），或者做简单变换
        # 这里做简单变换：
        R_prev = R_curr.dot(R_rel.T)  # 回推上一帧旋转
        t_prev = t_curr - R_prev.dot(t_rel)  # 回推上一帧平移

        for p in points_3d_local:
            # 过滤太远的点
            if np.linalg.norm(p) < 50:
                p_global = R_prev.dot(p) + t_prev.ravel()
                all_points_3d.append(p_global)

    # 6. 可视化
    all_points_3d = np.array(all_points_3d)
    camera_centers = np.array(camera_centers)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 降采样点云以加快显示
    if len(all_points_3d) > 0:
        skip = max(1, len(all_points_3d) // 2000)
        ax.scatter(
            all_points_3d[::skip, 0],
            all_points_3d[::skip, 1],
            all_points_3d[::skip, 2],
            c="b",
            s=1,
            alpha=0.3,
            label="Structure",
        )

    # 画相机轨迹
    ax.plot(
        camera_centers[:, 0],
        camera_centers[:, 1],
        camera_centers[:, 2],
        "-r",
        linewidth=2,
        label="Trajectory",
    )
    ax.scatter(
        camera_centers[:, 0],
        camera_centers[:, 1],
        camera_centers[:, 2],
        c="r",
        marker="o",
    )

    ax.set_title(f"Multi-View SfM ({len(img_files)} images)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    run_task3()
