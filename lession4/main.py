import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def canny_compare(image_path="test.png"):
    if not os.path.exists(image_path):
        print(f"文件不存在: {image_path}")
        return

    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # 尝试用BGR读取
        img_color = cv2.imread(image_path)
        if img_color is None:
            print("无法读取图像")
            return
        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # 1. 自动计算阈值
    v = np.median(img)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))

    # 手动Canny实现
    # 步骤1: 高斯滤波
    sigma = 1.2
    size = 5
    blurred = cv2.GaussianBlur(img, (size, size), sigma)

    # 步骤2: 梯度计算 (使用Scharr算子提高精度)
    grad_x = cv2.Scharr(blurred, cv2.CV_64F, 1, 0)
    grad_y = cv2.Scharr(blurred, cv2.CV_64F, 0, 1)

    # 梯度幅值和方向
    magnitude = np.hypot(grad_x, grad_y)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
    angle[angle < 0] += 180

    # 步骤3: 改进的非极大值抑制
    height, width = magnitude.shape
    suppressed = np.zeros_like(magnitude, dtype=np.uint8)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            grad_angle = angle[i, j]
            grad_mag = magnitude[i, j]

            # 离散化角度
            if (0 <= grad_angle < 22.5) or (157.5 <= grad_angle <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= grad_angle < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 67.5 <= grad_angle < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            else:  # 112.5 <= grad_angle < 157.5
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if grad_mag >= q and grad_mag >= r:
                suppressed[i, j] = grad_mag

    # 步骤4: 双阈值
    high_thresh = np.max(suppressed) * 0.3
    low_thresh = high_thresh * 0.5

    strong = np.zeros_like(suppressed, dtype=np.uint8)
    weak = np.zeros_like(suppressed, dtype=np.uint8)

    strong[suppressed >= high_thresh] = 255
    weak[(suppressed >= low_thresh) & (suppressed < high_thresh)] = 128

    # 步骤5: 改进的边缘连接
    result = strong.copy()

    # 使用队列进行边缘连接
    from collections import deque

    height, width = strong.shape
    visited = np.zeros_like(strong, dtype=bool)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if strong[i, j] == 255 and not visited[i, j]:
                queue = deque([(i, j)])
                visited[i, j] = True

                while queue:
                    x, y = queue.popleft()

                    # 检查8邻域
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue

                            nx, ny = x + dx, y + dy
                            if 0 <= nx < height and 0 <= ny < width:
                                if not visited[nx, ny] and weak[nx, ny] == 128:
                                    result[nx, ny] = 255
                                    visited[nx, ny] = True
                                    queue.append((nx, ny))

    manual_edges = result

    # OpenCV Canny
    opencv_edges = cv2.Canny(img, lower, upper)

    # 计算相似度
    similarity = 100.0 * np.sum(manual_edges == opencv_edges) / manual_edges.size

    print(f"图像: {os.path.basename(image_path)}")
    print(f"尺寸: {width} x {height}")
    print(f"手动阈值: 低={int(low_thresh):.0f}, 高={int(high_thresh):.0f}")
    print(f"OpenCV阈值: 低={lower}, 高={upper}")
    print(f"相似度: {similarity:.2f}%")

    # 显示结果
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    images = [
        (img, "Original", "gray"),
        (manual_edges, f"Manual Canny\n{np.sum(manual_edges > 0)} edges", "gray"),
        (opencv_edges, f"OpenCV Canny\n{np.sum(opencv_edges > 0)} edges", "gray"),
        (
            np.hstack([manual_edges, opencv_edges]),
            f"Comparison\nSimilarity: {similarity:.1f}%",
            "gray",
        ),
        (cv2.absdiff(manual_edges, opencv_edges), "Difference", "hot"),
        (np.zeros_like(img), "", "gray"),
    ]

    for idx, (image, title, cmap) in enumerate(images[:5]):
        ax = axes[idx // 3, idx % 3]
        ax.imshow(image, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    # 直方图
    ax = axes[1, 2]
    bins = np.linspace(0, 255, 50)
    ax.hist(
        manual_edges.ravel(),
        bins=bins,
        alpha=0.5,
        label="Manual",
        color="blue",
        density=True,
    )
    ax.hist(
        opencv_edges.ravel(),
        bins=bins,
        alpha=0.5,
        label="OpenCV",
        color="red",
        density=True,
    )
    ax.set_title("Pixel Distribution")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Canny Edge Detection Comparison\n{os.path.basename(image_path)}", fontsize=14
    )
    plt.tight_layout()
    plt.show()


# 运行
if __name__ == "__main__":
    png_files = [f for f in os.listdir(".") if f.lower().endswith(".png")]
    if png_files:
        canny_compare(png_files[0])
    else:
        print("没有找到PNG文件，请确保有test.png文件")
        canny_compare("test.png")
