import cv2
import matplotlib.pyplot as plt
from typing import Tuple
import numpy as np


def otsu_segmentation(image_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Otsu阈值分割

    参数:
        image_path: 输入图像路径

    返回:
        original_image: 原始图像
        segmented: 分割结果
    """
    # 读取图像
    image = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用Otsu阈值分割
    _, otsu_result = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return original_rgb, otsu_result


# 执行分割
original_img, otsu_img = otsu_segmentation("balloon.png")

# 可视化结果
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(original_img)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(otsu_img, cmap="gray")
axes[1].set_title("Otsu Threshold Segmentation")
axes[1].axis("off")

plt.tight_layout()
plt.show()
