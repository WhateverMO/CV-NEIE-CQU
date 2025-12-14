import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def double_threshold_segmentation(
    image_path: str, low_threshold: int = 60, high_threshold: int = 120
) -> Tuple[np.ndarray, np.ndarray]:
    """
    双阈值分割函数

    参数:
        image_path: 输入图像路径
        low_threshold: 低阈值
        high_threshold: 高阈值

    返回:
        gray_image: 灰度图像
        segmented: 分割结果
    """
    # 读取图像并转换为灰度
    image = Image.open(image_path)
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # 应用低阈值和高阈值
    _, low_mask = cv2.threshold(gray_image, low_threshold, 255, cv2.THRESH_BINARY)
    _, high_mask = cv2.threshold(gray_image, high_threshold, 255, cv2.THRESH_BINARY_INV)

    # 结合两个掩码
    segmented = cv2.bitwise_and(low_mask, high_mask)

    return gray_image, segmented


# 执行分割
gray_img, result = double_threshold_segmentation("baboon.png")

# 可视化结果
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(gray_img, cmap="gray")
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(result, cmap="gray")
axes[1].set_title("Double Threshold Segmentation")
axes[1].axis("off")

plt.tight_layout()
plt.show()
