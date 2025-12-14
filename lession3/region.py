from skimage import io, color
from skimage.segmentation import flood
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


def region_growing_segmentation(
    image_path: str, seed_point: Tuple[int, int] = (150, 200), tolerance: float = 0.2
) -> Tuple[np.ndarray, np.ndarray]:
    """
    区域生长法分割

    参数:
        image_path: 输入图像路径
        seed_point: 种子点坐标
        tolerance: 灰度相似性容差

    返回:
        original_image: 原始图像
        segmented: 分割结果
    """
    # 读取图像
    image = io.imread(image_path)

    # 处理4通道图片（RGBA）
    if image.shape[-1] == 4:
        # 将RGBA转换为RGB（去除alpha通道）
        image = color.rgba2rgb(image)

    # 转换为灰度图
    gray_image = color.rgb2gray(image)

    # 应用区域生长
    segmented = flood(gray_image, seed_point, tolerance=tolerance)

    return image, segmented


# 执行分割
original_img, segmented_img = region_growing_segmentation("balloon.png")

# 可视化结果
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(original_img)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(segmented_img, cmap="gray")
axes[1].set_title("Region Growing Segmentation")
axes[1].axis("off")

plt.tight_layout()
plt.show()
