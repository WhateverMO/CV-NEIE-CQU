import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


def watershed_segmentation(
    image_path: str,
) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
    """
    分水岭算法分割

    参数:
        image_path: 输入图像路径

    返回:
        final_result: 最终分割结果
        process_steps: 处理步骤中间结果
    """
    # 读取图像
    image = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用Otsu阈值
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 形态学开操作去除噪声
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 距离变换找到前景区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # 确定未知区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 标记连接区域
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 应用分水岭算法
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]  # 标记边界为红色

    final_result = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 保存处理步骤
    process_steps = [
        ("Original", original_rgb),
        ("Thresholded", thresh),
        ("Distance Transform", dist_transform),
        ("Segmented Result", final_result),
    ]

    return final_result, process_steps


# 执行分割
result, steps = watershed_segmentation("orange.png")

# 可视化处理过程
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, (title, img) in enumerate(steps):
    row, col = divmod(idx, 2)
    ax = axes[row, col]

    if title == "Distance Transform":
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)

    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
