import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def ssd_matching(image, template):
    """SSD: 平方差之和 (越小越匹配)"""
    image_h, image_w = image.shape
    template_h, template_w = template.shape
    result = np.zeros((image_h - template_h + 1, image_w - template_w + 1))

    # 简单优化：使用OpenCV的SQDIFF进行加速计算，原理同手写循环
    # 如果老师要求必须手写循环，请替换回之前的双重for循环版本
    # 这里为了处理本地大图不卡顿，使用了cv2.matchTemplate实现原理
    result = cv2.matchTemplate(image, template, cv2.TM_SQDIFF)

    return result


def ncc_matching(image, template):
    """NCC: 归一化互相关 (越接近1越匹配)"""
    # 同样使用OpenCV内置函数实现标准NCC算法，保证速度
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    return result


def run_local_experiment():
    # 1. 读取本地图片
    # 注意：这里读取为灰度图 (0)
    img_path = "images/search.jpg"
    mpl_path = "images/template.jpg"

    if not os.path.exists(img_path) or not os.path.exists(mpl_path):
        print(f"错误：找不到图片文件！请确保 {img_path} 和 {mpl_path} 存在。")
        return

    img = cv2.imread(img_path, 0)
    template = cv2.imread(mpl_path, 0)

    print(f"原图尺寸: {img.shape}, 模板尺寸: {template.shape}")

    # 2. 执行 SSD 匹配
    print("正在进行 SSD 匹配...")
    ssd_res = ssd_matching(img, template)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ssd_res)
    top_left_ssd = min_loc  # SSD 找最小值

    # 3. 执行 NCC 匹配
    print("正在进行 NCC 匹配...")
    ncc_res = ncc_matching(img, template)
    min_val_ncc, max_val_ncc, min_loc_ncc, max_loc_ncc = cv2.minMaxLoc(ncc_res)
    top_left_ncc = max_loc_ncc  # NCC 找最大值

    # 4. 绘图展示
    h, w = template.shape
    plt.figure(figsize=(12, 8))

    # --- SSD 结果 ---
    plt.subplot(2, 2, 1)
    plt.imshow(ssd_res, cmap="jet")
    plt.title("SSD Result Map (Darker is better)")
    plt.colorbar()

    plt.subplot(2, 2, 2)
    img_ssd_show = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(
        img_ssd_show,
        top_left_ssd,
        (top_left_ssd[0] + w, top_left_ssd[1] + h),
        (0, 255, 0),
        3,
    )
    plt.imshow(img_ssd_show)
    plt.title("SSD Detection")
    plt.axis("off")

    # --- NCC 结果 ---
    plt.subplot(2, 2, 3)
    plt.imshow(ncc_res, cmap="jet")
    plt.title("NCC Result Map (Brighter is better)")
    plt.colorbar()

    plt.subplot(2, 2, 4)
    img_ncc_show = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    cv2.rectangle(
        img_ncc_show,
        top_left_ncc,
        (top_left_ncc[0] + w, top_left_ncc[1] + h),
        (0, 0, 255),
        3,
    )
    plt.imshow(img_ncc_show)
    plt.title("NCC Detection")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_local_experiment()
