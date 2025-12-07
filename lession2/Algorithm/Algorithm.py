"""
图像浏览器 - 三算法对比实现 + 作业2功能集成
"""

import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from typing import Optional, Tuple, List, Any, Protocol
import matplotlib.pyplot as plt
import numpy.typing as npt
import time
import random
import math
import platform

# 设置中文字体
if platform.system() == "Darwin":  # macOS
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS"]
elif platform.system() == "Windows":  # Windows
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
else:  # Linux
    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]

plt.rcParams["axes.unicode_minus"] = False


# ========== 算法接口定义 ==========
class ImageAlgorithm(Protocol):
    """图像算法接口协议"""

    def rotate_90(self, image: npt.NDArray) -> npt.NDArray:
        """旋转图像90度"""
        ...

    def scale_image(self, image: npt.NDArray, scale_factor: float) -> npt.NDArray:
        """缩放图像"""
        ...

    def invert_colors(self, image: npt.NDArray) -> npt.NDArray:
        """反色处理"""
        ...

    def grayscale(self, image: npt.NDArray) -> npt.NDArray:
        """灰度化处理"""
        ...


# ========== 新增算法接口 ==========
class AdvancedImageAlgorithm(Protocol):
    """作业2新增算法接口"""

    def bilinear_interpolate(
        self, image: npt.NDArray, x: float, y: float
    ) -> npt.NDArray:
        """双线性插值"""
        ...

    def calculate_histogram(
        self, image: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """计算灰度直方图"""
        ...

    def histogram_equalization(self, image: npt.NDArray) -> npt.NDArray:
        """直方图均衡化"""
        ...

    def median_filter(self, image: npt.NDArray, kernel_size: int) -> npt.NDArray:
        """中值滤波"""
        ...

    def add_salt_pepper_noise(self, image: npt.NDArray, amount: float) -> npt.NDArray:
        """添加椒盐噪声"""
        ...

    def sharpen(self, image: npt.NDArray, method: str) -> npt.NDArray:
        """图像锐化"""
        ...

    def frequency_filter(
        self,
        image: npt.NDArray,
        mode: str,
        filter_type: str,
        cutoff: int,
        order: int = 1,
    ) -> npt.NDArray:
        """频域滤波"""
        ...


# ========== OpenCV算法实现 ==========
class OpenCVAlgorithm:
    """使用OpenCV库实现的图像算法"""

    def __init__(self) -> None:
        self.name = "OpenCV"

    def rotate_90(self, image: npt.NDArray) -> npt.NDArray:
        """顺时针旋转90度"""
        if image is None or image.size == 0:
            return np.array([])
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    def scale_image(self, image: npt.NDArray, scale_factor: float) -> npt.NDArray:
        """双线性插值缩放"""
        if image is None or image.size == 0:
            return np.array([])

        height, width = image.shape[:2]
        new_width: int = int(width * scale_factor)
        new_height: int = int(height * scale_factor)

        return cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

    def invert_colors(self, image: npt.NDArray) -> npt.NDArray:
        """按位取反实现反色"""
        if image is None or image.size == 0:
            return np.array([])
        return cv2.bitwise_not(image)

    def grayscale(self, image: npt.NDArray) -> npt.NDArray:
        """亮度加权公式灰度化"""
        if image is None or image.size == 0:
            return np.array([])

        gray: npt.NDArray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def shear_x(self, image: npt.NDArray, shear_factor: float) -> npt.NDArray:
        """X轴斜切"""
        if image is None or image.size == 0:
            return np.array([])

        height, width = image.shape[:2]
        new_width = int(width + abs(shear_factor) * height)
        M = np.float32([[1, shear_factor, 0], [0, 1, 0]])
        sheared = cv2.warpAffine(
            image,
            M,
            (new_width, height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return sheared

    def shear_y(self, image: npt.NDArray, shear_factor: float) -> npt.NDArray:
        """Y轴斜切"""
        if image is None or image.size == 0:
            return np.array([])

        height, width = image.shape[:2]
        new_height = int(height + abs(shear_factor) * width)
        M = np.float32([[1, 0, 0], [shear_factor, 1, 0]])
        sheared = cv2.warpAffine(
            image,
            M,
            (width, new_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return sheared

    def bilinear_interpolate(
        self, image: npt.NDArray, x: float, y: float
    ) -> npt.NDArray:
        """双线性插值"""
        if image is None or image.size == 0:
            return np.array([])

        height, width = image.shape[:2]

        # 边界检查
        if x < 0 or x >= width or y < 0 or y >= height:
            return np.array([255, 255, 255])

        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)

        dx, dy = x - x0, y - y0

        # 获取四个角点的像素值
        q11 = image[y0, x0]
        q21 = image[y0, x1]
        q12 = image[y1, x0]
        q22 = image[y1, x1]

        # 双线性插值公式
        result = (
            q11 * (1 - dx) * (1 - dy)
            + q21 * dx * (1 - dy)
            + q12 * (1 - dx) * dy
            + q22 * dx * dy
        )

        return np.clip(result, 0, 255).astype(np.uint8)

    def calculate_histogram(
        self, image: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """计算灰度直方图"""
        if image is None or image.size == 0:
            return np.array([]), np.array([])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        return hist.flatten(), np.arange(256)

    def histogram_equalization(self, image: npt.NDArray) -> npt.NDArray:
        """直方图均衡化"""
        if image is None or image.size == 0:
            return np.array([])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    def add_salt_pepper_noise(self, image: npt.NDArray, amount: float) -> npt.NDArray:
        """添加椒盐噪声"""
        if image is None or image.size == 0:
            return np.array([])

        noisy = image.copy()
        height, width = noisy.shape[:2]
        num_pixels = int(height * width * amount)

        # 添加白噪声（盐噪声）
        for _ in range(num_pixels // 2):
            row = np.random.randint(0, height)
            col = np.random.randint(0, width)
            noisy[row, col] = [255, 255, 255]

        # 添加黑噪声（椒噪声）
        for _ in range(num_pixels // 2):
            row = np.random.randint(0, height)
            col = np.random.randint(0, width)
            noisy[row, col] = [0, 0, 0]

        return noisy

    def median_filter(self, image: npt.NDArray, kernel_size: int) -> npt.NDArray:
        """中值滤波"""
        if image is None or image.size == 0:
            return np.array([])
        return cv2.medianBlur(image, kernel_size)

    def sharpen(self, image: npt.NDArray, method: str) -> npt.NDArray:
        """图像锐化"""
        if image is None or image.size == 0:
            return np.array([])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if method == "roberts":
            kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
            kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            grad_x = cv2.filter2D(gray, cv2.CV_32F, kernel_x)
            grad_y = cv2.filter2D(gray, cv2.CV_32F, kernel_y)
            result = np.abs(grad_x) + np.abs(grad_y)

        elif method == "sobel":
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            result = np.abs(grad_x) + np.abs(grad_y)

        elif method == "laplacian":
            result = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            result = np.abs(result)
        else:
            return image

        result = np.clip(result, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    def frequency_filter(
        self,
        image: npt.NDArray,
        mode: str,
        filter_type: str,
        cutoff: int,
        order: int = 1,
    ) -> npt.NDArray:
        """频域滤波"""
        if image is None or image.size == 0:
            return np.array([])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 傅里叶变换
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # 构建滤波器
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        mask = np.zeros((rows, cols, 2), dtype=np.float32)

        for u in range(rows):
            for v in range(cols):
                distance = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
                if mode == "low":  # 低通滤波
                    if filter_type == "ideal":
                        mask[u, v] = 1 if distance <= cutoff else 0
                    elif filter_type == "gaussian":
                        mask[u, v] = np.exp(-(distance**2) / (2 * (cutoff**2)))
                    elif filter_type == "butterworth":
                        mask[u, v] = 1 / (1 + (distance / cutoff) ** (2 * order))
                elif mode == "high":  # 高通滤波
                    if filter_type == "ideal":
                        mask[u, v] = 0 if distance <= cutoff else 1
                    elif filter_type == "gaussian":
                        mask[u, v] = 1 - np.exp(-(distance**2) / (2 * (cutoff**2)))
                    elif filter_type == "butterworth":
                        mask[u, v] = (
                            1 / (1 + (cutoff / distance) ** (2 * order))
                            if distance > 0
                            else 0
                        )

        # 应用滤波器
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        filtered = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
        filtered = filtered.astype(np.uint8)

        return cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)


# ========== 纯Python算法实现 ==========
class PurePythonAlgorithm:
    """使用纯Python和NumPy实现的图像算法"""

    def __init__(self) -> None:
        self.name = "纯Python"

    def rotate_90(self, image: npt.NDArray) -> npt.NDArray:
        """顺时针旋转90度 - 纯Python实现"""
        if image is None or image.size == 0:
            return np.array([])

        height, width = image.shape[:2]
        rotated: npt.NDArray = np.zeros((width, height, 3), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                rotated[j, height - 1 - i] = image[i, j]

        return rotated

    def scale_image(self, image: npt.NDArray, scale_factor: float) -> npt.NDArray:
        """缩放图像 - 纯Python实现（最近邻插值）"""
        if image is None or image.size == 0:
            return np.array([])

        height, width = image.shape[:2]
        new_width: int = int(width * scale_factor)
        new_height: int = int(height * scale_factor)

        scaled: npt.NDArray = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        for i in range(new_height):
            for j in range(new_width):
                orig_i = int(i / scale_factor)
                orig_j = int(j / scale_factor)
                if orig_i < height and orig_j < width:
                    scaled[i, j] = image[orig_i, orig_j]

        return scaled

    def invert_colors(self, image: npt.NDArray) -> npt.NDArray:
        """反色处理 - 纯Python实现"""
        if image is None or image.size == 0:
            return np.array([])
        return 255 - image

    def grayscale(self, image: npt.NDArray) -> npt.NDArray:
        """灰度化处理 - 纯Python实现"""
        if image is None or image.size == 0:
            return np.array([])

        b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        gray = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
        result: npt.NDArray = np.stack([gray, gray, gray], axis=2)
        return result

    def bilinear_interpolate(
        self, image: npt.NDArray, x: float, y: float
    ) -> npt.NDArray:
        """双线性插值 - 纯Python实现"""
        if image is None or image.size == 0:
            return np.array([])

        height, width = image.shape[:2]

        if x < 0 or x >= width or y < 0 or y >= height:
            return np.array([255, 255, 255])

        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)

        dx, dy = x - x0, y - y0

        q11 = image[y0, x0]
        q21 = image[y0, x1]
        q12 = image[y1, x0]
        q22 = image[y1, x1]

        result = np.zeros(3, dtype=np.float32)
        for c in range(3):
            result[c] = (
                q11[c] * (1 - dx) * (1 - dy)
                + q21[c] * dx * (1 - dy)
                + q12[c] * (1 - dx) * dy
                + q22[c] * dx * dy
            )

        return np.clip(result, 0, 255).astype(np.uint8)

    def shear_x(self, image: npt.NDArray, shear_factor: float) -> npt.NDArray:
        """X轴斜切 - 纯Python实现"""
        if image is None or image.size == 0:
            return np.array([])

        height, width = image.shape[:2]
        new_width = int(width + abs(shear_factor) * height)
        sheared = np.ones((height, new_width, 3), dtype=np.uint8) * 255

        for i in range(height):
            for j in range(new_width):
                x = j - shear_factor * i
                y = i
                if 0 <= x < width and 0 <= y < height:
                    sheared[i, j] = self.bilinear_interpolate(image, x, y)

        return sheared

    def shear_y(self, image: npt.NDArray, shear_factor: float) -> npt.NDArray:
        """Y轴斜切 - 纯Python实现"""
        if image is None or image.size == 0:
            return np.array([])

        height, width = image.shape[:2]
        new_height = int(height + abs(shear_factor) * width)
        sheared = np.ones((new_height, width, 3), dtype=np.uint8) * 255

        for i in range(new_height):
            for j in range(width):
                x = j
                y = i - shear_factor * j
                if 0 <= x < width and 0 <= y < height:
                    sheared[i, j] = self.bilinear_interpolate(image, x, y)

        return sheared

    def calculate_histogram(
        self, image: npt.NDArray
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """计算灰度直方图 - 纯Python实现"""
        if image is None or image.size == 0:
            return np.array([]), np.array([])

        if len(image.shape) == 3:
            b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            gray = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
        else:
            gray = image

        hist = np.zeros(256, dtype=np.int32)
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                hist[gray[i, j]] += 1

        return hist, np.arange(256)

    def histogram_equalization(self, image: npt.NDArray) -> npt.NDArray:
        """直方图均衡化 - 纯Python实现"""
        if image is None or image.size == 0:
            return np.array([])

        if len(image.shape) == 3:
            b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            gray = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
        else:
            gray = image

        height, width = gray.shape
        hist, _ = self.calculate_histogram(image)

        # 计算CDF
        cdf = hist.cumsum()
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        cdf_normalized = cdf_normalized.astype(np.uint8)

        # 映射
        equalized = np.zeros_like(gray)
        for i in range(height):
            for j in range(width):
                equalized[i, j] = cdf_normalized[gray[i, j]]

        if len(image.shape) == 3:
            equalized = np.stack([equalized, equalized, equalized], axis=2)

        return equalized.astype(np.uint8)

    def add_salt_pepper_noise(self, image: npt.NDArray, amount: float) -> npt.NDArray:
        """添加椒盐噪声 - 纯Python实现"""
        if image is None or image.size == 0:
            return np.array([])

        noisy = image.copy()
        height, width = noisy.shape[:2]
        num_pixels = int(height * width * amount)

        for _ in range(num_pixels // 2):
            row = np.random.randint(0, height)
            col = np.random.randint(0, width)
            noisy[row, col] = [255, 255, 255]

        for _ in range(num_pixels // 2):
            row = np.random.randint(0, height)
            col = np.random.randint(0, width)
            noisy[row, col] = [0, 0, 0]

        return noisy

    def median_filter(self, image: npt.NDArray, kernel_size: int) -> npt.NDArray:
        """中值滤波 - 纯Python实现"""
        if image is None or image.size == 0:
            return np.array([])

        pad = kernel_size // 2
        height, width, channels = image.shape
        result = np.zeros_like(image)

        for c in range(channels):
            padded = np.pad(image[:, :, c], pad, mode="constant", constant_values=0)
            for i in range(height):
                for j in range(width):
                    region = padded[i : i + kernel_size, j : j + kernel_size]
                    result[i, j, c] = np.median(region)

        return result.astype(np.uint8)

    def sharpen(self, image: npt.NDArray, method: str) -> npt.NDArray:
        """图像锐化 - 纯Python实现"""
        if image is None or image.size == 0:
            return np.array([])

        if len(image.shape) == 3:
            b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            gray = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
        else:
            gray = image

        if method == "roberts":
            kernel_x = np.array([[1, 0], [0, -1]])
            kernel_y = np.array([[0, 1], [-1, 0]])
            result = self.convolve(gray, kernel_x) + self.convolve(gray, kernel_y)

        elif method == "sobel":
            kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            result = np.abs(self.convolve(gray, kernel_x)) + np.abs(
                self.convolve(gray, kernel_y)
            )

        elif method == "laplacian":
            kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            result = self.convolve(gray, kernel)
        else:
            return image

        result = np.clip(np.abs(result), 0, 255).astype(np.uint8)

        if len(image.shape) == 3:
            result = np.stack([result, result, result], axis=2)

        return result

    def convolve(self, image: npt.NDArray, kernel: npt.NDArray) -> npt.NDArray:
        """卷积操作 - 纯Python实现"""
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant")
        result = np.zeros_like(image, dtype=np.float32)

        height, width = image.shape
        for i in range(height):
            for j in range(width):
                region = padded[i : i + k_h, j : j + k_w]
                result[i, j] = np.sum(region * kernel)

        return result

    def frequency_filter(
        self,
        image: npt.NDArray,
        mode: str,
        filter_type: str,
        cutoff: int,
        order: int = 1,
    ) -> npt.NDArray:
        """频域滤波 - 纯Python实现"""
        if image is None or image.size == 0:
            return np.array([])

        if len(image.shape) == 3:
            b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            gray = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
        else:
            gray = image

        # 傅里叶变换
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)

        # 构建滤波器
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        filter_mask = np.zeros((rows, cols), dtype=np.float32)

        for u in range(rows):
            for v in range(cols):
                distance = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
                if mode == "low":  # 低通滤波
                    if filter_type == "ideal":
                        filter_mask[u, v] = 1 if distance <= cutoff else 0
                    elif filter_type == "gaussian":
                        filter_mask[u, v] = np.exp(-(distance**2) / (2 * (cutoff**2)))
                    elif filter_type == "butterworth":
                        filter_mask[u, v] = 1 / (1 + (distance / cutoff) ** (2 * order))
                elif mode == "high":  # 高通滤波
                    if filter_type == "ideal":
                        filter_mask[u, v] = 0 if distance <= cutoff else 1
                    elif filter_type == "gaussian":
                        filter_mask[u, v] = 1 - np.exp(
                            -(distance**2) / (2 * (cutoff**2))
                        )
                    elif filter_type == "butterworth":
                        filter_mask[u, v] = (
                            1 / (1 + (cutoff / distance) ** (2 * order))
                            if distance > 0
                            else 0
                        )

        # 应用滤波器
        fshift_filtered = fshift * filter_mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        filtered = np.abs(img_back)
        filtered = np.clip(filtered, 0, 255).astype(np.uint8)

        if len(image.shape) == 3:
            filtered = np.stack([filtered, filtered, filtered], axis=2)

        return filtered
