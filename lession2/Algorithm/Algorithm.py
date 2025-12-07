import cv2
import numpy as np
from typing import Optional, Tuple, List, Any, Protocol
import einops
import numpy.typing as npt
import time


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


# ========== Einops算法实现 ==========
class EinopsAlgorithm:
    """使用Einops张量操作库实现的图像算法"""

    def __init__(self) -> None:
        self.name = "Einops"

    def rotate_90(self, image: npt.NDArray) -> npt.NDArray:
        """顺时针旋转90度 - 修复旋转方向"""
        if image is None or image.size == 0:
            return np.array([])

        # 修复旋转方向：先垂直翻转再转置
        # 正确的顺时针90度旋转
        flipped = image[::-1, :, :]  # 垂直翻转
        rotated = einops.rearrange(flipped, "h w c -> w h c")  # 转置
        return rotated

    def scale_image(self, image: npt.NDArray, scale_factor: float) -> npt.NDArray:
        """缩放图像 - 纯Einops实现"""
        if image is None or image.size == 0:
            return np.array([])

        height, width = image.shape[:2]
        new_width: int = int(width * scale_factor)
        new_height: int = int(height * scale_factor)

        if scale_factor > 1.0:
            # 放大：使用repeat重复像素
            scale_int = max(1, int(scale_factor))
            scaled = einops.repeat(
                image, "h w c -> (h h2) (w w2) c", h2=scale_int, w2=scale_int
            )
            return scaled[:new_height, :new_width]
        else:
            # 缩小：使用最近邻插值
            scaled = np.zeros((new_height, new_width, 3), dtype=np.uint8)
            for i in range(new_height):
                for j in range(new_width):
                    orig_i = int(i / scale_factor)
                    orig_j = int(j / scale_factor)
                    if orig_i < height and orig_j < width:
                        scaled[i, j] = image[orig_i, orig_j]
            return scaled

    def invert_colors(self, image: npt.NDArray) -> npt.NDArray:
        """数值计算反色 - 纯Einops实现"""
        if image is None or image.size == 0:
            return np.array([])
        return 255 - image

    def grayscale(self, image: npt.NDArray) -> npt.NDArray:
        """灰度化处理 - 纯Einops实现"""
        if image is None or image.size == 0:
            return np.array([])

        # 使用加权平均：Y = 0.299R + 0.587G + 0.114B
        weights = np.array([0.114, 0.587, 0.299])  # BGR顺序
        image_float = image.astype(np.float32)
        gray = einops.einsum(image_float, weights, "h w c, c -> h w")
        gray_uint8 = np.clip(gray, 0, 255).astype(np.uint8)
        return einops.repeat(gray_uint8, "h w -> h w c", c=3)


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


# ========== 算法管理器 ==========
class AlgorithmManager:
    """管理三种算法实现"""

    def __init__(self) -> None:
        self.algorithms: List[ImageAlgorithm] = [
            OpenCVAlgorithm(),
            EinopsAlgorithm(),
            PurePythonAlgorithm(),
        ]
        self.algorithm_names: List[str] = [alg.name for alg in self.algorithms]

    def process_rotation(self, image: npt.NDArray, log_func) -> List[npt.NDArray]:
        """使用三种算法进行旋转处理"""
        results: List[npt.NDArray] = []
        log_func("开始旋转处理...")

        for algorithm in self.algorithms:
            log_func(f"\n{algorithm.name} 算法旋转处理:")
            start_time = time.time()

            try:
                result = algorithm.rotate_90(image)
                elapsed = time.time() - start_time

                log_func(f"  ✓ 处理成功")
                log_func(f"  输入尺寸: {image.shape}")
                log_func(f"  输出尺寸: {result.shape}")
                log_func(f"  处理耗时: {elapsed:.4f}秒")

                results.append(result)
            except Exception as e:
                log_func(f"  ✗ 处理失败: {e}")
                results.append(np.array([]))

        return results

    def process_scaling(
        self, image: npt.NDArray, scale_factor: float, log_func
    ) -> List[npt.NDArray]:
        """使用三种算法进行缩放处理"""
        results: List[npt.NDArray] = []
        log_func(f"\n开始缩放处理，缩放因子: {scale_factor:.2f}")

        for algorithm in self.algorithms:
            log_func(f"\n{algorithm.name} 算法缩放处理:")
            start_time = time.time()

            try:
                result = algorithm.scale_image(image, scale_factor)
                elapsed = time.time() - start_time

                log_func(f"  ✓ 处理成功")
                log_func(f"  输入尺寸: {image.shape}")
                log_func(f"  输出尺寸: {result.shape}")
                log_func(f"  缩放比例: {scale_factor:.2f}")
                log_func(f"  处理耗时: {elapsed:.4f}秒")

                results.append(result)
            except Exception as e:
                log_func(f"  ✗ 处理失败: {e}")
                results.append(np.array([]))

        return results

    def process_inversion(self, image: npt.NDArray, log_func) -> List[npt.NDArray]:
        """使用三种算法进行反色处理"""
        results: List[npt.NDArray] = []
        log_func("开始反色处理...")

        for algorithm in self.algorithms:
            log_func(f"\n{algorithm.name} 算法反色处理:")
            start_time = time.time()

            try:
                result = algorithm.invert_colors(image)
                elapsed = time.time() - start_time

                log_func(f"  ✓ 处理成功")
                log_func(f"  输入尺寸: {image.shape}")
                log_func(f"  输出尺寸: {result.shape}")
                log_func(f"  处理耗时: {elapsed:.4f}秒")

                results.append(result)
            except Exception as e:
                log_func(f"  ✗ 处理失败: {e}")
                results.append(np.array([]))

        return results

    def process_grayscale(self, image: npt.NDArray, log_func) -> List[npt.NDArray]:
        """使用三种算法进行灰度化处理"""
        results: List[npt.NDArray] = []
        log_func("开始灰度化处理...")

        for algorithm in self.algorithms:
            log_func(f"\n{algorithm.name} 算法灰度化处理:")
            start_time = time.time()

            try:
                result = algorithm.grayscale(image)
                elapsed = time.time() - start_time

                log_func(f"  ✓ 处理成功")
                log_func(f"  输入尺寸: {image.shape}")
                log_func(f"  输出尺寸: {result.shape}")
                log_func(f"  处理耗时: {elapsed:.4f}秒")

                results.append(result)
            except Exception as e:
                log_func(f"  ✗ 处理失败: {e}")
                results.append(np.array([]))

        return results
