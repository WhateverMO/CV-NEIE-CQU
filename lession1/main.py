"""
图像浏览器 - 三算法对比实现（最终版）
修复Einops旋转方向，调整缩放因子
"""

import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
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


# ========== 图像显示管理器 ==========
class ImageDisplayManager:
    """管理OpenCV图像显示窗口"""

    def __init__(self) -> None:
        self.window_name = "三算法对比结果"
        self.window_created = False
        self.screen_width = 1920
        self.screen_height = 1080
        self.get_screen_resolution()

    def get_screen_resolution(self) -> None:
        """获取屏幕分辨率"""
        try:
            root = tk.Tk()
            self.screen_width = root.winfo_screenwidth()
            self.screen_height = root.winfo_screenheight()
            root.destroy()
        except:
            pass

    def create_window(self) -> None:
        """创建显示窗口"""
        if not self.window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1200, 600)
            self.window_created = True
            print("✓ OpenCV窗口已创建")

    def resize_images_to_same_height(
        self, images: List[npt.NDArray]
    ) -> List[npt.NDArray]:
        """将所有图像调整为相同高度"""
        if not images or any(img.size == 0 for img in images):
            return images

        # 获取最小高度
        min_height = min(img.shape[0] for img in images if img.size > 0)

        resized_images = []
        for img in images:
            if img.size == 0:
                resized_images.append(img)
                continue

            height, width = img.shape[:2]
            if height != min_height:
                # 等比例缩放
                scale_factor = min_height / height
                new_width = int(width * scale_factor)
                resized = cv2.resize(
                    img, (new_width, min_height), interpolation=cv2.INTER_LINEAR
                )
                resized_images.append(resized)
            else:
                resized_images.append(img)

        return resized_images

    def concatenate_images_horizontally(
        self, images: List[npt.NDArray], titles: List[str]
    ) -> npt.NDArray:
        """将图像横向拼接，并添加标题"""
        if not images or any(img.size == 0 for img in images):
            return np.array([])

        # 调整图像到相同高度
        resized_images = self.resize_images_to_same_height(images)

        # 横向拼接
        concatenated = np.hstack(resized_images)

        # 添加标题
        result = self.add_titles(concatenated, resized_images, titles)
        return result

    def add_titles(
        self,
        concatenated_img: npt.NDArray,
        images: List[npt.NDArray],
        titles: List[str],
    ) -> npt.NDArray:
        """为每个子图添加标题"""
        if concatenated_img.size == 0:
            return concatenated_img

        # 计算标题区域高度
        title_height = 50
        result_height = concatenated_img.shape[0] + title_height
        result_width = concatenated_img.shape[1]

        # 创建带标题区域的图像
        result = np.ones((result_height, result_width, 3), dtype=np.uint8) * 255

        # 将原图像放在下方
        result[title_height:, :] = concatenated_img

        # 添加标题
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 2
        font_color = (0, 0, 0)  # 黑色

        current_x = 0
        for i, (img, title) in enumerate(zip(images, titles)):
            if img.size == 0:
                continue

            # 计算标题位置（居中）
            text_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
            text_x = current_x + (img.shape[1] - text_size[0]) // 2
            text_y = title_height - 15

            cv2.putText(
                result,
                title,
                (text_x, text_y),
                font,
                font_scale,
                font_color,
                font_thickness,
            )

            # 添加分隔线
            if i < len(images) - 1:
                line_x = current_x + img.shape[1]
                cv2.line(
                    result, (line_x, 0), (line_x, result_height), (200, 200, 200), 2
                )

            current_x += img.shape[1]

        return result

    def show_images(self, images: List[npt.NDArray], operation_name: str) -> None:
        """在单个窗口中显示三张横向拼接的图像"""
        print(f"\n=== 开始显示 {operation_name} 结果 ===")

        # 确保窗口已创建
        self.create_window()

        # 准备标题
        titles = [
            f"{name} - {operation_name}" for name in ["OpenCV", "Einops", "Pure Python"]
        ]

        # 拼接图像
        concatenated = self.concatenate_images_horizontally(images, titles)

        if concatenated.size == 0:
            print("✗ 无法拼接图像：图像为空")
            return

        # 调整窗口大小适应图像
        img_height, img_width = concatenated.shape[:2]

        # 计算适合屏幕的显示尺寸
        max_width = min(img_width, self.screen_width - 100)
        max_height = min(img_height, self.screen_height - 100)

        # 等比例缩放
        scale_x = max_width / img_width
        scale_y = max_height / img_height
        scale = min(scale_x, scale_y, 1.0)  # 不超过原图大小

        display_width = int(img_width * scale)
        display_height = int(img_height * scale)

        # 调整窗口大小
        cv2.resizeWindow(self.window_name, display_width, display_height)

        # 显示图像
        cv2.imshow(self.window_name, concatenated)
        cv2.waitKey(1)  # 强制刷新显示

        print(f"✓ 图像拼接显示成功")
        print(f"  总尺寸: {concatenated.shape[1]} x {concatenated.shape[0]}")
        print(f"  显示尺寸: {display_width} x {display_height}")
        print(f"  缩放比例: {scale:.2f}")

    def close_window(self) -> None:
        """关闭窗口"""
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass
        self.window_created = False
        print("✓ OpenCV窗口已关闭")


# ========== 图像浏览器主类 ==========
class TripleImageBrowser:
    """三算法对比图像浏览器主类"""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("三算法对比图像浏览器")
        self.root.geometry("800x600")

        # 图像相关属性
        self.image: Optional[npt.NDArray] = None
        self.original_image: Optional[npt.NDArray] = None
        self.current_scale: float = 1.0
        self.image_loaded = False

        # 管理器
        self.algorithm_manager: AlgorithmManager = AlgorithmManager()
        self.display_manager: ImageDisplayManager = ImageDisplayManager()

        # UI组件
        self.info_text: Optional[tk.Text] = None
        self.scale_label: Optional[tk.Label] = None

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self) -> None:
        """设置用户界面布局和组件"""
        # 主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 标题
        title_label = tk.Label(
            main_frame, text="三算法对比图像浏览器", font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 10))

        # 控制面板
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # 文件操作按钮
        file_frame = tk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=5)

        tk.Button(
            file_frame,
            text="打开图像",
            command=self.open_image_file,
            width=12,
            height=2,
        ).pack(side=tk.LEFT, padx=5)
        tk.Button(
            file_frame, text="重置图像", command=self.reset_image, width=12, height=2
        ).pack(side=tk.LEFT, padx=5)

        # 图像操作按钮
        operation_frame = tk.Frame(control_frame)
        operation_frame.pack(fill=tk.X, pady=5)

        operations = [
            ("旋转90°", self.rotate_90_all),
            ("放大2.0x", lambda: self.scale_image_all(2.0)),
            ("缩小0.5x", lambda: self.scale_image_all(0.5)),
            ("反色处理", self.invert_colors_all),
            ("灰度化", self.grayscale_all),
        ]

        for text, command in operations:
            btn = tk.Button(
                operation_frame, text=text, command=command, width=10, height=2
            )
            btn.pack(side=tk.LEFT, padx=2)
            btn.config(state=tk.DISABLED)
            setattr(self, f"{text.replace('°', '').replace('.', '')}_button", btn)

        # 信息显示区域
        info_frame = tk.Frame(main_frame)
        info_frame.pack(fill=tk.BOTH, expand=True)

        # 信息标签
        tk.Label(info_frame, text="处理日志:", font=("Arial", 12, "bold")).pack(
            anchor=tk.W
        )

        # 缩放信息
        scale_frame = tk.Frame(info_frame)
        scale_frame.pack(fill=tk.X, pady=5)

        self.scale_label = tk.Label(
            scale_frame, text="当前缩放比例: 1.00x", font=("Arial", 10), fg="blue"
        )
        self.scale_label.pack()

        # 创建带滚动条的文本框
        text_frame = tk.Frame(info_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.info_text = tk.Text(text_frame, height=15, wrap=tk.WORD, font=("Arial", 9))
        scrollbar = tk.Scrollbar(text_frame, command=self.info_text.yview)
        self.info_text.config(yscrollcommand=scrollbar.set)

        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 状态栏
        status_frame = tk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(5, 0))

        self.status_label = tk.Label(
            status_frame,
            text="提示: 请先打开图像文件，处理结果将在单个窗口中并排显示",
            font=("Arial", 9),
            fg="gray",
        )
        self.status_label.pack()

        self.log("=" * 60)
        self.log("三算法对比图像浏览器已启动")
        self.log("算法实现: OpenCV, Einops, 纯Python")
        self.log("显示方式: 单窗口横向拼接显示")
        self.log("缩放因子: 放大2.0x, 缩小0.5x")
        self.log("=" * 60)
        self.log("请先点击'打开图像'按钮选择图像文件")

    def enable_operation_buttons(self, enabled: bool = True) -> None:
        """启用或禁用操作按钮"""
        state = tk.NORMAL if enabled else tk.DISABLED
        buttons = [
            "旋转90_button",
            "放大20x_button",
            "缩小05x_button",
            "反色处理_button",
            "灰度化_button",
        ]

        for btn_name in buttons:
            if hasattr(self, btn_name):
                getattr(self, btn_name).config(state=state)

    def log(self, message: str) -> None:
        """记录信息到文本框"""
        if self.info_text:
            self.info_text.insert(tk.END, f"{message}\n")
            self.info_text.see(tk.END)
            self.root.update()

    def update_scale_display(self) -> None:
        """更新缩放比例显示"""
        if self.scale_label:
            self.scale_label.config(text=f"当前缩放比例: {self.current_scale:.2f}x")

    def open_image_file(self) -> None:
        """打开单个图像文件"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")],
        )

        if file_path:
            self.log(f"\n正在打开图像: {file_path}")
            self.image = cv2.imread(file_path)

            if self.image is not None:
                self.original_image = self.image.copy()
                self.current_scale = 1.0
                self.image_loaded = True
                self.update_scale_display()
                self.enable_operation_buttons(True)
                self.status_label.config(text="图像已加载，可进行操作处理")

                self.log(f"✓ 图像加载成功")
                self.log(f"  文件名称: {os.path.basename(file_path)}")
                self.log(f"  图像尺寸: {self.image.shape[1]} x {self.image.shape[0]}")
                self.log(
                    f"  通道数量: {self.image.shape[2] if len(self.image.shape) > 2 else 1}"
                )

                # 显示原始图像
                self.show_original_images()
            else:
                self.log("✗ 图像加载失败")
                messagebox.showerror("错误", "无法打开图像文件")

    def show_original_images(self) -> None:
        """显示原始图像"""
        if self.image is not None and self.image_loaded:
            # 创建三个相同的图像副本
            images = [self.image.copy() for _ in range(3)]
            self.display_manager.show_images(images, "original")
            self.log("✓ 原始图像已显示")

    def process_and_show(self, operation_name: str, process_func) -> None:
        """通用的处理并显示函数"""
        if not self.image_loaded or self.image is None:
            messagebox.showwarning("警告", "请先打开图像")
            return

        try:
            # 获取三种算法的处理结果
            results = process_func()

            # 确保每个算法都有有效的结果
            valid_results = []
            for i, result in enumerate(results):
                if result is not None and result.size > 0:
                    valid_results.append(result)
                    self.log(f"  {self.algorithm_manager.algorithm_names[i]}: 结果有效")
                else:
                    # 如果结果无效，使用原始图像
                    valid_results.append(self.image.copy())
                    self.log(
                        f"  {self.algorithm_manager.algorithm_names[i]}: 使用原始图像"
                    )

            # 显示处理结果
            self.display_manager.show_images(valid_results, operation_name)
            self.log(f"✓ {operation_name}处理完成，三种算法结果已拼接显示")

            # 更新当前图像为OpenCV算法的结果
            if len(valid_results) > 0 and valid_results[0].size > 0:
                self.image = valid_results[0]

        except Exception as e:
            self.log(f"✗ {operation_name}处理失败: {e}")
            messagebox.showerror("错误", f"{operation_name}处理失败: {e}")

    def rotate_90_all(self) -> None:
        """使用三种方法旋转图像90度"""
        self.process_and_show(
            "rotate 90°",
            lambda: self.algorithm_manager.process_rotation(self.image, self.log),
        )

    def scale_image_all(self, scale_factor: float) -> None:
        """使用三种方法缩放图像"""
        if not self.image_loaded or self.image is None:
            messagebox.showwarning("警告", "请先打开图像")
            return

        self.current_scale *= scale_factor
        self.update_scale_display()

        self.process_and_show(
            f"scale {scale_factor:.2f}x",
            lambda: self.algorithm_manager.process_scaling(
                self.image, scale_factor, self.log
            ),
        )

    def invert_colors_all(self) -> None:
        """使用三种方法进行反色处理"""
        self.process_and_show(
            "invert colors",
            lambda: self.algorithm_manager.process_inversion(self.image, self.log),
        )

    def grayscale_all(self) -> None:
        """使用三种方法进行灰度化处理"""
        self.process_and_show(
            "grayscale",
            lambda: self.algorithm_manager.process_grayscale(self.image, self.log),
        )

    def reset_image(self) -> None:
        """重置图像到原始状态"""
        if self.original_image is not None and self.image_loaded:
            self.image = self.original_image.copy()
            self.current_scale = 1.0
            self.update_scale_display()
            self.show_original_images()
            self.log("\n✓ 图像已重置到原始状态")

    def on_closing(self) -> None:
        """程序关闭时的清理工作"""
        self.display_manager.close_window()
        cv2.destroyAllWindows()
        self.root.quit()
        self.root.destroy()

    def run(self) -> None:
        """运行图像浏览器主循环"""
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"程序运行错误: {e}")
        finally:
            cv2.destroyAllWindows()


def main() -> None:
    """主函数"""
    try:
        import cv2
        import numpy as np
        import einops

        print("正在启动图像浏览器...")
        print("提示: 请先打开图像文件，处理结果将在单个窗口中并排显示")
        print("窗口标题: 三算法对比结果")
        print("缩放因子: 放大2.0x, 缩小0.5x")

        browser = TripleImageBrowser()
        browser.run()

    except ImportError as e:
        print(f"缺少依赖库: {e}")
        print("请安装: pip install opencv-python numpy einops")


if __name__ == "__main__":
    main()
