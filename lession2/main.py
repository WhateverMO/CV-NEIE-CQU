import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from typing import Optional, Tuple, List, Any, Protocol
import matplotlib.pyplot as plt
import numpy.typing as npt
import time
from Algorithm.Algorithm import (
    ImageAlgorithm,
    AdvancedImageAlgorithm,
    OpenCVAlgorithm,
    PurePythonAlgorithm,
)


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

    def resize_images_to_same_height(
        self, images: List[npt.NDArray]
    ) -> List[npt.NDArray]:
        """将所有图像调整为相同高度"""
        if not images or any(img.size == 0 for img in images):
            return images

        min_height = min(img.shape[0] for img in images if img.size > 0)
        resized_images = []

        for img in images:
            if img.size == 0:
                resized_images.append(img)
                continue

            height, width = img.shape[:2]
            if height != min_height:
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

        resized_images = self.resize_images_to_same_height(images)
        concatenated = np.hstack(resized_images)
        return concatenated

    def show_images(self, images: List[npt.NDArray], operation_name: str) -> None:
        """在单个窗口中显示三张横向拼接的图像"""
        self.create_window()

        titles = [f"{name} - {operation_name}" for name in ["OpenCV", "纯Python"]]
        concatenated = self.concatenate_images_horizontally(images, titles)

        if concatenated.size == 0:
            return

        img_height, img_width = concatenated.shape[:2]
        max_width = min(img_width, self.screen_width - 100)
        max_height = min(img_height, self.screen_height - 100)

        scale_x = max_width / img_width
        scale_y = max_height / img_height
        scale = min(scale_x, scale_y, 1.0)

        display_width = int(img_width * scale)
        display_height = int(img_height * scale)

        cv2.resizeWindow(self.window_name, display_width, display_height)
        cv2.imshow(self.window_name, concatenated)
        cv2.waitKey(1)

    def close_window(self) -> None:
        """关闭窗口"""
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass
        self.window_created = False


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

        # 继续TripleImageBrowser主类的剩余部分
        self.open_cv_algorithm = OpenCVAlgorithm()
        self.pure_python_algorithm = PurePythonAlgorithm()
        self.display_manager = ImageDisplayManager()

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
            main_frame,
            text="三算法对比图像浏览器 - 作业2功能",
            font=("Arial", 16, "bold"),
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

        # 基础图像操作按钮
        basic_frame = tk.Frame(control_frame)
        basic_frame.pack(fill=tk.X, pady=5)

        basic_operations = [
            ("旋转90°", self.rotate_90_all),
            ("放大2.0x", lambda: self.scale_image_all(2.0)),
            ("缩小0.5x", lambda: self.scale_image_all(0.5)),
            ("反色处理", self.invert_colors_all),
            ("灰度化", self.grayscale_all),
        ]

        for text, command in basic_operations:
            btn = tk.Button(basic_frame, text=text, command=command, width=10, height=2)
            btn.pack(side=tk.LEFT, padx=2)
            btn.config(state=tk.DISABLED)
            setattr(
                self,
                f"{text.replace('°', '').replace('.', '').replace(' ', '')}_button",
                btn,
            )

        # 几何变换按钮
        geo_frame = tk.Frame(control_frame)
        geo_frame.pack(fill=tk.X, pady=5)

        tk.Label(geo_frame, text="几何变换:", font=("Arial", 9, "bold")).pack(
            side=tk.LEFT, padx=5
        )

        geo_operations = [
            ("X轴斜切", self.shear_x_all),
            ("Y轴斜切", self.shear_y_all),
        ]

        for text, command in geo_operations:
            btn = tk.Button(geo_frame, text=text, command=command, width=10, height=2)
            btn.pack(side=tk.LEFT, padx=2)
            btn.config(state=tk.DISABLED)
            setattr(self, f"{text.replace(' ', '')}_button", btn)

        # 图像增强按钮
        enhance_frame = tk.Frame(control_frame)
        enhance_frame.pack(fill=tk.X, pady=5)

        tk.Label(enhance_frame, text="图像增强:", font=("Arial", 9, "bold")).pack(
            side=tk.LEFT, padx=5
        )

        enhance_operations = [
            ("显示直方图", self.show_histogram_all),
            ("直方图均衡", self.histogram_equalization_all),
            ("添加噪声", self.add_noise_all),
            ("中值滤波", self.median_filter_all),
        ]

        for text, command in enhance_operations:
            btn = tk.Button(
                enhance_frame, text=text, command=command, width=10, height=2
            )
            btn.pack(side=tk.LEFT, padx=2)
            btn.config(state=tk.DISABLED)
            setattr(self, f"{text.replace(' ', '')}_button", btn)

        # 锐化按钮
        sharpen_frame = tk.Frame(control_frame)
        sharpen_frame.pack(fill=tk.X, pady=5)

        tk.Label(sharpen_frame, text="图像锐化:", font=("Arial", 9, "bold")).pack(
            side=tk.LEFT, padx=5
        )

        sharpen_operations = [
            ("Roberts", lambda: self.sharpen_all("roberts")),
            ("Sobel", lambda: self.sharpen_all("sobel")),
            ("Laplacian", lambda: self.sharpen_all("laplacian")),
        ]

        for text, command in sharpen_operations:
            btn = tk.Button(
                sharpen_frame, text=text, command=command, width=10, height=2
            )
            btn.pack(side=tk.LEFT, padx=2)
            btn.config(state=tk.DISABLED)
            setattr(self, f"{text}_button", btn)

        # 频域处理按钮
        freq_frame = tk.Frame(control_frame)
        freq_frame.pack(fill=tk.X, pady=5)

        tk.Label(freq_frame, text="频域处理:", font=("Arial", 9, "bold")).pack(
            side=tk.LEFT, padx=5
        )

        freq_operations = [
            ("低通滤波", lambda: self.frequency_filter_all("low")),
            ("高通滤波", lambda: self.frequency_filter_all("high")),
        ]

        for text, command in freq_operations:
            btn = tk.Button(freq_frame, text=text, command=command, width=10, height=2)
            btn.pack(side=tk.LEFT, padx=2)
            btn.config(state=tk.DISABLED)
            setattr(self, f"{text.replace(' ', '')}_button", btn)

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
        self.log("三算法对比图像浏览器 - 作业2功能")
        self.log("=" * 60)
        self.log("功能列表:")
        self.log("1. 基础操作: 旋转、缩放、反色、灰度化")
        self.log("2. 几何变换: 斜切变换")
        self.log("3. 图像增强: 直方图均衡、噪声添加、中值滤波")
        self.log("4. 图像锐化: Roberts, Sobel, Laplacian")
        self.log("5. 频域处理: 低通滤波、高通滤波")
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
            "X轴斜切_button",
            "Y轴斜切_button",
            "显示直方图_button",
            "直方图均衡_button",
            "添加噪声_button",
            "中值滤波_button",
            "Roberts_button",
            "Sobel_button",
            "Laplacian_button",
            "低通滤波_button",
            "高通滤波_button",
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
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")],
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
            images = [self.image.copy(), self.image.copy()]
            self.display_manager.show_images(images, "原始图像")
            self.log("✓ 原始图像已显示")

    def process_and_show(self, operation_name: str, process_func) -> None:
        """通用的处理并显示函数"""
        if not self.image_loaded or self.image is None:
            messagebox.showwarning("警告", "请先打开图像")
            return

        try:
            # 获取两种算法的处理结果
            start_time = time.time()
            opencv_result = process_func(self.open_cv_algorithm)
            opencv_time = time.time() - start_time

            start_time = time.time()
            python_result = process_func(self.pure_python_algorithm)
            python_time = time.time() - start_time

            results = [opencv_result, python_result]

            # 检查结果是否有效
            valid_results = []
            for i, (result, name) in enumerate(
                zip(results, ["OpenCV算法", "纯Python算法"])
            ):
                if result is not None and result.size > 0:
                    valid_results.append(result)
                    self.log(f"  {name}: 处理成功")
                else:
                    valid_results.append(self.image.copy())
                    self.log(f"  {name}: 使用原始图像")

            # 显示处理结果
            self.display_manager.show_images(valid_results, operation_name)
            self.log(f"✓ {operation_name}处理完成")
            self.log(f"  OpenCV耗时: {opencv_time:.4f}秒")
            self.log(f"  纯Python耗时: {python_time:.4f}秒")

            # 更新当前图像为OpenCV算法的结果
            if len(valid_results) > 0 and valid_results[0].size > 0:
                self.image = valid_results[0]

        except Exception as e:
            self.log(f"✗ {operation_name}处理失败: {e}")
            messagebox.showerror("错误", f"{operation_name}处理失败: {e}")

    def rotate_90_all(self) -> None:
        """旋转图像90度"""
        self.process_and_show("旋转90°", lambda alg: alg.rotate_90(self.image))

    def scale_image_all(self, scale_factor: float) -> None:
        """缩放图像"""
        if not self.image_loaded or self.image is None:
            messagebox.showwarning("警告", "请先打开图像")
            return

        self.current_scale *= scale_factor
        self.update_scale_display()

        self.process_and_show(
            f"缩放{scale_factor:.2f}x",
            lambda alg: alg.scale_image(self.image, scale_factor),
        )

    def invert_colors_all(self) -> None:
        """反色处理"""
        self.process_and_show("反色处理", lambda alg: alg.invert_colors(self.image))

    def grayscale_all(self) -> None:
        """灰度化处理"""
        self.process_and_show("灰度化", lambda alg: alg.grayscale(self.image))

    def shear_x_all(self) -> None:
        """X轴斜切"""
        if not self.image_loaded or self.image is None:
            messagebox.showwarning("警告", "请先打开图像")
            return

        shear_factor = simpledialog.askfloat("斜切因子", "请输入X轴斜切因子:")
        if shear_factor is None:
            return

        self.process_and_show(
            f"X轴斜切({shear_factor:.2f})",
            lambda alg: alg.shear_x(self.image, shear_factor),
        )

    def shear_y_all(self) -> None:
        """Y轴斜切"""
        if not self.image_loaded or self.image is None:
            messagebox.showwarning("警告", "请先打开图像")
            return

        shear_factor = simpledialog.askfloat("斜切因子", "请输入Y轴斜切因子:")
        if shear_factor is None:
            return

        self.process_and_show(
            f"Y轴斜切({shear_factor:.2f})",
            lambda alg: alg.shear_y(self.image, shear_factor),
        )

    def show_histogram_all(self) -> None:
        """显示灰度直方图"""
        if not self.image_loaded or self.image is None:
            messagebox.showwarning("警告", "请先打开图像")
            return

        try:
            # OpenCV算法计算直方图
            hist_opencv, bins = self.open_cv_algorithm.calculate_histogram(self.image)

            # 纯Python算法计算直方图
            hist_python, bins = self.pure_python_algorithm.calculate_histogram(
                self.image
            )

            # 绘制直方图
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle("灰度直方图对比", fontsize=16)

            axes[0].bar(bins, hist_opencv[: len(bins)], color="blue", alpha=0.7)
            axes[0].set_title("OpenCV算法")
            axes[0].set_xlabel("灰度值")
            axes[0].set_ylabel("像素数量")
            axes[0].grid(True, alpha=0.3)

            axes[1].bar(bins, hist_python[: len(bins)], color="green", alpha=0.7)
            axes[1].set_title("纯Python算法")
            axes[1].set_xlabel("灰度值")
            axes[1].set_ylabel("像素数量")
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
            self.log("✓ 灰度直方图显示成功")

        except Exception as e:
            self.log(f"✗ 显示灰度直方图失败: {e}")
            messagebox.showerror("错误", f"显示灰度直方图失败: {e}")

    def histogram_equalization_all(self) -> None:
        """直方图均衡化"""
        self.process_and_show(
            "直方图均衡化", lambda alg: alg.histogram_equalization(self.image)
        )

    def add_noise_all(self) -> None:
        """添加椒盐噪声"""
        if not self.image_loaded or self.image is None:
            messagebox.showwarning("警告", "请先打开图像")
            return

        amount = simpledialog.askfloat(
            "噪声比例",
            "请输入噪声比例(0.0-0.5):",
            minvalue=0.0,
            maxvalue=0.5,
            initialvalue=0.05,
        )
        if amount is None:
            return

        self.process_and_show(
            f"添加噪声({amount:.2f})",
            lambda alg: alg.add_salt_pepper_noise(self.image, amount),
        )

    def median_filter_all(self) -> None:
        """中值滤波"""
        if not self.image_loaded or self.image is None:
            messagebox.showwarning("警告", "请先打开图像")
            return

        kernel_size = simpledialog.askinteger(
            "滤波器大小",
            "请输入滤波器大小(奇数):",
            minvalue=3,
            maxvalue=15,
            initialvalue=3,
        )
        if kernel_size is None or kernel_size % 2 == 0:
            messagebox.showerror("错误", "滤波器大小必须为奇数")
            return

        self.process_and_show(
            f"中值滤波({kernel_size}x{kernel_size})",
            lambda alg: alg.median_filter(self.image, kernel_size),
        )

    def sharpen_all(self, method: str) -> None:
        """图像锐化"""
        if not self.image_loaded or self.image is None:
            messagebox.showwarning("警告", "请先打开图像")
            return

        method_names = {
            "roberts": "Roberts算子",
            "sobel": "Sobel算子",
            "laplacian": "Laplacian算子",
        }
        method_name = method_names.get(method, method)

        self.process_and_show(
            f"锐化-{method_name}", lambda alg: alg.sharpen(self.image, method)
        )

    def frequency_filter_all(self, mode: str) -> None:
        """频域滤波"""
        if not self.image_loaded or self.image is None:
            messagebox.showwarning("警告", "请先打开图像")
            return

        mode_names = {"low": "低通", "high": "高通"}
        mode_name = mode_names.get(mode, mode)

        filter_type = simpledialog.askstring(
            "滤波器类型", "请选择滤波器类型(ideal/gaussian/butterworth):"
        )
        if filter_type not in ["ideal", "gaussian", "butterworth"]:
            messagebox.showerror(
                "错误", "无效的滤波器类型，请输入ideal/gaussian/butterworth"
            )
            return

        cutoff = simpledialog.askinteger(
            "截止频率",
            "请输入截止频率(像素):",
            minvalue=1,
            maxvalue=100,
            initialvalue=20,
        )
        if cutoff is None:
            return

        order = 1
        if filter_type == "butterworth":
            order = simpledialog.askinteger(
                "滤波器阶数",
                "请输入巴特沃斯滤波器阶数:",
                minvalue=1,
                maxvalue=10,
                initialvalue=2,
            )
            if order is None:
                return

        self.process_and_show(
            f"{mode_name}滤波({filter_type}, 截止频率={cutoff})",
            lambda alg: alg.frequency_filter(
                self.image, mode, filter_type, cutoff, order
            ),
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
        import matplotlib.pyplot as plt

        print("正在启动图像浏览器...")
        print("=" * 60)
        print("功能列表:")
        print("1. 基础操作: 旋转、缩放、反色、灰度化")
        print("2. 几何变换: 斜切变换")
        print("3. 图像增强: 直方图均衡、噪声添加、中值滤波")
        print("4. 图像锐化: Roberts, Sobel, Laplacian")
        print("5. 频域处理: 低通滤波、高通滤波")
        print("=" * 60)
        print("请先点击'打开图像'按钮选择图像文件")

        browser = TripleImageBrowser()
        browser.run()

    except ImportError as e:
        print(f"缺少依赖库: {e}")
        print("请安装: pip install opencv-python numpy matplotlib")


if __name__ == "__main__":
    print("start ")
    main()
