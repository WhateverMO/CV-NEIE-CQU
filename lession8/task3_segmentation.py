from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import torch
import ssl

# 忽略SSL错误
ssl._create_default_https_context = ssl._create_unverified_context


# 🚀 自动获取 YOLO 设备字符串
def get_yolo_device():
    if torch.cuda.is_available():
        return "0"  # CUDA 设备 ID
    elif torch.backends.mps.is_available():
        return "mps"  # Mac MPS
    else:
        return "cpu"


if __name__ == "__main__":
    device_str = get_yolo_device()
    print(f"🚀 YOLO Running on: {device_str}")

    # 1. 初始化模型
    print("加载 YOLOv8-Seg 模型...")
    model = YOLO("yolov8n-seg.pt")

    # 2. 训练
    print("开始训练...")
    # workers=0 在某些系统上能避免多进程报错
    model.train(
        data="coco128-seg.yaml", epochs=10, imgsz=640, device=device_str, workers=0
    )

    # 3. 推理
    print("开始推理...")
    img_url = "https://ultralytics.com/images/bus.jpg"
    preds = model.predict(source=img_url, save=True, conf=0.5, device=device_str)

    # 4. 展示结果
    res = preds[0]
    im_array = res.plot()

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
    plt.title(f"YOLOv8 Segmentation (Device: {device_str})")
    plt.axis("off")
    plt.show()
