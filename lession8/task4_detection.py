from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import os
import ssl

# 忽略SSL错误
ssl._create_default_https_context = ssl._create_unverified_context


# 🚀 自动获取 YOLO 设备字符串
def get_yolo_device():
    if torch.cuda.is_available():
        return "0"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def plot_training_loss(train_dir):
    """查找最新的训练结果并绘制 Loss"""
    try:
        csv_file = os.path.join(train_dir, "results.csv")
        df = pd.read_csv(csv_file)
        df.columns = [c.strip() for c in df.columns]

        plt.figure(figsize=(10, 5))
        plt.plot(df["train/box_loss"], label="Box Loss")
        plt.plot(df["train/obj_loss"], label="Obj Loss")
        plt.plot(df["train/cls_loss"], label="Cls Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("YOLOv8 Detection Training Loss")
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"无法绘制 Loss 曲线: {e}")


if __name__ == "__main__":
    device_str = get_yolo_device()
    print(f"🚀 YOLO Running on: {device_str}")

    # 1. 加载检测模型
    model = YOLO("yolov8n.pt")

    # 2. 训练
    print("开始目标检测训练...")
    train_res = model.train(
        data="coco128.yaml", epochs=10, imgsz=640, device=device_str, workers=0
    )

    # 获取训练结果保存目录
    save_dir = train_res.save_dir
    print(f"训练结果保存在: {save_dir}")

    # 3. 绘制 Loss 曲线
    plot_training_loss(save_dir)

    # 4. 推理
    print("开始推理...")
    img_url = "https://ultralytics.com/images/bus.jpg"
    preds = model.predict(source=img_url, save=True, conf=0.5, device=device_str)

    # 5. 展示推理结果
    res = preds[0]
    im_array = res.plot()

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
    plt.title(f"YOLOv8 Detection Result (Device: {device_str})")
    plt.axis("off")
    plt.show()
