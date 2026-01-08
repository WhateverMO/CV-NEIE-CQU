import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from tqdm import tqdm
import ssl

# --- 配置 ---
ssl._create_default_https_context = ssl._create_unverified_context
classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def load_cifar_data():
    print("正在加载 CIFAR-10 数据集...")
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # 抽取子集 (训练10000, 测试2000)
    x_train = trainset.data[:10000]
    y_train = np.array(trainset.targets)[:10000]
    x_test = testset.data[:2000]
    y_test = np.array(testset.targets)[:2000]
    return x_train, y_train, x_test, y_test


def get_hog_features(images):
    feats = []
    print("提取 HOG 特征中...")
    for img in tqdm(images, unit="img"):
        gray = rgb2gray(img)
        f = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False,
        )
        feats.append(f)
    return np.array(feats)


# --- 可视化函数 ---
def plot_results(acc_raw, acc_hog):
    """绘制准确率对比柱状图"""
    methods = ["Raw Pixels", "HOG Features"]
    accuracies = [acc_raw * 100, acc_hog * 100]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, accuracies, color=["gray", "orange"])
    plt.ylabel("Accuracy (%)")
    plt.title("Linear Classification Performance Comparison")
    plt.ylim(0, max(accuracies) + 10)

    # 在柱子上标数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 1,
            f"{yval:.2f}%",
            ha="center",
            va="bottom",
        )

    plt.show()


def visualize_predictions(images, true_labels, pred_labels, title="Model Predictions"):
    """随机展示5张图片的预测结果"""
    indices = np.random.choice(len(images), 5, replace=False)

    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, 5, i + 1)
        plt.imshow(images[idx])

        true_name = classes[true_labels[idx]]
        pred_name = classes[pred_labels[idx]]

        # 预测正确绿色，错误红色
        color = "green" if true_labels[idx] == pred_labels[idx] else "red"

        plt.title(f"True: {true_name}\nPred: {pred_name}", color=color)
        plt.axis("off")
    plt.suptitle(title)
    plt.show()


def run_experiment():
    x_train, y_train, x_test, y_test = load_cifar_data()

    # --- 实验 1: 原始像素 ---
    print("\n[实验 1] 训练原始像素模型...")
    x_train_flat = x_train.reshape(len(x_train), -1).astype(np.float32)
    x_test_flat = x_test.reshape(len(x_test), -1).astype(np.float32)

    scaler = StandardScaler()
    x_train_std = scaler.fit_transform(x_train_flat)
    x_test_std = scaler.transform(x_test_flat)

    # 修改点：删除了 multi_class 参数
    clf_raw = LogisticRegression(solver="saga", max_iter=100)
    clf_raw.fit(x_train_std, y_train)
    preds_raw = clf_raw.predict(x_test_std)
    acc_raw = accuracy_score(y_test, preds_raw)

    print(f"原始像素准确率: {acc_raw:.4f}")
    print(classification_report(y_test, preds_raw, target_names=classes))

    # --- 实验 2: HOG 特征 ---
    print("\n[实验 2] 训练 HOG 特征模型...")
    x_train_hog = get_hog_features(x_train)
    x_test_hog = get_hog_features(x_test)

    scaler_hog = StandardScaler()
    x_train_hog_std = scaler_hog.fit_transform(x_train_hog)
    x_test_hog_std = scaler_hog.transform(x_test_hog)

    # 修改点：删除了 multi_class 参数
    clf_hog = LogisticRegression(solver="saga", max_iter=100)
    clf_hog.fit(x_train_hog_std, y_train)
    preds_hog = clf_hog.predict(x_test_hog_std)
    acc_hog = accuracy_score(y_test, preds_hog)

    print(f"HOG 特征准确率: {acc_hog:.4f}")
    print(classification_report(y_test, preds_hog, target_names=classes))

    # --- 可视化展示 ---
    print("正在生成可视化报告...")
    # 1. 柱状图对比
    plot_results(acc_raw, acc_hog)
    # 2. HOG模型的预测结果展示
    visualize_predictions(
        x_test, y_test, preds_hog, title="HOG + Logistic Regression Predictions"
    )


if __name__ == "__main__":
    run_experiment()
