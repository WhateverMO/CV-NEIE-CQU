import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from skimage.feature import hog
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import ssl
from tqdm import tqdm

# ==========================================
# 配置：自动下载 + 忽略SSL
# ==========================================
ssl._create_default_https_context = ssl._create_unverified_context
datasets.MNIST.mirrors = ["https://storage.googleapis.com/cvdf-datasets/mnist/"]


def load_data():
    """加载数据，使用缓存避免重复下载"""
    print("正在检查/下载 MNIST 数据集...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # 抽取子集 (训练5000, 测试1000) 加快速度
    # 注意：为了让KMeans结果更稳定，聚类时我们通常需要更多数据，但为了演示这里保持5000
    print("抽取数据子集...")
    train_idx = np.random.choice(len(train_set), 5000, replace=False)
    test_idx = np.random.choice(len(test_set), 1000, replace=False)

    return (
        train_set.data[train_idx].numpy(),
        train_set.targets[train_idx].numpy(),
        test_set.data[test_idx].numpy(),
        test_set.targets[test_idx].numpy(),
    )


def extract_hog_features(images):
    """提取 HOG 特征"""
    feature_list = []
    print(f"正在提取 HOG 特征...")
    for img in tqdm(images, unit="img"):
        fd = hog(
            img,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            visualize=False,
        )
        feature_list.append(fd)
    return np.array(feature_list)


def visualize_hog_demo(image):
    """展示一张原图和对应的HOG特征图"""
    fd, hog_image = hog(
        image,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        visualize=True,
    )

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(hog_image, cmap="gray")
    plt.title("HOG Visualization")
    plt.axis("off")
    plt.suptitle("HOG Feature Extraction Demo")
    plt.show()


def plot_comparison_matrices(y_true_kmeans, y_pred_kmeans, y_true_svm, y_pred_svm):
    """并排绘制 KMeans 和 SVM 的混淆矩阵"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. 绘制 KMeans 混淆矩阵
    ConfusionMatrixDisplay.from_predictions(
        y_true_kmeans, y_pred_kmeans, ax=axes[0], cmap="Blues", normalize=None
    )
    axes[0].set_title("KMeans Clustering Confusion Matrix\n(Unsupervised)")

    # 2. 绘制 SVM 混淆矩阵
    ConfusionMatrixDisplay.from_predictions(
        y_true_svm, y_pred_svm, ax=axes[1], cmap="Greens", normalize=None
    )
    axes[1].set_title("SVM Classification Confusion Matrix\n(Supervised)")

    plt.suptitle("Performance Comparison: HOG + KMeans vs HOG + SVM")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. 准备数据
    train_x, train_y, test_x, test_y = load_data()

    # [可视化 1] 展示 HOG 特征提取效果
    print("展示 HOG 特征示例...")
    visualize_hog_demo(train_x[0])

    # 2. 提取特征
    train_feat = extract_hog_features(train_x)
    test_feat = extract_hog_features(test_x)

    # ==========================================
    # 任务A: KMeans 聚类
    # ==========================================
    print("\n" + "=" * 40)
    print("任务A: KMeans 聚类评估")
    print("=" * 40)
    # 使用训练集进行聚类演示
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    pred_clusters = kmeans.fit_predict(train_feat)

    # 将聚类结果映射为真实标签
    label_map = {}
    for i in range(10):
        indices = np.where(pred_clusters == i)[0]
        if len(indices) > 0:
            # 这一簇中出现最多的真实标签，即为该簇的预测标签
            label_map[i] = np.bincount(train_y[indices]).argmax()
        else:
            label_map[i] = -1

    mapped_preds_kmeans = np.array([label_map[c] for c in pred_clusters])

    acc_cluster = accuracy_score(train_y, mapped_preds_kmeans)
    print(f"KMeans 聚类正确率 (Accuracy): {acc_cluster * 100:.2f}%")
    print("详细报告:")
    print(classification_report(train_y, mapped_preds_kmeans, zero_division=0))

    # ==========================================
    # 任务B: SVM 分类
    # ==========================================
    print("\n" + "=" * 40)
    print("任务B: SVM 分类评估")
    print("=" * 40)
    # 训练 SVM
    clf = SVC(kernel="linear")
    clf.fit(train_feat, train_y)
    # 预测测试集
    preds_svm = clf.predict(test_feat)

    acc_svm = accuracy_score(test_y, preds_svm)
    print(f"SVM 分类正确率 (Accuracy): {acc_svm * 100:.2f}%")
    print("详细报告:")
    print(classification_report(test_y, preds_svm))

    # ==========================================
    # [可视化 2] 并排展示混淆矩阵
    # ==========================================
    print("正在生成混淆矩阵对比图...")
    # 注意：KMeans我们是在 train_set 上做的（无监督通常看聚类效果），SVM是在 test_set 上评估的
    # 为了对比公平性展示各自的最佳效果，这里传入各自评估所用的数据
    plot_comparison_matrices(train_y, mapped_preds_kmeans, test_y, preds_svm)
