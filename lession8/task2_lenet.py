import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import ssl

# ==========================================
# 🚨 MNIST 下载修复配置 🚨
# ==========================================
ssl._create_default_https_context = ssl._create_unverified_context
datasets.MNIST.mirrors = ["https://storage.googleapis.com/cvdf-datasets/mnist/"]
datasets.MNIST.resources = [
    ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbc9874895"),
    ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
]
# ==========================================


# 🚀 自动获取设备
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


device = get_device()
print(f"🚀 PyTorch Running on: {device}")


# 1. 定义 LeNet-5 模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 2. 训练与测试函数 (增加了 .to(device))
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    train_losses = []
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # 数据搬运到 GPU/MPS
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")
    return train_losses


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # 数据搬运到 GPU/MPS
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


# 3. 主程序
if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    print("加载 MNIST 数据集...")
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transform, download=True
    )

    # 增加 num_workers 加速数据读取
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=1000, shuffle=False, num_workers=0
    )

    criterion = nn.CrossEntropyLoss()

    # --- 实验 1: 标准训练 ---
    print("\n[实验 1] 标准训练 (LR=0.001)...")
    # 模型搬运到 GPU/MPS
    model_std = LeNet5().to(device)
    opt_std = optim.Adam(model_std.parameters(), lr=0.001)

    losses_std = train_model(model_std, train_loader, criterion, opt_std, epochs=5)
    acc_std = test_model(model_std, test_loader)
    print(f"Final Accuracy: {acc_std * 100:.2f}%")

    plt.figure(figsize=(8, 5))
    plt.plot(losses_std, label="Training Loss (LR=0.001)")
    plt.title(f"LeNet-5 Training Loss (Device: {device})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # --- 实验 2: 学习率对比 ---
    print("\n[实验 2] 学习率对比实验...")
    learning_rates = [0.01, 0.001, 0.0001]
    accuracies = []

    for lr in learning_rates:
        print(f"Testing LR: {lr} ...")
        # 模型搬运到 GPU/MPS
        model = LeNet5().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_model(model, train_loader, criterion, optimizer, epochs=5)
        acc = test_model(model, test_loader)
        accuracies.append(acc)

    plt.figure(figsize=(8, 5))
    plt.plot(learning_rates, accuracies, marker="o")
    plt.xscale("log")
    plt.title("Effect of Learning Rate on Accuracy")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Test Accuracy")
    plt.show()
