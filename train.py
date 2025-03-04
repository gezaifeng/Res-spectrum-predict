import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import ResCNN
from dataset import SpectralDataset
# ========================= 训练代码 =========================

# 设备选择
device = "cuda" if torch.cuda.is_available() else "cpu"

# 训练数据路径
train_dir = "D:\Desktop\model\Res spectrum predict\dataset-76\\train"  # 请确保路径正确
val_dir = "D:\Desktop\model\Res spectrum predict\dataset-76\\val"

# 加载数据
train_dataset = SpectralDataset(train_dir)
val_dataset = SpectralDataset(val_dir)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 初始化模型
model = ResCNN().to(device)

# 损失函数 & 优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 训练参数
num_epochs = 50
best_val_loss = float("inf")  # 记录最佳验证损失
save_path = "D:\\Desktop\\model\\Res spectrum predict\\best_model.pth"

# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # 训练
    for images, spectra in train_loader:
        images, spectra = images.to(device).float(), spectra.to(device).float()
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, spectra)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 计算训练集损失
    train_loss = running_loss / len(train_loader)

    # 计算验证损失
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, spectra in val_loader:
            images, spectra = images.to(device).float(), spectra.to(device).float()
            outputs = model(images)
            val_loss += criterion(outputs, spectra).item()

    val_loss /= len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 保存最优模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
        print(f"✅ Model saved at epoch {epoch+1} with Val Loss: {val_loss:.4f}")

print("🎯 训练完成！最佳模型已保存。")
