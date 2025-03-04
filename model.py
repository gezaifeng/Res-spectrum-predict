import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================= ResCNN 模型定义 =========================
class ResCNN(nn.Module):
    def __init__(self):
        super(ResCNN, self).__init__()

        # 2D 卷积层用于处理 RGB 数据 (3, 100, 4, 6)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 池化层减少特征维度

        # 残差连接的1x1卷积映射层，匹配维度
        self.residual = nn.Conv2d(3, 128, kernel_size=1, stride=1, padding=0)

        # 计算展平后的正确维度
        flattened_dim = 128 * 1 * 1  # 由于两次池化，H,W变为1x1

        # 全连接层
        self.fc1 = nn.Linear(flattened_dim, 256)  # 计算展平后的维度
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # 输出10维光谱数据

    def forward(self, x):
        b, c, seq, h, w = x.shape  # (batch, 3, 100, 4, 6)
        x = x.reshape(b * seq, c, h, w)  # 变换形状适应 2D 卷积

        # 主路径
        x1 = F.relu(self.conv1(x))
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = self.pool(F.relu(self.conv3(x1)))  # (batch*100, 128, 1, 1)

        # 残差连接
        x_res = self.residual(x)  # 让原始输入数据通过 1x1 卷积调整维度
        x_res = self.pool(self.pool(x_res))  # 与 x1 形状匹配 (batch*100, 128, 1, 1)
        x1 = x1 + x_res  # 残差相加

        # 展平并通过全连接层
        x1 = x1.reshape(b, seq, -1)  # (batch, 100, 128*1*1)
        x1 = F.relu(self.fc1(x1))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)

        return x1.mean(dim=1)  # 取均值作为最终光谱预测结果
