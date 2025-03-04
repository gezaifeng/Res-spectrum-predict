import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ========================= 数据集定义 =========================
class SpectralDataset(Dataset):
    def __init__(self, data_dir, normalize_spectra=True):
        """
        读取存储在 data_dir 目录下的 RGB 和光谱数据
        :param data_dir: 数据集文件夹路径
        :param normalize_spectra: 是否对光谱数据进行归一化
        """
        self.data_dir = data_dir
        self.rgb_files = sorted([f for f in os.listdir(data_dir) if f.startswith("rgb") and f.endswith(".npy")])
        self.spectral_files = sorted([f.replace("rgb_", "spectral_") for f in self.rgb_files])

        self.normalize_spectra = normalize_spectra
        self.mean = None
        self.std = None

        # 计算光谱数据的均值和标准差
        if self.normalize_spectra:
            self.mean, self.std = self.compute_spectra_stats()

    def compute_spectra_stats(self):
        """ 计算所有光谱数据的均值和标准差 """
        all_spectra = []
        for spectral_file in self.spectral_files:
            spectral_path = os.path.join(self.data_dir, spectral_file)
            spectral_data = np.load(spectral_path).astype(np.float32)
            all_spectra.append(spectral_data)

        all_spectra = np.stack(all_spectra)  # (N, 10)
        return np.mean(all_spectra, axis=0), np.std(all_spectra, axis=0)

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        # 读取 RGB 数据
        rgb_path = os.path.join(self.data_dir, self.rgb_files[idx])
        spectral_path = os.path.join(self.data_dir, self.spectral_files[idx])

        rgb_data = np.load(rgb_path).astype(np.float32)  # (4, 6, 100, 3)
        spectral_data = np.load(spectral_path).astype(np.float32)  # (10,)

        # 归一化 RGB 数据到 [0,1]
        rgb_data /= 255.0

        # 交换维度: (4, 6, 100, 3) → (3, 100, 4, 6)
        rgb_data = np.transpose(rgb_data, (3, 2, 0, 1))

        # 归一化光谱数据
        if self.normalize_spectra:
            spectral_data = (spectral_data - self.mean) / self.std

        # 转换为 PyTorch tensor
        rgb_tensor = torch.tensor(rgb_data)  # (3, 100, 4, 6)
        spectral_tensor = torch.tensor(spectral_data, dtype=torch.float32)  # (10,)

        return rgb_tensor, spectral_tensor


