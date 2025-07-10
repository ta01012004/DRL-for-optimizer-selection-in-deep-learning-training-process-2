import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Sử dụng GPU nếu có, nếu không thì dùng CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Tải dữ liệu CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor()  # chuyển ảnh PIL thành Tensor và scale về [0,1]
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_dataset   = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# DataLoader cho tập train và val
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False)
