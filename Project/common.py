# common.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

# ----------------------------------
# 1) Thiết bị (device)
# ----------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------
# 2) Xây dựng mô hình ResNet
# ----------------------------------
def build_resnet_model(num_classes=10, pretrained=False):
    """
    Khởi tạo một ResNet-18 từ torchvision và thay thế fc layer
    để phù hợp số lớp đầu ra.
    pretrained=True => dùng trọng số ImageNet.
    """
    model = models.resnet18(pretrained=pretrained)
    # Thay lớp fully-connected cuối cùng
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ----------------------------------
# 3) Hàm train 1 epoch
# ----------------------------------
def train_one_epoch(model, optimizer, train_loader, device, criterion=nn.CrossEntropyLoss()):
    """
    Huấn luyện model trong 1 epoch trên train_loader
    Trả về: (average_loss, accuracy)
    """
    model.train()
    correct = 0
    total_loss = 0.0
    total = 0
    for (data, label) in train_loader:
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.size(0)
        _, pred = out.max(1)
        correct += pred.eq(label).sum().item()
        total += label.size(0)

    avg_loss = total_loss / total
    avg_acc = correct / total
    return avg_loss, avg_acc

# ----------------------------------
# 4) Hàm validate
# ----------------------------------
def validate(model, val_loader, device):
    """
    Đánh giá độ chính xác (accuracy) của model trên val_loader
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (data, label) in val_loader:
            data, label = data.to(device), label.to(device)
            out = model(data)
            _, pred = out.max(1)
            correct += pred.eq(label).sum().item()
            total += label.size(0)
    return correct / total
