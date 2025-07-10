import torchvision
import torchvision.transforms as transforms
import time

# Chuẩn bị dữ liệu CIFAR-100
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),        # cắt ngẫu nhiên 32x32 sau khi đệm 4px
    transforms.RandomHorizontalFlip(),           # lật ngang ngẫu nhiên
    transforms.ToTensor(),
    transforms.Normalize((0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761))  # chuẩn hóa CIFAR-100
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761))
])
train_dataset = torchvision.datasets.CIFAR100(root="./data", train=True, transform=train_transform, download=True)
test_dataset = torchvision.datasets.CIFAR100(root="./data", train=False, transform=test_transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# Khởi tạo mô hình và optimizer
model = WideResNet(depth=16, widen_factor=8, dropout_rate=0.0, num_classes=100).to(device)
base_optimizer = torch.optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9, weight_decay=5e-4, rho=0.05, adaptive=False)
# Sử dụng scheduler giảm learning rate dần (ví dụ: CosineAnnealingLR hoặc StepLR nếu cần)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=100)

# Vòng lặp huấn luyện
num_epochs = 100
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        # Tính toán loss bước 1
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)  # có thể dùng label smoothing: F.cross_entropy(outputs, labels, label_smoothing=0.1)
        loss.backward()
        optimizer.first_step(zero_grad=True)     # bước SAM đầu tiên

        # Tính toán loss bước 2 (với trọng số đã dịch chuyển)
        outputs2 = model(images)
        loss2 = F.cross_entropy(outputs2, labels)
        loss2.backward()
        optimizer.second_step(zero_grad=True)    # bước SAM thứ hai

    # Lịch trình giảm học rate (nếu có)
    scheduler.step()

    # Đánh giá nhanh trên tập validation mỗi epoch (sử dụng tập test ở đây cho đơn giản)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100. * correct / total
    print(f"Epoch {epoch+1}: Accuracy trên tập validation = {val_acc:.2f}%")
end_time = time.time()
print(f"Tổng thời gian huấn luyện: {end_time - start_time:.2f} giây")
# Sau khi huấn luyện hoàn tất, lưu mô hìn
model_save_path = "C:/Users/PhamTuanAnh/Desktop/DRL/Project/wide_resnet_cifar100.pth"
# Kiểm tra và tạo thư mục nếu chưa tồn tại
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

torch.save(model.state_dict(), model_save_path)
print(f"Mô hình đã được lưu tại {model_save_path}")
