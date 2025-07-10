import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from torchvision import datasets, transforms, models

# Thiết lập device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Kiểm tra bộ nhớ GPU (nếu dùng CUDA)
if device.type == 'cuda':
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(device)/1e9:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved(device)/1e9:.2f} GB")

# Tải dữ liệu CIFAR-10
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Tăng batch size
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=512, shuffle=False)

# Định nghĩa SAM Optimizer
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert isinstance(base_optimizer, torch.optim.Optimizer), "Base optimizer phải là một Optimizer"
        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # Thêm nhiễu
                self.state[p]["e_w"] = e_w
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]: continue
                p.sub_(self.state[p]["e_w"])  # Xóa nhiễu
        self.base_optimizer.step()  # Cập nhật trọng số
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(torch.stack([
            p.grad.norm(p=2).to(shared_device) for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]), p=2)
        return norm

# Hàm tính accuracy
def calculate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Hàm huấn luyện tổng quát
def train_model(model, train_loader, val_loader, optimizer, num_epochs=150, patience=5):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    start_time = time.time()
    best_val_acc = 0.0
    best_epoch = 0
    counter = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Huấn luyện
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if isinstance(optimizer, SAM):
                # Bước 1: Tính gradient với trọng số hiện tại
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                
                # Tính metrics từ lần forward đầu tiên
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)
                
                # Bước 2: Tính gradient với trọng số đã nhiễu
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.second_step(zero_grad=True)
                
            else:
                # SGD thông thường
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

        # Tính metrics tập huấn luyện
        avg_train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_acc)

        # Đánh giá trên validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val
        val_loss_history.append(avg_val_loss)
        val_acc_history.append(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}!")
                break

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/{num_epochs}] | Time: {epoch_duration:.2f}s | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    total_time = time.time() - start_time
    return train_loss_history, val_loss_history, train_acc_history, val_acc_history, total_time, best_val_acc, best_epoch

# Khởi tạo hai mô hình riêng biệt
model_sgd = models.resnet18(pretrained=True)
model_sgd.fc = nn.Linear(model_sgd.fc.in_features, 10)

model_sam = models.resnet18(pretrained=True)
model_sam.fc = nn.Linear(model_sam.fc.in_features, 10)

# Tạo optimizers
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
base_optimizer = optim.SGD(model_sam.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
optimizer_sam = SAM(model_sam.parameters(), base_optimizer, rho=0.05)

# Huấn luyện với SGD
print("\nTraining with SGD...")
train_loss_sgd, val_loss_sgd, train_acc_sgd, val_acc_sgd, time_sgd, best_val_sgd, epoch_sgd = train_model(
    model_sgd, train_loader, val_loader, optimizer_sgd
)

# Huấn luyện với SAM
print("\nTraining with SAM...")
train_loss_sam, val_loss_sam, train_acc_sam, val_acc_sam, time_sam, best_val_sam, epoch_sam = train_model(
    model_sam, train_loader, val_loader, optimizer_sam
)

# Vẽ biểu đồ so sánh
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_acc_sgd, label='SGD Train')
plt.plot(val_acc_sgd, '--', label='SGD Val')
plt.plot(train_acc_sam, label='SAM Train')
plt.plot(val_acc_sam, '--', label='SAM Val')
plt.title('Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss_sgd, label='SGD Train')
plt.plot(val_loss_sgd, '--', label='SGD Val')
plt.plot(train_loss_sam, label='SAM Train')
plt.plot(val_loss_sam, '--', label='SAM Val')
plt.title('Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# In kết quả
print("\nSGD Results:")
print(f"Best Validation Accuracy: {best_val_sgd:.2f}% at epoch {epoch_sgd}")
print(f"Training Time: {time_sgd:.2f}s")

print("\nSAM Results:")
print(f"Best Validation Accuracy: {best_val_sam:.2f}% at epoch {epoch_sam}")
print(f"Training Time: {time_sam:.2f}s")