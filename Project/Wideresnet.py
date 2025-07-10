import torch
import torch.nn as nn
import torch.nn.functional as F

# Định nghĩa khối Residual cơ bản cho WideResNet
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False)
        self.drop_rate = drop_rate
        # Shortcut conv nếu thay đổi kích thước hoặc số kênh
        self.shortcut = None
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)
    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return out + shortcut

# Định nghĩa WideResNet
class WideResNet(nn.Module):
    def __init__(self, depth=16, widen_factor=8, dropout_rate=0.0, num_classes=100):
        super(WideResNet, self).__init__()
        assert (depth - 4) % 6 == 0, "Depth phải dạng 6n+4"
        n = (depth - 4) // 6  # số block mỗi nhóm
        # Các kênh cho các nhóm WideResNet
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, self.in_planes, 3, padding=1, bias=False)  # conv đầu vào
        # Nhóm các block Residual:
        self.layer1 = self._make_layer(n, 16 * widen_factor, stride=1, drop_rate=dropout_rate)
        self.layer2 = self._make_layer(n, 32 * widen_factor, stride=2, drop_rate=dropout_rate)
        self.layer3 = self._make_layer(n, 64 * widen_factor, stride=2, drop_rate=dropout_rate)
        # Lớp BatchNorm và ReLU cuối cùng
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        # Lớp fully-connected cho đầu ra 100 lớp
        self.fc = nn.Linear(64 * widen_factor, num_classes)
    def _make_layer(self, num_blocks, out_planes, stride, drop_rate):
        layers = []
        for i in range(num_blocks):
            stride_i = stride if i == 0 else 1
            layers.append(BasicBlock(self.in_planes, out_planes, stride=stride_i, drop_rate=drop_rate))
            self.in_planes = out_planes
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn(out))
        # Global Average Pooling 2D
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        return self.fc(out)

# Định nghĩa optimizer SAM (Sharpness-Aware Minimization)
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        # base_optimizer: lớp optimizer gốc, ví dụ torch.optim.SGD
        assert rho >= 0.0, "Rho phải không âm"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        # Khởi tạo optimizer cơ sở
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups  # đồng bộ param_groups
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # Bước 1: cập nhật tham số theo hướng tăng loss
        grad_norm = torch.norm(torch.stack([
            p.grad.norm(p=2) for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]), p=2)
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Tính epsilon(w) = rho * grad / ||grad||
                e_w = p.grad * scale.to(p)
                if group["adaptive"]:
                    # Với ASAM, scale theo độ lớn của w
                    e_w *= torch.norm(p.detach(), p=2)
                p.add_(e_w)  # cập nhật trọng số tạm thời w <- w + e_w
                self.state[p]["e_w"] = e_w
        if zero_grad:
            # Xóa gradient để chuẩn bị cho bước tiếp theo
            self.zero_grad()
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        # Bước 2: khôi phục trọng số và cập nhật bước tối ưu hóa cuối cùng
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])  # w <- w - e_w (quay về điểm ban đầu)
        # Thực hiện update thực sự bằng optimizer gốc trên gradient mới
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()
