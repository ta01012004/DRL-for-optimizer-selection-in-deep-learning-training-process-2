.# train.py
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

# Import từ file common.py
from common import (
    device,
    build_resnet_model,
    train_one_epoch,
    validate
)

# Import từ file dqn_agent.py
from dqn_agent import (
    QNetwork,
    PrioritizedReplayBuffer,
    select_action,
    compute_td_loss,
    soft_update,
    compute_reward
)

def train_rl_agent(train_loader, val_loader, num_epochs=30, gamma=0.9,
                   w_acc=1.0, w_loss=1.0, w_time=0.1,
                   epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.03,
                   tau=0.01, batch_size=32):
    """
    Vòng lặp chính để huấn luyện RL chọn optimizer cho ResNet.
    """
    # 1) Xây dựng ResNet
    num_classes = 10  # tùy theo dataset
    model = build_resnet_model(num_classes=num_classes, pretrained=False).to(device)

    # 2) Khởi tạo 2 optimizer (hành động 0: SGD, 1: Adam)
    optimizer_sgd  = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
    optimizers = [optimizer_sgd, optimizer_adam]

    # 3) Khởi tạo Double DQN
    state_dim  = 6  # (ví dụ) [loss, acc, ... + epoch_ratio, last_action,...]
    action_dim = 2  # 0: SGD, 1: Adam
    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    dqn_optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = PrioritizedReplayBuffer(capacity=1000, alpha=0.6)

    # 4) Lưu trữ history
    reward_history = []
    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    time_history = []
    dqn_loss_history = []

    # 5) Đo validation ban đầu
    init_val_acc = validate(model, val_loader, device)
    prev_acc  = init_val_acc
    prev_loss = None
    epsilon = epsilon_start

    # 6) Bắt đầu vòng lặp epoch
    for epoch in range(1, num_epochs+1):
        start_time = time.time()

        # Xây dựng state cũ
        st_loss = 0.0 if prev_loss is None else prev_loss
        state = np.array([st_loss, prev_acc, 0.0, 0.0, epoch/num_epochs, 0], dtype=np.float32)

        # Epsilon-greedy chọn action (0 hoặc 1)
        action = select_action(state, policy_net, epsilon, device=device)
        epsilon = max(epsilon_end, epsilon - epsilon_decay)

        # Lấy optimizer tương ứng
        optimizer = optimizers[action]

        # Huấn luyện 1 epoch
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device)
        val_acc = validate(model, val_loader, device)

        epoch_time = time.time() - start_time
        # Tính reward
        reward = compute_reward(prev_acc, prev_loss, train_acc, train_loss, epoch_time,
                                w_acc, w_loss, w_time)

        # Lưu lại lịch sử
        reward_history.append(reward)
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)
        val_acc_history.append(val_acc)
        time_history.append(epoch_time)

        # Tạo next_state
        next_state = np.array([train_loss, train_acc, 0.0, 0.0, epoch/num_epochs, action], dtype=np.float32)
        done = (epoch == num_epochs)

        # Đưa vào Replay Buffer
        memory.push((state, action, reward, next_state, float(done)))

        # Huấn luyện Double DQN nếu đủ batch
        batch_data = memory.sample(batch_size, beta=0.4, device=device)
        if batch_data[0] is not None:
            loss_dqn, td_errors = compute_td_loss(
                batch_data, 
                policy_net, 
                target_net, 
                dqn_optimizer, 
                gamma, 
                batch_data[-1],  # weights
                device
            )
            # Cập nhật priority
            memory.update_priorities(batch_data[5], torch.abs(td_errors) + 1e-6)

            # Soft update target
            soft_update(target_net, policy_net, tau)
            dqn_loss_history.append(loss_dqn.item())
        else:
            dqn_loss_history.append(0.0)

        # Cập nhật biến cho vòng sau
        prev_acc = train_acc
        prev_loss = train_loss

        # In ra kết quả chi tiết
        print(f"[Epoch {epoch}/{num_epochs}] "
              f"Action={action} (0=SGD,1=Adam) | "
              f"TrainAcc={train_acc:.4f} | ValAcc={val_acc:.4f} | "
              f"Reward={reward:.4f} | DQN_Loss={dqn_loss_history[-1]:.4f} | "
              f"Time={epoch_time:.2f}s")

    # Kết thúc, trả về dict history
    return {
        'model': model,
        'policy_net': policy_net,
        'target_net': target_net,
        'reward_history': reward_history,
        'train_acc_history': train_acc_history,
        'train_loss_history': train_loss_history,
        'val_acc_history': val_acc_history,
        'time_history': time_history,
        'dqn_loss_history': dqn_loss_history
    }


# -----------------------------
# Code chạy chính (main)
# -----------------------------
if __name__ == "__main__":
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR10

    # Chuẩn bị dataset, dataloader
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    train_dataset = CIFAR10(root='./data', train=True,  download=True, transform=transform)
    val_dataset   = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Gọi hàm train RL Agent
    results = train_rl_agent(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=200,        # train thử 5 epoch
        gamma=0.9,
        w_acc=1.0, 
        w_loss=1.0, 
        w_time=0.1,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.05,
        tau=0.01,
        batch_size=32
    )

    # Vẽ biểu đồ
    epochs = range(1, len(results['reward_history'])+1)

    plt.figure(figsize=(14,10))
    # Reward
    plt.subplot(2,2,1)
    plt.plot(epochs, results['reward_history'], label='Reward')
    plt.xlabel('Epoch'); plt.ylabel('Reward'); plt.title('Reward per Epoch')
    plt.legend()

    # Train Loss
    plt.subplot(2,2,2)
    plt.plot(epochs, results['train_loss_history'], label='Train Loss', color='orange')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Train Loss')
    plt.legend()

    # Train/Val Acc
    plt.subplot(2,2,3)
    plt.plot(epochs, results['train_acc_history'], label='Train Acc')
    plt.plot(epochs, results['val_acc_history'], label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy per Epoch')
    plt.legend()

    # DQN Loss
    plt.subplot(2,2,4)
    plt.plot(epochs, results['dqn_loss_history'], label='DQN Loss', color='red')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('DQN Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
