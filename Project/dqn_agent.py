# dqn_agent.py
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.position = 0
        self.alpha = alpha

    def push(self, experience):
        max_prio = max(self.priorities) if self.priorities else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4, device='cpu'):
        if len(self.buffer) < batch_size:
            return None, None, None, None, None, None, None

        probs = np.array(self.priorities, dtype=np.float32) ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states      = torch.tensor(states, dtype=torch.float32, device=device)
        actions     = torch.tensor(actions, dtype=torch.int64, device=device)
        rewards     = torch.tensor(rewards, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
        dones       = torch.tensor(dones, dtype=torch.float32, device=device)

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        for i, td in zip(indices, td_errors):
            self.priorities[i] = float(abs(td.item()) + 1e-6)

    def __len__(self):
        return len(self.buffer)


def select_action(state, policy_net, epsilon, device='cpu'):
    if random.random() < epsilon:
        action_dim = policy_net.fc3.out_features
        return random.randint(0, action_dim - 1)
    else:
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = policy_net(s)
            action = q_values.argmax(dim=1).item()
        return action


def compute_td_loss(batch_data, policy_net, target_net, optimizer, gamma, weights, device='cpu'):
    states, actions, rewards, next_states, dones, indices, _ = batch_data

    with torch.no_grad():
        next_action = policy_net(next_states).argmax(dim=1, keepdim=True)
        next_q = target_net(next_states).gather(1, next_action).squeeze(1)
        target_value = rewards + gamma * next_q * (1 - dones)

    current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    td_errors = target_value - current_q

    loss_each = F.smooth_l1_loss(current_q, target_value, reduction='none')
    loss = (loss_each * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, td_errors


def soft_update(target_net, policy_net, tau=0.01):
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)


def compute_reward(prev_acc, prev_loss, curr_acc, curr_loss, epoch_time, 
                   w_acc=1.0, w_loss=1.0, w_time=0.1):
    if prev_loss is None:
        acc_gain  = curr_acc - (prev_acc if prev_acc is not None else 0.0)
        loss_gain = 0.0
    else:
        acc_gain  = curr_acc - prev_acc
        loss_gain = prev_loss - curr_loss

    reward = w_acc*acc_gain + w_loss*loss_gain - w_time*epoch_time
    return reward  

# train.py
import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from common import (
    device,
    build_resnet_model,
    train_one_epoch,
    validate
)

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
    num_classes = 10
    model = build_resnet_model(num_classes=num_classes, pretrained=False).to(device)

    optimizer_sgd  = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer_adam = optim.Adam(model.parameters(), lr=0.001)
    optimizers = [optimizer_sgd, optimizer_adam]

    state_dim  = 6
    action_dim = 2
    policy_net = QNetwork(state_dim, action_dim).to(device)
    target_net = QNetwork(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    dqn_optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    memory = PrioritizedReplayBuffer(capacity=1000, alpha=0.6)

    reward_history = []
    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    time_history = []
    dqn_loss_history = []

    init_val_acc = validate(model, val_loader, device)
    prev_acc  = init_val_acc
    prev_loss = None
    epsilon = epsilon_start

    for epoch in range(1, num_epochs+1):
        start_time = time.time()

        st_loss = 0.0 if prev_loss is None else prev_loss
        state = np.array([st_loss, prev_acc, 0.0, 0.0, epoch/num_epochs, 0], dtype=np.float32)

        action = select_action(state, policy_net, epsilon, device=device)
        epsilon = max(epsilon_end, epsilon - epsilon_decay)

        optimizer = optimizers[action]
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device)
        val_acc = validate(model, val_loader, device)

        epoch_time = time.time() - start_time
        reward = compute_reward(prev_acc, prev_loss, train_acc, train_loss, epoch_time,
                                w_acc, w_loss, w_time)

        reward_history.append(reward)
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)
        val_acc_history.append(val_acc)
        time_history.append(epoch_time)

        next_state = np.array([train_loss, train_acc, 0.0, 0.0, epoch/num_epochs, action], dtype=np.float32)
        done = (epoch == num_epochs)

        memory.push((state, action, reward, next_state, float(done)))

        batch_data = memory.sample(batch_size, beta=0.4, device=device)
        if batch_data[0] is not None:
            loss_dqn, td_errors = compute_td_loss(
                batch_data, 
                policy_net, 
                target_net, 
                dqn_optimizer, 
                gamma, 
                batch_data[-1], 
                device
            )
            memory.update_priorities(batch_data[5], torch.abs(td_errors) + 1e-6)
            soft_update(target_net, policy_net, tau)
            dqn_loss_history.append(loss_dqn.item())
        else:
            dqn_loss_history.append(0.0)

        prev_acc = train_acc
        prev_loss = train_loss

        print(f"[Epoch {epoch}/{num_epochs}] "
              f"Action={action} (0=SGD,1=Adam) | "
              f"TrainAcc={train_acc:.4f} | ValAcc={val_acc:.4f} | "
              f"Reward={reward:.4f} | DQN_Loss={dqn_loss_history[-1]:.4f} | "
              f"Time={epoch_time:.2f}s")

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
