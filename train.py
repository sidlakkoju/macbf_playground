import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys

# At the beginning of the script, add:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration parameters (same as before)
class Config:
    TIME_STEP = 0.02
    TRAIN_STEPS = 20000
    DISPLAY_STEPS = 100
    INNER_LOOPS = 10
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    ALPHA_CBF = 1.0
    DIST_MIN_THRES = 0.1
    TIME_TO_COLLISION = 3.0
    OBS_RADIUS = 1.0
    ADD_NOISE_PROB = 0.1
    NOISE_SCALE = 0.1
    TOP_K = 5

config = Config()

# Data generation function (same as before)
def generate_data(num_agents, dist_min_thres):
    side_length = np.sqrt(max(1.0, num_agents / 8.0))
    states = np.zeros((num_agents, 4), dtype=np.float32)
    goals = np.zeros((num_agents, 2), dtype=np.float32)
    i = 0
    while i < num_agents:
        candidate = np.random.uniform(0, side_length, size=(2,))
        if i > 0:
            dist_min = np.linalg.norm(states[:i, :2] - candidate, axis=1).min()
            if dist_min <= dist_min_thres:
                continue
        states[i, :2] = candidate
        i += 1
    i = 0
    while i < num_agents:
        candidate = np.random.uniform(-0.5, 0.5, size=(2,)) + states[i, :2]
        if i > 0:
            dist_min = np.linalg.norm(goals[:i] - candidate, axis=1).min()
            if dist_min <= dist_min_thres:
                continue
        goals[i] = candidate
        i += 1
    return states, goals

# Control Barrier Function (CBF) Network
class CBFNetwork(nn.Module):
    def __init__(self):
        super(CBFNetwork, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1)  # Input channels changed to 6
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, features, neighbors]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        h = self.conv4(x)
        h = h.permute(0, 2, 1)  # [batch, neighbors, 1]
        return h

# Action Network
class ActionNetwork(nn.Module):
    def __init__(self):
        super(ActionNetwork, self).__init__()
        self.conv1 = nn.Conv1d(5, 64, kernel_size=1)  # Input channels changed to 5
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.fc1 = nn.Linear(128 + 4, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)

    def forward(self, s, g, neighbor_features):
        x = neighbor_features.permute(0, 2, 1)  # [batch, features, neighbors]
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max(x, dim=2)[0]  # Max over neighbors
        state_goal_diff = torch.cat([s[:, :2] - g, s[:, 2:]], dim=1)
        x = torch.cat([x, state_goal_diff], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        gains = 2.0 * torch.sigmoid(self.fc4(x)) + 0.2
        k1, k2, k3, k4 = torch.chunk(gains, 4, dim=1)
        zeros = torch.zeros_like(k1)
        gain_x = -torch.cat([k1, zeros, k2, zeros], dim=1)
        gain_y = -torch.cat([zeros, k3, zeros, k4], dim=1)
        state = torch.cat([s[:, :2] - g, s[:, 2:]], dim=1)
        a_x = torch.sum(state * gain_x, dim=1, keepdim=True)
        a_y = torch.sum(state * gain_y, dim=1, keepdim=True)
        a = torch.cat([a_x, a_y], dim=1)
        return a

# compute_neighbor_features function
def compute_neighbor_features(s, r, k, include_d_norm=False, indices=None):
    num_agents = s.size(0)
    s_diff = s.unsqueeze(1) - s.unsqueeze(0)  # [num_agents, num_agents, 4]
    distances = torch.norm(s_diff[:, :, :2], dim=2) + 1e-6  # [num_agents, num_agents]
    if indices is None:
        _, indices = torch.topk(-distances, k=k, dim=1)
    neighbor_features = s_diff[torch.arange(num_agents).unsqueeze(1), indices]
    eye = torch.eye(num_agents, device=s.device)
    eye = eye[torch.arange(num_agents).unsqueeze(1), indices].unsqueeze(2)
    neighbor_features = torch.cat([neighbor_features, eye], dim=2)
    if include_d_norm:
        d_norm = distances[torch.arange(num_agents).unsqueeze(1), indices].unsqueeze(2) - r
        neighbor_features = torch.cat([neighbor_features, d_norm], dim=2)
    return neighbor_features, indices

# ttc_dangerous_mask function
def ttc_dangerous_mask(s, r, ttc, indices):
    num_agents = s.size(0)
    s_diff = s.unsqueeze(1) - s.unsqueeze(0)  # [num_agents, num_agents, 4]
    s_diff = s_diff[torch.arange(num_agents).unsqueeze(1), indices]  # Select top k neighbors
    x, y, vx, vy = s_diff.split(1, dim=2)
    alpha = vx ** 2 + vy ** 2
    beta = 2 * (x * vx + y * vy)
    gamma = x ** 2 + y ** 2 - r ** 2
    discriminant = beta ** 2 - 4 * alpha * gamma
    mask = (gamma < 0) | ((discriminant > 0) & (gamma > 0) & (beta < 0))
    return mask.squeeze(2)  # Shape: [num_agents, k]

# barrier_loss function
def barrier_loss(h, s, r, ttc, indices):
    mask = ttc_dangerous_mask(s, r, ttc, indices)
    h = h.squeeze(2)  # If h has shape [num_agents, k, 1]
    h = h.view(-1)  # Shape: [num_agents * k]
    mask = mask.view(-1)  # Shape: [num_agents * k]
    dang_h = h[mask]
    safe_h = h[~mask]
    loss_dang = torch.mean(torch.relu(dang_h + 1e-3)) if dang_h.numel() > 0 else 0
    loss_safe = torch.mean(torch.relu(-safe_h)) if safe_h.numel() > 0 else 0
    return loss_dang + loss_safe

# derivative_loss function
def derivative_loss(h, s, a, cbf_net, alpha, indices):
    s_next = s + dynamics(s, a) * config.TIME_STEP
    neighbor_features, _ = compute_neighbor_features(
        s_next, config.DIST_MIN_THRES, config.TOP_K, include_d_norm=True, indices=indices)
    h_next = cbf_net(neighbor_features)
    deriv = (h_next - h) + config.TIME_STEP * alpha * h
    loss_deriv = torch.mean(torch.relu(-deriv.view(-1) + 1e-3))
    return loss_deriv


def action_loss(a, s, g):
    state_gain = torch.tensor(np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3), dtype=torch.float32).to(s.device)
    s_ref = torch.cat([s[:, :2] - g, s[:, 2:]], dim=1)
    action_ref = s_ref @ state_gain.T
    loss = torch.mean(torch.abs(torch.norm(a, dim=1) - torch.norm(action_ref, dim=1)))
    return loss

def dynamics(s, a):
    dsdt = torch.cat([s[:, 2:], a], dim=1)
    return dsdt

def train(num_agents):
    cbf_net = CBFNetwork().to(device)
    action_net = ActionNetwork().to(device)
    optimizer = optim.Adam(list(cbf_net.parameters()) + list(action_net.parameters()), lr=config.LEARNING_RATE)
    for step in range(config.TRAIN_STEPS):
        s_np, g_np = generate_data(num_agents, config.DIST_MIN_THRES)
        s = torch.tensor(s_np, dtype=torch.float32).to(device)
        g = torch.tensor(g_np, dtype=torch.float32).to(device)
        for _ in range(config.INNER_LOOPS):
            optimizer.zero_grad()
            # For CBF Network
            neighbor_features_cbf, indices = compute_neighbor_features(
                s, config.DIST_MIN_THRES, config.TOP_K, include_d_norm=True)
            h = cbf_net(neighbor_features_cbf)
            # For Action Network
            neighbor_features_action, _ = compute_neighbor_features(
                s, config.DIST_MIN_THRES, config.TOP_K, include_d_norm=False, indices=indices)
            a = action_net(s, g, neighbor_features_action)
            loss_b = barrier_loss(h, s, config.DIST_MIN_THRES, config.TIME_TO_COLLISION, indices)
            loss_d = derivative_loss(h, s, a, cbf_net, config.ALPHA_CBF, indices)
            loss_a = action_loss(a, s, g)
            loss = 10 * (2 * loss_b + 2 * loss_d + 0.01 * loss_a)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                s = s + dynamics(s, a) * config.TIME_STEP

        if step % config.DISPLAY_STEPS == 0:
            print(f"Step: {step}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    num_agents = 10  # Example number of agents
    train(num_agents)
