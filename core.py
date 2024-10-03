import numpy as np
import torch
import torch.nn as nn
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Not Used
def generate_obstacle_circle(center, radius, num=12):
    theta = np.linspace(0, np.pi * 2, num=num, endpoint=False).reshape(-1, 1)
    unit_circle = np.concatenate([np.cos(theta), np.sin(theta)], axis=1)
    circle = np.array(center) + unit_circle * radius
    return circle

# Not used 
def generate_obstacle_rectangle(center, sides, num=12):
    a, b = sides  # side lengths
    n_side_1 = int(num // 2 * a / (a + b))
    n_side_2 = num // 2 - n_side_1
    n_side_3 = n_side_1
    n_side_4 = num - n_side_1 - n_side_2 - n_side_3

    # Define rectangle sides
    side_1 = np.concatenate([
        np.linspace(-a / 2, a / 2, n_side_1, endpoint=False).reshape(-1, 1),
        b / 2 * np.ones(n_side_1).reshape(-1, 1)], axis=1)
    side_2 = np.concatenate([
        a / 2 * np.ones(n_side_2).reshape(-1, 1),
        np.linspace(b / 2, -b / 2, n_side_2, endpoint=False).reshape(-1, 1)], axis=1)
    side_3 = np.concatenate([
        np.linspace(a / 2, -a / 2, n_side_3, endpoint=False).reshape(-1, 1),
        -b / 2 * np.ones(n_side_3).reshape(-1, 1)], axis=1)
    side_4 = np.concatenate([
        -a / 2 * np.ones(n_side_4).reshape(-1, 1),
        np.linspace(-b / 2, b / 2, n_side_4, endpoint=False).reshape(-1, 1)], axis=1)

    rectangle = np.concatenate([side_1, side_2, side_3, side_4], axis=0)
    rectangle = rectangle + np.array(center)
    return rectangle

# Identical to train.py
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

# Identical to train.py
class CBFNetwork(nn.Module):
    def __init__(self):
        super(CBFNetwork, self).__init__()
        self.conv1 = nn.Conv1d(6, 64, kernel_size=1)
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


# Identical to train.py
class ActionNetwork(nn.Module):
    def __init__(self):
        super(ActionNetwork, self).__init__()
        self.conv1 = nn.Conv1d(5, 64, kernel_size=1)
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


# Identical to train.py
def compute_neighbor_features(s, r, k, include_d_norm=False, indices=None):
    num_agents = s.size(0)
    s_diff = s.unsqueeze(1) - s.unsqueeze(0)  # [num_agents, num_agents, 4]
    distances = torch.norm(s_diff[:, :, :2], dim=2) + 1e-4  # [num_agents, num_agents]
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


# Identical to train.py
def dynamics(s, a):
    dsdt = torch.cat([s[:, 2:], a], dim=1)
    return dsdt


# Almost Same, has adding of 1e-6 for some reason
def ttc_dangerous_mask(s, r, ttc, indices):
    num_agents = s.size(0)
    s_diff = s.unsqueeze(1) - s.unsqueeze(0)  # [num_agents, num_agents, 4]
    s_diff = s_diff[torch.arange(num_agents).unsqueeze(1), indices]  # [num_agents, k, 4]
    x, y, vx, vy = s_diff[..., 0], s_diff[..., 1], s_diff[..., 2], s_diff[..., 3]   # [num_agents, k]

    # Avoid self-interactions by setting self-distances to a large value
    eye = torch.eye(num_agents, device=s.device)
    eye = eye[torch.arange(num_agents).unsqueeze(1), indices]
    x = x + eye
    y = y + eye
    alpha = vx ** 2 + vy ** 2
    beta = 2 * (x * vx + y * vy)
    gamma = x ** 2 + y ** 2 - r ** 2

    discriminant = beta ** 2 - 4 * alpha * gamma
    dist_dangerous = gamma < 0

    has_two_positive_roots = (discriminant > 0) & (gamma > 0) & (beta < 0)
    sqrt_discriminant = torch.sqrt(discriminant)
    root1 = (-beta - sqrt_discriminant) / (2 * alpha)
    root2 = (-beta + sqrt_discriminant) / (2 * alpha)
    root_less_than_ttc = ((root1 > 0) & (root1 < ttc)) | ((root2 > 0) & (root2 < ttc))
    has_root_less_than_ttc = has_two_positive_roots & root_less_than_ttc

    ttc_dangerous = dist_dangerous | has_root_less_than_ttc
    return ttc_dangerous  # Shape: [num_agents, k]



def barrier_loss(h, s, r, ttc, indices):
    mask = ttc_dangerous_mask(s, r, ttc, indices)
    h = h.squeeze(2)  # If h has shape [num_agents, k, 1]
    h = h.view(-1)  # Shape: [num_agents * k]
    mask = mask.view(-1)  # Shape: [num_agents * k]
    dang_h = h[mask]
    safe_h = h[~mask]
    loss_dang = torch.mean(torch.relu(dang_h + 1e-3)) if dang_h.numel() > 0 else 0
    loss_safe = torch.mean(torch.relu(-safe_h)) if safe_h.numel() > 0 else 0
    acc_dang = torch.mean((dang_h <= 0).float()) if dang_h.numel() > 0 else -1.0
    acc_safe = torch.mean((safe_h > 0).float()) if safe_h.numel() > 0 else -1.0
    return loss_dang, loss_safe, acc_dang, acc_safe


def derivative_loss(h, s, a, cbf_net, alpha, indices):
    s_next = s + dynamics(s, a) * config.TIME_STEP
    neighbor_features, _ = compute_neighbor_features(
        s_next, config.DIST_MIN_THRES, config.TOP_K, include_d_norm=True, indices=indices)
    h_next = cbf_net(neighbor_features)
    deriv = (h_next - h) + config.TIME_STEP * alpha * h
    deriv = deriv.view(-1)
    dang_mask = ttc_dangerous_mask(s, config.DIST_MIN_THRES, config.TIME_TO_COLLISION, indices)
    dang_mask = dang_mask.view(-1)
    dang_deriv = deriv[dang_mask]
    safe_deriv = deriv[~dang_mask]
    loss_dang_deriv = torch.mean(torch.relu(-dang_deriv + 1e-3)) if dang_deriv.numel() > 0 else 0
    loss_safe_deriv = torch.mean(torch.relu(-safe_deriv)) if safe_deriv.numel() > 0 else 0
    acc_dang_deriv = torch.mean((dang_deriv >= 0).float()) if dang_deriv.numel() > 0 else -1.0
    acc_safe_deriv = torch.mean((safe_deriv >= 0).float()) if safe_deriv.numel() > 0 else -1.0
    return loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv


def action_loss(a, s, g):
    state_gain = torch.tensor(np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3), dtype=torch.float32).to(s.device)
    s_ref = torch.cat([s[:, :2] - g, s[:, 2:]], dim=1)
    action_ref = s_ref @ state_gain.T
    action_ref_norm = torch.sum(action_ref ** 2, dim=1)
    action_net_norm = torch.sum(a ** 2, dim=1)
    norm_diff = torch.abs(action_net_norm - action_ref_norm)
    loss = torch.mean(norm_diff)
    return loss
