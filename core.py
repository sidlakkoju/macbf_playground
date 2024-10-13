import numpy as np
import torch
import torch.nn as nn
import config
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def generate_data_np(num_agents, dist_min_thres):
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

# Torch Version
def generate_data_torch(num_agents, dist_min_thres, device):
    side_length = torch.sqrt(torch.tensor(max(1.0, num_agents / 8.0), device=device))
    states = torch.zeros((num_agents, 4), dtype=torch.float32, device=device)
    goals = torch.zeros((num_agents, 2), dtype=torch.float32, device=device)
    i = 0
    while i < num_agents:
        candidate = torch.rand(2, device=device) * side_length
        if i > 0:
            dist_min = torch.norm(states[:i, :2] - candidate, dim=1).min()
            if dist_min <= dist_min_thres:
                continue
        states[i, :2] = candidate
        i += 1
    i = 0
    while i < num_agents:
        candidate = (torch.rand(2, device=device) - 0.5) + states[i, :2]
        if i > 0:
            dist_min = torch.norm(goals[:i] - candidate, dim=1).min()
            if dist_min <= dist_min_thres:
                continue
        goals[i] = candidate
        i += 1
    return states, goals


# Input: [num_agents, k, 6], 6: [del_x, del_y , del_vx, del_vy, identity, d_norm]
# Output: [num_agents, k, 1], 1: h
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


# Input: [num_agents, k, 5], 5: [del_x, del_y , del_vx, del_vy, identity]
# Output: [num_agents, 2], 2: [a_x, a_y]
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
    return neighbor_features, indices   # [num_agents, k, 6], [num_agents, k], 6 -> [del_x, del_y , del_vx, del_vy, identity, d_norm]


def dynamics(s, a):
    dsdt = torch.cat([s[:, 2:], a], dim=1)
    return dsdt


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
    return ttc_dangerous  # Shape: [num_agents, k], k: True if dangerous


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


def action_loss(a, s, g, state_gain):
    s_ref = torch.cat([s[:, :2] - g, s[:, 2:]], dim=1)
    action_ref = s_ref @ state_gain.T
    action_ref_norm = torch.sum(action_ref ** 2, dim=1)
    action_net_norm = torch.sum(a ** 2, dim=1)
    norm_diff = torch.abs(action_net_norm - action_ref_norm)
    loss = torch.mean(norm_diff)
    return loss


# Never used 
def action_loss_np(a, s, g):
    state_gain = torch.tensor(np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3), dtype=torch.float32).to(s.device)
    s_ref = torch.cat([s[:, :2] - g, s[:, 2:]], dim=1)
    action_ref = s_ref @ state_gain.T
    action_ref_norm = torch.sum(action_ref ** 2, dim=1)
    action_net_norm = torch.sum(a ** 2, dim=1)
    norm_diff = torch.abs(action_net_norm - action_ref_norm)
    loss = torch.mean(norm_diff)
    return loss


def compute_neighbor_features_with_wall_agents(s, wall_agents, r, k, include_d_norm=False, indices=None):
    num_agents = s.size(0)
    num_wall_agents = wall_agents.size(0)

    s_diff_agents = s.unsqueeze(1) - s.unsqueeze(0)  # Shape: [num_agents, num_agents, 4]
    distances_agents = torch.norm(s_diff_agents[:, :, :2], dim=2) + 1e-4  # Shape: [num_agents, num_agents]

    # Exclude self-interactions by setting self-distances to a large value
    eye = torch.eye(num_agents, device=s.device)
    distances_agents = distances_agents + eye * 1e6  # Large value to exclude self

    # Select top k closest agents
    if indices is None:
        _, indices = torch.topk(-distances_agents, k=k, dim=1)  # Negative for topk
    neighbor_features_agents = s_diff_agents[torch.arange(num_agents).unsqueeze(1), indices]

    # Add the 'eye' variable for agents
    eye_agents = eye[torch.arange(num_agents).unsqueeze(1), indices].unsqueeze(2)  # Shape: [num_agents, k, 1]
    neighbor_features_agents = torch.cat([neighbor_features_agents, eye_agents], dim=2)

    if include_d_norm:
        d_norm_agents = distances_agents[torch.arange(num_agents).unsqueeze(1), indices].unsqueeze(2) - r
        neighbor_features_agents = torch.cat([neighbor_features_agents, d_norm_agents], dim=2)

    # Compute s_diff between agents and obstacles
    s_diff_obstacles = s.unsqueeze(1) - wall_agents.unsqueeze(0)  # Shape: [num_agents, num_obstacles, 4]
    distances_obstacles = torch.norm(s_diff_obstacles[:, :, :2], dim=2) + 1e-4  # Shape: [num_agents, num_obstacles]

    # For obstacles, 'eye' variable is zero
    eye_obstacles = torch.zeros(num_agents, num_wall_agents, 1, device=s.device)  # Shape: [num_agents, num_obstacles, 1]
    neighbor_features_obstacles = torch.cat([s_diff_obstacles, eye_obstacles], dim=2)

    if include_d_norm:
        d_norm_obstacles = distances_obstacles.unsqueeze(2) - r  # Shape: [num_agents, num_obstacles, 1]
        neighbor_features_obstacles = torch.cat([neighbor_features_obstacles, d_norm_obstacles], dim=2)

    # Concatenate agent neighbors and obstacle neighbors
    neighbor_features = torch.cat([neighbor_features_agents, neighbor_features_obstacles], dim=1)  # Shape: [num_agents, k + num_obstacles, features]

    # Return neighbor features and indices (indices correspond to agent neighbors)
    return neighbor_features, indices


def ttc_dangerous_mask_obs(s, r, ttc, indices, neighbor_features):
    """
    s: agent states, shape [num_agents, 4]
    r: safe distance threshold
    ttc: time-to-collision threshold
    indices: indices of agent neighbors (from topk)
    neighbor_features: neighbor features including obstacles, shape [num_agents, k + num_obstacles, feature_dim]
    """
    # Extract features
    x = neighbor_features[..., 0]  # [num_agents, num_neighbors]
    y = neighbor_features[..., 1]
    vx = neighbor_features[..., 2]
    vy = neighbor_features[..., 3]
    eye = neighbor_features[..., 4]
    # d_norm = neighbor_features[..., 5] 

    # Avoid self-interactions by adding eye to x and y
    x = x + eye
    y = y + eye

    # Compute quadratic equation coefficients
    alpha = vx ** 2 + vy ** 2
    beta = 2 * (x * vx + y * vy)
    gamma = x ** 2 + y ** 2 - r ** 2

    # Compute discriminant and handle negative values
    discriminant = beta ** 2 - 4 * alpha * gamma
    discriminant = torch.clamp(discriminant, min=0.0)

    dist_dangerous = gamma < 0

    has_two_positive_roots = (discriminant > 0) & (gamma > 0) & (beta < 0)
    sqrt_discriminant = torch.sqrt(discriminant)
    alpha_safe = torch.where(alpha == 0, torch.full_like(alpha, 1e-6), alpha)

    root1 = (-beta - sqrt_discriminant) / (2 * alpha_safe)
    root2 = (-beta + sqrt_discriminant) / (2 * alpha_safe)

    root_less_than_ttc = ((root1 > 0) & (root1 < ttc)) | ((root2 > 0) & (root2 < ttc))
    has_root_less_than_ttc = has_two_positive_roots & root_less_than_ttc

    ttc_dangerous = dist_dangerous | has_root_less_than_ttc

    return ttc_dangerous  # Shape: [num_agents, k + num_obstacles]


def generate_social_mini_game_data():
    """
    Generates initial states and goals for a social mini-game with two agents and a wall.
    The wall is represented separately with its own states and goals.

    Returns:
        agent_states: np.array of shape (num_agents, 4), where each row is [x, y, vx, vy]
        agent_goals: np.array of shape (num_agents, 2), where each row is [goal_x, goal_y]
        wall_states: np.array of shape (num_wall_agents, 4)
        wall_goals: np.array of shape (num_wall_agents, 2)
    """
    # Map dimensions
    x_min, x_max = 0.0, 10.0
    y_min, y_max = 0.0, 10.0

    # Wall parameters
    wall_x = 5.0  # Wall is vertical at x=5
    hole_y_center = 5.0  # Hole is centered at y=5
    hole_height = 1.0  # Height of the hole
    # hole_height = config.DIST_MIN_THRES
    hole_y_min = hole_y_center - hole_height / 2
    hole_y_max = hole_y_center + hole_height / 2

    # Number of agents
    num_agents = 2

    # Agent positions and goals
    agent_states = np.zeros((num_agents, 4), dtype=np.float32)
    agent_goals = np.zeros((num_agents, 2), dtype=np.float32)

    agent_offset = 2.0  # Distance from the wall
    agent_states[0, :2] = [wall_x - agent_offset - 1, y_max - agent_offset + 1]  # Agent 1 in upper-left quadrant
    agent_states[1, :2] = [wall_x - agent_offset - 1, y_min + agent_offset - 1]  # Agent 2 in lower-left quadrant

    # Initial velocities set to zero
    agent_states[:, 2:] = 0.0

    # Goal positions (symmetric positions on the opposite side)
    agent_goals[0] = [wall_x + agent_offset, y_min + (agent_offset + 0.0)]       # Agent 1 goal in lower-right quadrant
    agent_goals[1] = [wall_x + agent_offset, y_max - (agent_offset + 0.0)]       # Agent 2 goal in upper-right quadrant

    

    # Wall representation as static agents
    # We will create wall agents along x=wall_x, excluding the hole
    wall_resolution = config.DIST_MIN_THRES  # Distance between wall agents
    wall_y_positions = np.arange(y_min, y_max + wall_resolution, wall_resolution)
    # Exclude positions within the hole
    wall_y_positions = wall_y_positions[(wall_y_positions < hole_y_min) | (wall_y_positions > hole_y_max)]

    num_wall_agents = len(wall_y_positions)
    wall_states = np.zeros((num_wall_agents, 4), dtype=np.float32)
    wall_goals = np.zeros((num_wall_agents, 2), dtype=np.float32)

    wall_states[:, 0] = wall_x
    wall_states[:, 1] = wall_y_positions
    wall_states[:, 2:] = 0.0  # Wall agents are static

    wall_goals[:, 0] = wall_x
    wall_goals[:, 1] = wall_y_positions  # Goal positions same as initial positions

    

    return agent_states, agent_goals, wall_states, wall_goals


def plot_single_state_with_original(state, target, safety, original_state, agent_size=100):
    """
    Plot a single frame given a state vector, target points, safety metrics, 
    and an original state vector for consistent scaling.

    Args:
        state: np.array of shape (n_agents, 4) where each row is [x, y, vx, vy]
        target: np.array of target points of shape (n_targets, 2)
        safety: np.array of safety values of shape (n_agents,)
        original_state: The original state vector (n_agents, 4) used for consistent scaling
        agent_size: Size of the agents for visualization
    """
    vis_range = max(1, np.amax(np.abs(original_state[:, :2])))
    state_vis = state[:, :2] / vis_range
    target_vis = target / vis_range
    plt.scatter(state_vis[:, 0], state_vis[:, 1], color='darkorange',
                s=agent_size, label='Agent', alpha=0.6)
    plt.scatter(target_vis[:, 0], target_vis[:, 1], color='deepskyblue',
                s=agent_size, label='Target', alpha=0.6)
    plt.scatter(state_vis[safety < 1, 0], state_vis[safety < 1, 1],
                color='red', s=agent_size, label='Collision', alpha=0.9)
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    ax = plt.gca()
    for side in ax.spines.keys():
        ax.spines[side].set_linewidth(2)
        ax.spines[side].set_color('grey')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.legend(loc='upper right', fontsize=14)


def plot_single_state_with_wall_separate(agent_state, agent_goal, wall_state, wall_goal,
                                         safety, original_state, action=None, action_res=None, action_opt=None, agent_size=100):
    """
    Plot a single frame given agent states, agent goals, wall states, wall goals,
    safety metrics, and an original state vector for consistent scaling. 
    Additionally, print action values on the plot with better spacing.

    Args:
        agent_state: np.array of shape (num_agents, 4)
        agent_goal: np.array of shape (num_agents, 2)
        wall_state: np.array of shape (num_wall_agents, 4)
        wall_goal: np.array of shape (num_wall_agents, 2)
        safety: np.array of safety values of shape (num_agents,)
        original_state: The original agent states used for consistent scaling
        action: np.array of shape (num_agents, 2) or None
        action_res: np.array of shape (num_agents, 2) or None
        action_opt: np.array of shape (num_agents, 2) or None
        agent_size: Size of the agents for visualization
    """
    # Combine agent and wall states for scaling
    combined_states = np.vstack((agent_state, wall_state))

    # vis_range = max(1, np.amax(np.abs(original_state[:, :2])))
    vis_range = np.float32(9.0)
    agent_state_vis = agent_state[:, :2] / vis_range
    agent_goal_vis = agent_goal / vis_range
    wall_state_vis = wall_state[:, :2] / vis_range

    # Plot wall agents
    plt.scatter(wall_state_vis[:, 0], wall_state_vis[:, 1], color='grey',
                s=agent_size, label='Wall', alpha=0.6)

    # Plot agent states
    plt.scatter(agent_state_vis[:, 0], agent_state_vis[:, 1], color='darkorange',
                s=agent_size, label='Agent', alpha=0.6)

    # Plot agent goals
    plt.scatter(agent_goal_vis[:, 0], agent_goal_vis[:, 1], color='deepskyblue',
                s=agent_size, label='Target', alpha=0.6)

    # Plot collision states where safety < 1
    collision_indices = np.where(safety < 1)[0]
    if collision_indices.size > 0:
        plt.scatter(agent_state_vis[collision_indices, 0], agent_state_vis[collision_indices, 1],
                    color='red', s=agent_size, label='Collision', alpha=0.9)

    # Plot actions if provided
    if action is not None:
        plt.quiver(agent_state_vis[:, 0], agent_state_vis[:, 1], action[:, 0], action[:, 1], color='green', scale=1, label='Action')
        for i in range(agent_state_vis.shape[0]):
            plt.text(agent_state_vis[i, 0] - 0.05, agent_state_vis[i, 1] + 0.05, f'{action[i]}', color='green', fontsize=8)

    if action_res is not None:
        plt.quiver(agent_state_vis[:, 0], agent_state_vis[:, 1], action_res[:, 0], action_res[:, 1], color='blue', scale=1, label='Action Res')
        for i in range(agent_state_vis.shape[0]):
            plt.text(agent_state_vis[i, 0] - 0.05, agent_state_vis[i, 1] - 0.05, f'{action_res[i]}', color='blue', fontsize=8)

    if action_opt is not None:
        plt.quiver(agent_state_vis[:, 0], agent_state_vis[:, 1], action_opt[:, 0], action_opt[:, 1], color='purple', scale=1, label='Action Opt')
        for i in range(agent_state_vis.shape[0]):
            plt.text(agent_state_vis[i, 0] - 0.05, agent_state_vis[i, 1] - 0.1, f'{action_opt[i]}', color='purple', fontsize=8)

    # Set plot limits and appearance
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.legend()

    # Customize axis spines
    ax = plt.gca()
    for side in ax.spines.keys():
        ax.spines[side].set_linewidth(2)
        ax.spines[side].set_color('grey')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.show()


'''
NOT USED
Adapt to create solid walls and obstacles 
Incorporate SMG creation
'''
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