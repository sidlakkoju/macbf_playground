import os
import re
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
from scipy import interpolate
import config
from core import *
from vis import *
from a_star import AStarPlanner
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, FFMpegWriter
from datetime import datetime
import random
import math

BASE_DIR = 'smg_testing'
LOAD_DIR = 'checkpoints_barrier_eval'

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


def extract_existing_model_path(directory=BASE_DIR):
    max_cbf = None
    max_action = None
    for filename in os.listdir(directory):
        match_cbf = re.search(r"cbf_net_step_(\d+)", filename)
        match_action = re.search(r"action_net_step_(\d+)", filename)
        if match_cbf:
            number = int(match_cbf.group(1))
            max_cbf = max(max_cbf, number) if max_cbf is not None else number
        if match_action:
            number = int(match_action.group(1))
            max_action = max(max_action, number) if max_action is not None else number
    return max_cbf, max_action


def load_model(model, model_name, step, base_dir=BASE_DIR):
    checkpoint_path = f"{base_dir}/{model_name}_step_{step}.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading {model_name} from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"No checkpoint found for {model_name} at step {step}")


def save_model(model, model_name, step, base_dir=BASE_DIR):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    torch.save(model.state_dict(), f"{base_dir}/{model_name}_step_{step}.pth")


def train():
    cbf_net = CBFNetwork().to(device)
    action_net = ActionNetwork().to(device)
    optimizer_h = optim.Adam(cbf_net.parameters(), lr=config.LEARNING_RATE)
    optimizer_a = optim.Adam(action_net.parameters(), lr=config.LEARNING_RATE)
    
    # Initialize state_gain once
    sqrt_3 = torch.sqrt(torch.tensor(3.0, device=device))
    state_gain = torch.tensor(
        [[1.0, 0.0, sqrt_3.item(), 0.0],
         [0.0, 1.0, 0.0, sqrt_3.item()]],
        dtype=torch.float32,
        device=device
    )

    max_cbf, max_action = extract_existing_model_path()
    last_saved = 0

    if max_cbf is not None:
        load_model(cbf_net, "cbf_net", max_cbf)
    if max_action is not None:
        load_model(action_net, "action_net", max_action)

    last_saved = min(max_cbf or 0, max_action or 0)

    for step in tqdm(range(last_saved, config.TRAIN_STEPS)):
        # s, g = generate_data_torch(num_agents, config.DIST_MIN_THRES, device)

        # Generate SMG environment data and augment
        s_np_ori, g_np_ori, obs_np, obs_g_np, border_points_np, wall_points_np = generate_social_mini_game_data(num_agents=random.randint(1, 4))
        ox = wall_points_np[:, 0].tolist() + border_points_np[:, 0].tolist()
        oy = wall_points_np[:, 1].tolist() + border_points_np[:, 1].tolist()
        a_star_planner = AStarPlanner(ox, oy, resolution=1 / config.RES, rr=config.DIST_MIN_THRES * 1.5)
        num_agents = s_np_ori.shape[0]
        trajectories = []
        trajectories_idx = [0 for _ in range(num_agents)]
        
        for i in range(num_agents):
            sx, sy = s_np_ori[i, 0], s_np_ori[i, 1]
            gx, gy = g_np_ori[i, 0], g_np_ori[i, 1]
            rx, ry = a_star_planner.planning(sx, sy, gx, gy)
            rx, ry = dynamic_downsampling(np.array(rx), np.array(ry), 15)
            rx, ry = np.expand_dims(rx[::-1], axis=1), np.expand_dims(ry[::-1], axis=1)
            trajectory = np.concatenate((rx, ry), axis=1)
            trajectories.append(trajectory)
        theta = random.random() * 2 * math.pi
        s_np_ori, g_np_ori, obs_np, obs_g_np, border_points_np, wall_points_np, trajectories = rotate_environment(theta, s_np_ori, g_np_ori, obs_np, obs_g_np, border_points_np, wall_points_np, trajectories)
        s_np, g_np = np.copy(s_np_ori), np.copy(g_np_ori)

        s, g = torch.tensor(s_np, dtype=torch.float32, device=device), torch.tensor(g_np, dtype=torch.float32, device=device)
        obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
        

        # Choose Network to backpropgate
        if (step // 10) % 2 == 0:
            optimizer = optimizer_h
            network_to_update = "cbf_net"
        else:
            optimizer = optimizer_a
            network_to_update = "action_net"            

        optimizer.zero_grad()


        for _ in range(config.INNER_LOOPS):

            # Identify next waypoint for each agent
            for agent_idx in range(num_agents):
                agent_pos = s_np[agent_idx, :2]
                traj = trajectories[agent_idx]
                idx = trajectories_idx[agent_idx]
                traj_len = traj.shape[0]

                if idx >= traj_len:
                    # Agent has reached the end of the trajectory
                    # Set next_waypoint to the final goal
                    next_waypoint = g_np_ori[agent_idx]
                else:
                    next_waypoint = traj[idx]
                    dist_to_next = np.linalg.norm(agent_pos - next_waypoint)
                    if dist_to_next < config.MIN_THRESHOLD:
                        idx += 1
                        if idx >= traj_len:
                            # Agent has reached the end of the trajectory
                            next_waypoint = g_np_ori[agent_idx]
                        else:
                            next_waypoint = traj[idx]
                        trajectories_idx[agent_idx] = idx
            g_np[agent_idx, :] = next_waypoint
            g = torch.tensor(g_np, dtype=torch.float32, device=device)

            # Model Forward Passes
            neighbor_features_cbf, indices = compute_neighbor_features(s, config.DIST_MIN_THRES, config.TOP_K, wall_agents=obs, include_d_norm=True)
            h = cbf_net(neighbor_features_cbf)
            neighbor_features_action, _ = compute_neighbor_features(s, config.DIST_MIN_THRES, config.TOP_K, wall_agents=obs, include_d_norm=False, indices=indices)
            a = action_net(s, g, neighbor_features_action)

            # Loss calculations
            loss_dang, loss_safe, acc_dang, acc_safe = barrier_loss(h, s, config.DIST_MIN_THRES, config.TIME_TO_COLLISION, indices)
            loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv = derivative_loss(h, s, a, cbf_net, config.ALPHA_CBF, indices, obs)
            loss_action = action_loss(a, s, g, state_gain)
            loss_distance = distance_loss(s, g)
            
            loss = 10 * (2 * loss_dang + loss_safe + 2 * loss_dang_deriv + loss_safe_deriv + 0.01 * loss_action + 0.1*loss_distance)
            loss = loss / config.INNER_LOOPS

            

            
            loss.backward()

            if network_to_update == "cbf_net":
                for param in action_net.parameters():
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad.zero_()
            else:
                for param in cbf_net.parameters():
                    if param.grad is not None:
                        param.grad.detach_()
                        param.grad.zero_()

            with torch.no_grad():
                s = s + dynamics(s, a) * config.TIME_STEP

        optimizer.step()

        if step % config.DISPLAY_STEPS == 0:
            print(f"Step: {step}, Loss: {loss.item() * config.INNER_LOOPS}")

        if (step + 1) % config.CHECKPOINT_STEPS == 0:
            save_model(cbf_net, "cbf_net", step + 1, base_dir=LOAD_DIR)
            save_model(action_net, "action_net", step + 1, base_dir=LOAD_DIR)


if __name__ == "__main__":
    train()
