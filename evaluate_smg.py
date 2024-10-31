import sys
import os
import time
import argparse
import numpy as np
import torch
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



'''
TORCH BACKEND
'''

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")



'''
UTILITY FUNCTIONS
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_step", type=int, default=None)
    parser.add_argument("--vis", type=int, default=0)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()
    return args


def print_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        acc_i = acc[:, i]
        acc_list.append(np.mean(acc_i[acc_i > 0]))
    print("Accuracy: {}".format(acc_list))


def calculate_curvature(rx, ry):
    rx, ry = np.array(rx), np.array(ry)
    dx = np.gradient(rx)
    dy = np.gradient(ry)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(ddx * dy - dx * ddy) / (dx**2 + dy**2)**1.5
    curvature = np.nan_to_num(curvature)
    return curvature


def dynamic_downsampling(rx, ry, target_num_points):
    curvature = calculate_curvature(rx, ry)
    curvature_sum = np.sum(curvature)
    curvature_norm = curvature / curvature_sum if curvature_sum != 0 else curvature
    cumulative_sum = np.cumsum(curvature_norm)
    u = np.linspace(0, 1, target_num_points)
    resample_indices = np.searchsorted(cumulative_sum, u, side='right')
    resample_indices = np.clip(resample_indices, 0, len(rx) - 1)
    resample_indices = np.unique(np.concatenate(([0], resample_indices, [len(rx) - 1])))
    rx_downsampled = rx[resample_indices]
    ry_downsampled = ry[resample_indices]
    return rx_downsampled, ry_downsampled



'''
MAIN LOOP
'''


def main():
    args = parse_args()
    cbf_net = CBFNetwork().to(device)
    action_net = ActionNetwork().to(device)
    cbf_net.load_state_dict(torch.load('checkpoints_lambda/cbf_net_step_70000.pth', weights_only=True, map_location=device))
    action_net.load_state_dict(torch.load('checkpoints_lambda/action_net_step_70000.pth', weights_only=True, map_location=device))
    cbf_net.eval()
    action_net.eval()

    safety_ratios_epoch = []
    safety_ratios_epoch_lqr = []
    dist_errors = []
    init_dist_errors = []
    accuracy_lists = []
    safety_reward = []
    dist_reward = []
    safety_reward_baseline = []
    dist_reward_baseline = []

    if args.vis:
        plt.ioff()
        plt.close()
        fig = plt.figure(figsize=(20, 10))
        ax_ours = fig.add_subplot(121)
        ax_lqr = fig.add_subplot(122)

        def update(frame):
            ax_ours.cla()
            ax_lqr.cla()
            j_ours = min(frame, len(s_np_ours) - 1)
            state_ours = s_np_ours[j_ours]
            safety_ours_current = collision_check_ours[j_ours]
            plot_single_state_with_wall_separate(
                ax_ours, state_ours, g_np_ori, safety_ours_current,
                border_points_np, wall_points_np,
                trajectories=trajectories, wall_agent_state=obs_np,
                action=a_np_ours[j_ours], action_res=a_res_np_ours[j_ours],
                action_opt=a_opt_np_ours[j_ours], agent_size=30,
            )
            ax_ours.set_title("MaCBF: Safety Rate = {:.3f}".format(np.mean(safety_ratios_epoch)), fontsize=14)
            j_lqr = min(frame, len(s_np_lqr) - 1)
            state_lqr = s_np_lqr[j_lqr]
            safety_lqr_current = collision_check_lqr[j_lqr]
            plot_single_state_with_wall_separate(
                ax_lqr, state_lqr, g_np_ori, safety_lqr_current,
                border_points_np, wall_points_np,
                trajectories=trajectories, wall_agent_state=obs_np,
                action_opt=a_np_lqr[j_lqr], agent_size=30,
            )
            ax_lqr.set_title("LQR: Safety Rate = {:.3f}".format(np.mean(safety_ratios_epoch_lqr)), fontsize=14)

    for istep in range(1):
        start_time = time.time()
        safety_info = []
        safety_info_baseline = []
        theta = random.random() * 2 * math.pi
        s_np_ori, g_np_ori, obs_np, obs_g_np, border_points_np, wall_points_np = generate_social_mini_game_data(theta=theta)
        s_np, g_np = np.copy(s_np_ori), np.copy(g_np_ori)
        init_dist_errors.append(np.mean(np.linalg.norm(s_np[:, :2] - g_np, axis=1)))
        s = torch.tensor(s_np, dtype=torch.float32, device=device)
        g = torch.tensor(g_np, dtype=torch.float32, device=device)
        obs = torch.tensor(obs_np, dtype=torch.float32, device=device)

        s_np_ours = []
        a_np_ours = []
        a_res_np_ours = []
        a_opt_np_ours = []
        s_np_lqr = []
        a_np_lqr = []
        safety_ours = []
        safety_lqr = []
        collision_check_ours = []
        collision_check_lqr = []

        ox = wall_points_np[:, 0].tolist() + border_points_np[:, 0].tolist()
        oy = wall_points_np[:, 1].tolist() + border_points_np[:, 1].tolist()
        a_star_planner = AStarPlanner(ox, oy, resolution=1 / config.RES, rr=config.DIST_MIN_THRES * 1.5)

        num_agents = s_np_ori.shape[0]
        trajectories = []
        trajectories_idx_macbf = [0 for _ in range(num_agents)]
        trajectories_idx_lqr = [0 for _ in range(num_agents)]

        for i in range(num_agents):
            sx, sy = s_np_ori[i, 0], s_np_ori[i, 1]
            gx, gy = g_np_ori[i, 0], g_np_ori[i, 1]
            rx, ry = a_star_planner.planning(sx, sy, gx, gy)
            rx, ry = dynamic_downsampling(np.array(rx), np.array(ry), 15)
            rx, ry = np.expand_dims(rx[::-1], axis=1), np.expand_dims(ry[::-1], axis=1)
            trajectory = np.concatenate((rx, ry), axis=1)
            trajectories.append(trajectory)

        trajectories_lqr = copy.deepcopy(trajectories)
        # initial_trajectories = copy.deepcopy(trajectories)

        for i in tqdm(range(config.EVALUATE_LENGTH), desc="MaCBF Evaluation"):
            for agent_idx in range(num_agents):
                agent_pos = s_np[agent_idx, :2]
                traj = trajectories[agent_idx]
                idx = trajectories_idx_macbf[agent_idx]
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
                        trajectories_idx_macbf[agent_idx] = idx
                    elif dist_to_next > config.MAX_THRESHOLD:
                        # Replan the trajectory
                        sx, sy = agent_pos[0], agent_pos[1]
                        gx, gy = g_np_ori[agent_idx, 0], g_np_ori[agent_idx, 1]
                        rx, ry = a_star_planner.planning(sx, sy, gx, gy)
                        rx, ry = dynamic_downsampling(np.array(rx), np.array(ry), 10)
                        rx, ry = np.expand_dims(rx[::-1], axis=1), np.expand_dims(ry[::-1], axis=1)
                        trajectory = np.concatenate((rx, ry), axis=1)
                        trajectories[agent_idx] = trajectory
                        trajectories_idx_macbf[agent_idx] = 0
                        idx = 0
                        next_waypoint = traj[idx]
                g_np[agent_idx, :] = next_waypoint


            g = torch.tensor(g_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                neighbor_features_cbf, indices = compute_neighbor_features(s, config.DIST_MIN_THRES, config.TOP_K, wall_agents=obs, include_d_norm=True)
                h = cbf_net(neighbor_features_cbf)
                neighbor_features_action, _ = compute_neighbor_features(s, config.DIST_MIN_THRES, config.TOP_K, wall_agents=obs, include_d_norm=False, indices=indices)
                a = action_net(s, g, neighbor_features_action)

            a_res = torch.zeros_like(a, requires_grad=True)
            optimizer_res = torch.optim.SGD([a_res], lr=config.REFINE_LEARNING_RATE)

            for _ in range(config.REFINE_LOOPS):
                optimizer_res.zero_grad()
                dsdt = dynamics(s, a + a_res)
                s_next = s + dsdt * config.TIME_STEP
                neighbor_features_cbf_next, _ = compute_neighbor_features(s_next, config.DIST_MIN_THRES, config.TOP_K, wall_agents=obs, include_d_norm=True, indices=None)
                h_next = cbf_net(neighbor_features_cbf_next)
                deriv = h_next - h + config.TIME_STEP * config.ALPHA_CBF * h
                deriv_flat = deriv.view(-1)
                error = torch.sum(torch.relu(-deriv_flat))
                error.backward()
                optimizer_res.step()

            with torch.no_grad():
                a_opt = a + a_res.detach()
                s = take_step_obstacles(s, a, wall_agents=obs)
            s_np = s.cpu().numpy()

            s_np_ours.append(s_np)
            a_np_ours.append(a.cpu().numpy())
            a_res_np_ours.append(a_res.detach().cpu().numpy())
            a_opt_np_ours.append(a_opt.detach().cpu().numpy())

            ttc_mask = ttc_dangerous_mask(config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK, neighbor_features_cbf)
            collision_check_ours.append(torch.any(ttc_mask, dim=1).cpu().numpy())
            safety_ratio = 1 - torch.mean(ttc_mask.float(), dim=1).cpu().numpy()
            safety_ours.append(safety_ratio)
            safety_info.append((safety_ratio == 1).astype(np.float32).reshape((1, -1)))
            safety_ratio_mean = np.mean(safety_ratio == 1)
            safety_ratios_epoch.append(safety_ratio_mean)

            if args.vis:
                if (np.mean(np.linalg.norm(s_np[:, :2] - g_np, axis=1)) < config.DIST_MIN_CHECK):
                    print("Break condition")
                    break

        dist_errors.append(np.mean(np.linalg.norm(s_np[:, :2] - g_np, axis=1)))
        safety_reward.append(np.mean(np.sum(np.concatenate(safety_info, axis=0) - 1, axis=0)))
        dist_reward.append(np.mean((np.linalg.norm(s_np[:, :2] - g_np, axis=1) < 0.2).astype(np.float32) * 10))

        # LQR Baseline
        s_lqr = torch.tensor(s_np_ori, dtype=torch.float32, device=device)
        s_np_lqr_current = s_np_ori.copy()
        g_np_lqr = g_np_ori.copy()
        
        # Initialize trajectory index counters for each agent
        trajectories_idx_lqr = [0 for _ in range(num_agents)]
        for i in tqdm(range(config.EVALUATE_LENGTH), desc="Basline LQR Evaluation"):

            for agent_idx in range(num_agents):
                agent_pos = s_np_lqr_current[agent_idx, :2]
                traj = trajectories_lqr[agent_idx]
                idx = trajectories_idx_lqr[agent_idx]
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
                        trajectories_idx_lqr[agent_idx] = idx
                    elif dist_to_next > config.MAX_THRESHOLD:
                        # Replan the trajectory
                        sx, sy = agent_pos[0], agent_pos[1]
                        gx, gy = g_np_ori[agent_idx, 0], g_np_ori[agent_idx, 1]
                        rx, ry = a_star_planner.planning(sx, sy, gx, gy)
                        rx, ry = dynamic_downsampling(np.array(rx), np.array(ry), 10)
                        rx, ry = np.expand_dims(rx[::-1], axis=1), np.expand_dims(ry[::-1], axis=1)
                        trajectory = np.concatenate((rx, ry), axis=1)
                        trajectories_lqr[agent_idx] = trajectory
                        trajectories_idx_lqr[agent_idx] = 0
                        idx = 0
                        next_waypoint = traj[idx]
                g_np_lqr[agent_idx, :] = next_waypoint


            g_lqr = torch.tensor(g_np_lqr, dtype=torch.float32, device=device)
            K = torch.tensor(np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3), dtype=torch.float32, device=device)
            s_ref = torch.cat([s_lqr[:, :2] - g_lqr, s_lqr[:, 2:]], dim=1)
            a_lqr = -s_ref @ K.T
            s_lqr = take_step_obstacles(s_lqr, a_lqr, wall_agents=obs)
            s_np_lqr_current = s_lqr.cpu().numpy()
            s_np_lqr.append(s_np_lqr_current)
            a_np_lqr.append(a_lqr.cpu().numpy())

            neighbor_features_cbf, indices = compute_neighbor_features(s_lqr, config.DIST_MIN_THRES, config.TOP_K, wall_agents=obs, include_d_norm=True)
            ttc_mask = ttc_dangerous_mask(config.DIST_MIN_CHECK, config.TIME_TO_COLLISION_CHECK, neighbor_features_cbf)
            collision_check_lqr.append(torch.any(ttc_mask, dim=1).cpu().numpy())
            safety_ratio = 1 - torch.mean(ttc_mask.float(), dim=1).cpu().numpy()
            safety_lqr.append(safety_ratio)
            safety_info_baseline.append((safety_ratio == 1).astype(np.float32).reshape((1, -1)))
            safety_ratio_mean = np.mean(safety_ratio == 1)
            safety_ratios_epoch_lqr.append(safety_ratio_mean)

            if (np.mean(np.linalg.norm(s_np_lqr_current[:, :2] - g_np_lqr, axis=1)) < config.DIST_MIN_CHECK / 3):
                break

        safety_reward_baseline.append(np.mean(np.sum(np.concatenate(safety_info_baseline, axis=0) - 1, axis=0)))
        dist_reward_baseline.append(np.mean((np.linalg.norm(s_np_lqr_current[:, :2] - g_np, axis=1) < 0.2).astype(np.float32) * 10))

        if args.vis:
            animations_dir = os.path.join(os.getcwd(), 'animations')            
            if not os.path.exists(animations_dir):
                os.makedirs(animations_dir)            
            max_frames = max(len(s_np_ours), len(s_np_lqr))
            ani = FuncAnimation(fig, update, frames=range(max_frames), interval=100, repeat=False)
            writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
            time_str = datetime.today().strftime('%Y_%m_%d-%H:%M:%S')
            file_path = os.path.join(animations_dir, time_str + f"_smg_{istep}.mp4")
            ani.save(file_path, writer=writer)

        end_time = time.time()
        print("Evaluation Step: {} | {}, Time: {:.4f}".format(istep + 1, config.EVALUATE_STEPS, end_time - start_time))

    print("Distance Error (Final | Initial): {:.4f} | {:.4f}".format(np.mean(dist_errors), np.mean(init_dist_errors)))
    print("Mean Safety Ratio (Learning | LQR): {:.4f} | {:.4f}".format(np.mean(safety_ratios_epoch), np.mean(safety_ratios_epoch_lqr)))
    print("Reward Safety (Learning | LQR): {:.4f} | {:.4f}, Reward Distance: {:.4f} | {:.4f}".format(
        np.mean(safety_reward), np.mean(safety_reward_baseline), np.mean(dist_reward), np.mean(dist_reward_baseline)))

if __name__ == "__main__":
    main()
