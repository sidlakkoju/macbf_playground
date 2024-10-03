import sys
import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import core
import config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_agents', type=int, required=True)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--vis', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    return args


def print_accuracy(accuracy_lists):
    acc = np.array(accuracy_lists)
    acc_list = []
    for i in range(acc.shape[1]):
        acc_i = acc[:, i]
        acc_list.append(np.mean(acc_i[acc_i > 0]))
    print('Accuracy: {}'.format(acc_list))


def render_init():
    fig = plt.figure(figsize=(9, 4))
    return fig


def main():
    args = parse_args()

    # Load models
    cbf_net = core.CBFNetwork().to(device)
    action_net = core.ActionNetwork().to(device)

    # cbf_net.load_state_dict(torch.load(os.path.join(args.model_path, 'cbf_net.pth')))
    # action_net.load_state_dict(torch.load(os.path.join(args.model_path, 'action_net.pth')))
    
    # cbf_net.load_state_dict(torch.load('checkpoints/cbf_net_step_100.pth'))
    # action_net.load_state_dict(torch.load('checkpoints/action_net_step_100.pth'))

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
        plt.ion()
        plt.close()
        fig = render_init()

    for istep in range(config.EVALUATE_STEPS):
        start_time = time.time()

        safety_info = []
        safety_info_baseline = []

        s_np_ori, g_np_ori = core.generate_data_np(args.num_agents, config.DIST_MIN_THRES * 1.5)
        s_np, g_np = np.copy(s_np_ori), np.copy(g_np_ori)
        init_dist_errors.append(np.mean(np.linalg.norm(s_np[:, :2] - g_np, axis=1)))

        s_np_ours = []
        s_np_lqr = []

        safety_ours = []
        safety_lqr = []

        s = torch.tensor(s_np, dtype=torch.float32, device=device)
        g = torch.tensor(g_np, dtype=torch.float32, device=device)

        # Run INNER_LOOPS steps to reach the current goals
        for i in range(config.INNER_LOOPS):
            with torch.no_grad():
                # For CBF Network
                neighbor_features_cbf, indices = core.compute_neighbor_features(
                    s, config.DIST_MIN_THRES, config.TOP_K, include_d_norm=True)
                h = cbf_net(neighbor_features_cbf)
                # For Action Network
                neighbor_features_action, _ = core.compute_neighbor_features(
                    s, config.DIST_MIN_THRES, config.TOP_K, include_d_norm=False, indices=indices)
                a = action_net(s, g, neighbor_features_action)

            # Initialize a_res for refinement
            a_res = torch.zeros_like(a, requires_grad=True)
            optimizer_res = torch.optim.SGD([a_res], lr=config.REFINE_LEARNING_RATE)

            # Refinement loop
            for _ in range(config.REFINE_LOOPS):
                optimizer_res.zero_grad()
                dsdt = core.dynamics(s, a + a_res)
                s_next = s + dsdt * config.TIME_STEP
                neighbor_features_cbf_next, _ = core.compute_neighbor_features(
                    s_next, config.DIST_MIN_THRES, config.TOP_K, include_d_norm=True, indices=indices)
                h_next = cbf_net(neighbor_features_cbf_next)
                deriv = h_next - h + config.TIME_STEP * config.ALPHA_CBF * h
                deriv_flat = deriv.view(-1)
                error = torch.sum(torch.relu(-deriv_flat))
                error.backward()
                optimizer_res.step()

            a_opt = a + a_res.detach()
            dsdt = core.dynamics(s, a_opt)
            s = s + dsdt * config.TIME_STEP

            s_np_current = s.cpu().numpy()
            s_np_ours.append(s_np_current)

            # Safety check
            ttc_mask = core.ttc_dangerous_mask(s, config.DIST_MIN_CHECK,
                                               config.TIME_TO_COLLISION_CHECK, indices)
            safety_ratio = 1 - torch.mean(ttc_mask.float(), dim=1).cpu().numpy()
            safety_ours.append(safety_ratio)
            safety_info.append((safety_ratio == 1).astype(np.float32).reshape((1, -1)))
            safety_ratio_mean = np.mean(safety_ratio == 1)
            safety_ratios_epoch.append(safety_ratio_mean)

            if args.vis:
                if np.amax(np.linalg.norm(s_np_current[:, :2] - g_np, axis=1)) < config.DIST_MIN_CHECK / 3:
                    time.sleep(1)
                    break
                if np.mean(np.linalg.norm(s_np_current[:, :2] - g_np, axis=1)) < config.DIST_MIN_CHECK / 2:
                    K = np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3)
                    s_ref = np.concatenate([s_np_current[:, :2] - g_np, s_np_current[:, 2:]], axis=1)
                    a_lqr = -s_ref.dot(K.T)
                    dsdt = np.concatenate([s_np_current[:, 2:], a_lqr], axis=1)
                    s_np_current = s_np_current + dsdt * config.TIME_STEP
                    s = torch.tensor(s_np_current, dtype=torch.float32, device=device)

            else:
                if np.mean(np.linalg.norm(s_np_current[:, :2] - g_np, axis=1)) < config.DIST_MIN_CHECK:
                    break

        dist_errors.append(np.mean(np.linalg.norm(s_np_current[:, :2] - g_np, axis=1)))
        safety_reward.append(np.mean(np.sum(np.concatenate(safety_info, axis=0) - 1, axis=0)))
        dist_reward.append(np.mean(
            (np.linalg.norm(s_np_current[:, :2] - g_np, axis=1) < 0.2).astype(np.float32) * 10))

        # LQR Baseline
        s_lqr = torch.tensor(s_np_ori, dtype=torch.float32, device=device)
        for i in range(config.INNER_LOOPS):
            K = torch.tensor(np.eye(2, 4) + np.eye(2, 4, k=2) * np.sqrt(3),
                             dtype=torch.float32, device=device)
            s_ref = torch.cat([s_lqr[:, :2] - g, s_lqr[:, 2:]], dim=1)
            a_lqr = -s_ref @ K.T
            s_lqr = s_lqr + core.dynamics(s_lqr, a_lqr) * config.TIME_STEP

            s_np_lqr_current = s_lqr.cpu().numpy()
            s_np_lqr.append(s_np_lqr_current)

            ttc_mask = core.ttc_dangerous_mask(s_lqr, config.DIST_MIN_CHECK,
                                               config.TIME_TO_COLLISION_CHECK, indices)
            safety_ratio = 1 - torch.mean(ttc_mask.float(), dim=1).cpu().numpy()
            safety_lqr.append(safety_ratio)
            safety_info_baseline.append((safety_ratio == 1).astype(np.float32).reshape((1, -1)))
            safety_ratio_mean = np.mean(safety_ratio == 1)
            safety_ratios_epoch_lqr.append(safety_ratio_mean)

            if np.mean(np.linalg.norm(s_np_lqr_current[:, :2] - g_np, axis=1)) < config.DIST_MIN_CHECK / 3:
                break

        safety_reward_baseline.append(np.mean(
            np.sum(np.concatenate(safety_info_baseline, axis=0) - 1, axis=0)))
        dist_reward_baseline.append(np.mean(
            (np.linalg.norm(s_np_lqr_current[:, :2] - g_np, axis=1) < 0.2).astype(np.float32) * 10))

        if args.vis:
            # Visualize the trajectories
            vis_range = max(1, np.amax(np.abs(s_np_ori[:, :2])))
            agent_size = 100 / vis_range ** 2
            g_np_vis = g_np / vis_range
            max_steps = max(len(s_np_ours), len(s_np_lqr))
            for j in range(max_steps):
                plt.clf()

                plt.subplot(121)
                j_ours = min(j, len(s_np_ours) - 1)
                s_np_vis = s_np_ours[j_ours] / vis_range
                plt.scatter(s_np_vis[:, 0], s_np_vis[:, 1],
                            color='darkorange',
                            s=agent_size, label='Agent', alpha=0.6)
                plt.scatter(g_np_vis[:, 0], g_np_vis[:, 1],
                            color='deepskyblue',
                            s=agent_size, label='Target', alpha=0.6)
                safety = np.squeeze(safety_ours[j_ours])
                plt.scatter(s_np_vis[safety < 1, 0], s_np_vis[safety < 1, 1],
                            color='red',
                            s=agent_size, label='Collision', alpha=0.9)
                plt.xlim(-0.5, 1.5)
                plt.ylim(-0.5, 1.5)
                ax = plt.gca()
                for side in ax.spines.keys():
                    ax.spines[side].set_linewidth(2)
                    ax.spines[side].set_color('grey')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                plt.legend(loc='upper right', fontsize=14)
                plt.title('Ours: Safety Rate = {:.3f}'.format(
                    np.mean(safety_ratios_epoch)), fontsize=14)

                plt.subplot(122)
                j_lqr = min(j, len(s_np_lqr) - 1)
                s_np_vis = s_np_lqr[j_lqr] / vis_range
                plt.scatter(s_np_vis[:, 0], s_np_vis[:, 1],
                            color='darkorange',
                            s=agent_size, label='Agent', alpha=0.6)
                plt.scatter(g_np_vis[:, 0], g_np_vis[:, 1],
                            color='deepskyblue',
                            s=agent_size, label='Target', alpha=0.6)
                safety = np.squeeze(safety_lqr[j_lqr])
                plt.scatter(s_np_vis[safety < 1, 0], s_np_vis[safety < 1, 1],
                            color='red',
                            s=agent_size, label='Collision', alpha=0.9)
                plt.xlim(-0.5, 1.5)
                plt.ylim(-0.5, 1.5)
                ax = plt.gca()
                for side in ax.spines.keys():
                    ax.spines[side].set_linewidth(2)
                    ax.spines[side].set_color('grey')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                plt.legend(loc='upper right', fontsize=14)
                plt.title('LQR: Safety Rate = {:.3f}'.format(
                    np.mean(safety_ratios_epoch_lqr)), fontsize=14)

                fig.canvas.draw()
                plt.pause(0.01)
            plt.clf()

        end_time = time.time()
        print('Evaluation Step: {} | {}, Time: {:.4f}'.format(
            istep + 1, config.EVALUATE_STEPS, end_time - start_time))

    # Optionally print accuracy if you collect accuracy lists
    # print_accuracy(accuracy_lists)

    print('Distance Error (Final | Initial): {:.4f} | {:.4f}'.format(
          np.mean(dist_errors), np.mean(init_dist_errors)))
    print('Mean Safety Ratio (Learning | LQR): {:.4f} | {:.4f}'.format(
          np.mean(safety_ratios_epoch), np.mean(safety_ratios_epoch_lqr)))
    print('Reward Safety (Learning | LQR): {:.4f} | {:.4f}, Reward Distance: {:.4f} | {:.4f}'.format(
        np.mean(safety_reward), np.mean(safety_reward_baseline),
        np.mean(dist_reward), np.mean(dist_reward_baseline)))


if __name__ == '__main__':
    main()

