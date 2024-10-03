import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import config
from core import *
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")
print(f"Using device: {device}")

# Save the model checkpoints
def save_model(model, model_name, step):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    torch.save(model.state_dict(), f'checkpoints/{model_name}_step_{step}.pth')


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

            loss_dang, loss_safe, acc_dang, acc_safe = barrier_loss(h, s, config.DIST_MIN_THRES, config.TIME_TO_COLLISION, indices)
            loss_dang_deriv, loss_safe_deriv, acc_dang_deriv, acc_safe_deriv = derivative_loss(h, s, a, cbf_net, config.ALPHA_CBF, indices)
            loss_action = action_loss(a, s, g)
            loss = 10 * (2*loss_dang + loss_safe + 2*loss_dang_deriv + loss_safe_deriv + 0.01*loss_action)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                s = s + dynamics(s, a) * config.TIME_STEP

        if step % config.DISPLAY_STEPS == 0:
            print(f"Step: {step}, Loss: {loss.item():.4f}")
        
        if (step + 1) % config.CHECKPOINT_STEPS == 0:
            save_model(cbf_net, "cbf_net", step + 1)
            save_model(action_net, "action_net", step + 1)

if __name__ == "__main__":
    num_agents = 32
    train(num_agents)
