import matplotlib.pyplot as plt
import torch
from core import *
from vis import *
import config

# Generate the data
(
    agent_states,
    agent_goals,
    wall_agent_states,
    wall_agent_goals,
    border_points,
    wall_points,
) = generate_social_mini_game_data()

agent_states_torch = torch.tensor(agent_states, dtype=torch.float32, device=device)
wall_agent_states_torch = torch.tensor(
    wall_agent_states, dtype=torch.float32, device=device
)


action_torch = torch.zeros(
    (agent_states_torch.shape[0], 2), dtype=torch.float32, device=device
)


key_pressed = False


def on_key_press(event):
    global action_torch, key_pressed

    if event.key == "q":
        plt.close()
        exit(1)

    key_pressed = True

    # Define actions for different key presses using PyTorch tensors
    if event.key == "right":
        action_torch[0, :] = torch.tensor([1, 0], dtype=torch.float32, device=device)
    elif event.key == "left":
        action_torch[0, :] = torch.tensor([-1, 0], dtype=torch.float32, device=device)
    elif event.key == "up":
        action_torch[0, :] = torch.tensor([0, 1], dtype=torch.float32, device=device)
    elif event.key == "down":
        action_torch[0, :] = torch.tensor([0, -1], dtype=torch.float32, device=device)
    else:
        action_torch[:] = torch.tensor([0, 0], dtype=torch.float32, device=device)


while True:
    plt.clf()

    # dsdt = dynamics(agent_states_torch, action_torch)
    # agent_states_torch = agent_states_torch + dsdt * config.TIME_STEP

    agent_states_torch = take_step_obstacles(
        agent_states_torch, action_torch, wall_agents=wall_agent_states_torch
    )

    neighbor_features_cbf, indices = compute_neighbor_features(
        agent_states_torch,
        config.DIST_MIN_THRES,
        config.TOP_K,
        wall_agents=wall_agent_states_torch,
        include_d_norm=True,
    )

    ttc_mask = ttc_dangerous_mask(
        config.DIST_MIN_THRES, config.TIME_TO_COLLISION_CHECK, neighbor_features_cbf
    )
    collision_check = torch.any(ttc_mask, dim=1).cpu().numpy()

    agent_states_np = agent_states_torch.cpu().numpy()

    plot_single_state_with_wall_separate(
        agent_state=agent_states_np,
        agent_goal=agent_goals,
        safety=collision_check,
        border_points=border_points,
        wall_points=wall_points,
        wall_agent_state=wall_agent_states,
        agent_size=30,
    )

    fig = plt.gcf()
    fig.canvas.mpl_connect("key_press_event", on_key_press)

    if not key_pressed:
        action_torch[:] = torch.tensor([0, 0], dtype=torch.float32, device=device)

    key_pressed = False

    fig.canvas.draw()
    plt.pause(config.TIME_STEP)
