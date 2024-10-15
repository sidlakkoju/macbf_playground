import numpy as np
import core
import config
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_wall_hole(wall_x, hole_y_center, hole_height):
    hole_y_min = hole_y_center - hole_height / 2
    hole_y_max = hole_y_center + hole_height / 2
    wall_y_positions = np.linspace(
        config.Y_MIN, config.Y_MAX, (config.Y_MAX - config.Y_MIN) * config.RES + 1
    )
    wall_y_positions = wall_y_positions[
        (wall_y_positions <= hole_y_min) | (wall_y_positions >= hole_y_max)
    ]
    wall_x_positions = np.ones_like(wall_y_positions) * wall_x
    wall_x_positions = np.expand_dims(wall_x_positions, axis=1)
    wall_y_positions = np.expand_dims(wall_y_positions, axis=1)
    return np.concatenate([wall_x_positions, wall_y_positions], axis=1)


print(generate_wall_hole(5.0, 5.0, 1.0))


# # Example usage:
# agent_states, agent_goals, wall_states, wall_goals = generate_social_mini_game_data()

# print("Agent States:\n", agent_states)
# print("Agent Goals:\n", agent_goals)
# print("Wall States:\n", wall_states)
# print("Wall Goals:\n", wall_goals)

# # Example usage:
# # Assuming safety values for agents (e.g., all safe)
# safety = np.ones(agent_states.shape[0])

# # Use initial agent states for consistent scaling
# original_agent_state = agent_states.copy()

# # Plot the frame
# plot_single_state_with_wall_separate(agent_states, agent_goals, wall_states, wall_goals,
#                                      safety, original_agent_state, agent_size=100)


# # Plot
# core.plot_single_state_with_original(s_np, g_np, safety_ratio, s_np, agent_size=100)

# import numpy as np

# def generate_social_mini_game_data():
#     """
#     Generates initial states and goals for a social mini-game with two agents and a wall.
#     The agents start on one side of the wall and need to pass through a hole to reach their goals.

#     Returns:
#         states: np.array of shape (2, 4), where each row is [x, y, vx, vy]
#         goals: np.array of shape (2, 2), where each row is [goal_x, goal_y]
#         wall: dict containing wall position and hole information
#     """
#     # Map dimensions
#     x_min, x_max = 0.0, 10.0
#     y_min, y_max = 0.0, 10.0

#     # Wall parameters
#     wall_x = 5.0  # Wall is vertical at x=5
#     hole_y_center = 5.0  # Hole is centered at y=5
#     hole_height = 1.0  # Height of the hole
#     hole_y_min = hole_y_center - hole_height / 2
#     hole_y_max = hole_y_center + hole_height / 2

#     # Number of agents
#     num_agents = 2
#     states = np.zeros((num_agents, 4), dtype=np.float32)
#     goals = np.zeros((num_agents, 2), dtype=np.float32)

#     # Initial positions (agents on the left side of the wall)
#     agent_offset = 2.0  # Distance from the wall
#     states[0, :2] = [wall_x - agent_offset, y_max - agent_offset]  # Agent 1 in upper-left quadrant
#     states[1, :2] = [wall_x - agent_offset, y_min + agent_offset]  # Agent 2 in lower-left quadrant

#     # Initial velocities set to zero
#     states[:, 2:] = 0.0

#     # Goal positions (symmetric positions on the opposite side)
#     goals[0] = [wall_x + agent_offset, y_min + agent_offset]       # Agent 1 goal in lower-right quadrant
#     goals[1] = [wall_x + agent_offset, y_max - agent_offset]       # Agent 2 goal in upper-right quadrant

#     # Wall representation (if needed for your simulation)
#     wall = {
#         'x': wall_x,
#         'hole_y_min': hole_y_min,
#         'hole_y_max': hole_y_max
#     }

#     return states, goals, wall

# # Example usage:
# states, goals, wall = generate_social_mini_game_data()

# print("Initial States:\n", states)
# print("Goal Positions:\n", goals)
# print("Wall Information:\n", wall)


# import matplotlib.pyplot as plt

# def plot_initial_setup(states, goals, wall):
#     """
#     Plots the initial positions of agents, their goals, and the wall with a hole.
#     """
#     plt.figure(figsize=(8, 8))
#     plt.scatter(states[:, 0], states[:, 1], color='darkorange', s=100, label='Agents')
#     plt.scatter(goals[:, 0], goals[:, 1], color='deepskyblue', s=100, label='Goals')

#     # Plot the wall
#     wall_x = wall['x']
#     hole_y_min = wall['hole_y_min']
#     hole_y_max = wall['hole_y_max']

#     # Wall segments above the hole
#     plt.plot([wall_x, wall_x], [wall['hole_y_max'], 10], color='grey', linewidth=4)
#     # Wall segments below the hole
#     plt.plot([wall_x, wall_x], [0, wall['hole_y_min']], color='grey', linewidth=4)

#     # Indicate the hole
#     plt.plot([wall_x, wall_x], [hole_y_min, hole_y_max], color='white', linewidth=4, label='Hole')

#     plt.xlim(0, 10)
#     plt.ylim(0, 10)
#     plt.legend()
#     plt.title('Initial Setup of the Social Mini-Game')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.grid(True)
#     plt.show()

# # Plot the initial setup
# plot_initial_setup(states, goals, wall)
