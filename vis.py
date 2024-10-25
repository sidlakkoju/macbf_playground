import numpy as np
import matplotlib.pyplot as plt


def plot_single_state_with_wall_separate(
    agent_state,
    agent_goal,
    safety,
    border_points,
    wall_points,
    trajectories=None,
    wall_agent_state=None,
    action=None,
    action_res=None,
    action_opt=None,
    agent_size=0.5,
):


    action = None
    action_res = None
    action_opt = None
    
    agent_state_vis = agent_state[:, :2]

    # Plot border and wall points first
    plt.scatter(
        border_points[:, 0],
        border_points[:, 1],
        label="Border Points",
        color="blue",
        s=10,
    )
    plt.scatter(
        wall_points[:, 0], wall_points[:, 1], label="Wall Points", color="green", s=10
    )

    # Plot wall agents if any
    if wall_agent_state is not None:
        wall_state_vis = wall_agent_state[:, :2]
        plt.scatter(
            wall_state_vis[:, 0],
            wall_state_vis[:, 1],
            color="grey",
            s=agent_size,
            label="Wall",
            alpha=0.6,
        )

    # Plot agent goals
    agent_goal_vis = agent_goal
    plt.scatter(
        agent_goal_vis[:, 0],
        agent_goal_vis[:, 1],
        color="deepskyblue",
        s=agent_size,
        label="Target",
        alpha=0.6,
    )

    # Plot trajectories
    if trajectories is not None:
        num_agents = len(trajectories)
        cmap = plt.get_cmap("tab10")
        for i in range(num_agents):
            trajectory = np.array(trajectories[i])
            if trajectory.shape[0] > 0:
                plt.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    linestyle="--",
                    color=cmap(i % 10),
                    label=f"Agent {i} Trajectory",
                )

    # Plot actions if provided
    if action is not None:
        plt.quiver(
            agent_state_vis[:, 0],
            agent_state_vis[:, 1],
            action[:, 0],
            action[:, 1],
            color="green",
            scale=0.5,
            label="Action",
        )
        for i in range(agent_state_vis.shape[0]):
            plt.text(
                agent_state_vis[i, 0] + 0.05,
                agent_state_vis[i, 1] + 0.05,
                f"{action[i]}",
                color="green",
                fontsize=8,
            )

    if action_res is not None:
        plt.quiver(
            agent_state_vis[:, 0],
            agent_state_vis[:, 1],
            action_res[:, 0],
            action_res[:, 1],
            color="blue",
            scale=0.5,
            label="Action Res",
        )
        for i in range(agent_state_vis.shape[0]):
            plt.text(
                agent_state_vis[i, 0] + 0.05,
                agent_state_vis[i, 1] - 0.1,
                f"{action_res[i]}",
                color="blue",
                fontsize=8,
            )

    if action_opt is not None:
        plt.quiver(
            agent_state_vis[:, 0],
            agent_state_vis[:, 1],
            action_opt[:, 0],
            action_opt[:, 1],
            color="purple",
            scale=0.5,
            label="Action Opt",
        )
        for i in range(agent_state_vis.shape[0]):
            plt.text(
                agent_state_vis[i, 0] + 0.05,
                agent_state_vis[i, 1] - 0.2,
                f"{action_opt[i]}",
                color="purple",
                fontsize=8,
            )

    # Finally, plot the agent states last so they appear on top
    plt.scatter(
        agent_state_vis[:, 0],
        agent_state_vis[:, 1],
        color="darkorange",
        s=agent_size,
        label="Agent",
        alpha=0.6,
    )

    # Plot collision states where safety < 1
    collision_indices = np.where(safety < 1)[0]
    if collision_indices.size > 0:
        plt.scatter(
            agent_state_vis[collision_indices, 0],
            agent_state_vis[collision_indices, 1],
            color="red",
            s=agent_size,
            label="Collision",
            alpha=0.9,
        )

    # Set plot limits to match the original data range
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.legend()

    # Customize axis spines
    ax = plt.gca()
    for side in ax.spines.keys():
        ax.spines[side].set_linewidth(2)
        ax.spines[side].set_color("grey")
    plt.show()



# LEGACY CHUCHU FAN CODE

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

