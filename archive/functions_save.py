def compute_neighbor_features(
    s, r, k, wall_agents=None, include_d_norm=False, indices=None
):
    num_agents = s.size(0)

    s_diff_agents = s.unsqueeze(1) - s.unsqueeze(0)  # Shape: [num_agents, num_agents, 4]
    distances_agents = (torch.norm(s_diff_agents[:, :, :2], dim=2) + 1e-4)  # Shape: [num_agents, num_agents]

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
        d_norm_agents = (
            distances_agents[torch.arange(num_agents).unsqueeze(1), indices].unsqueeze(2) - r
        )
        neighbor_features_agents = torch.cat(
            [neighbor_features_agents, d_norm_agents], dim=2
        )

    if wall_agents is not None:
        num_wall_agents = wall_agents.size(0)

        # Compute s_diff between agents and obstacles
        s_diff_obstacles = s.unsqueeze(1) - wall_agents.unsqueeze(0)  # Shape: [num_agents, num_obstacles, 4]
        distances_obstacles = (
            torch.norm(s_diff_obstacles[:, :, :2], dim=2) + 1e-4
        )  # Shape: [num_agents, num_obstacles]

        # For obstacles, 'eye' variable is zero
        eye_obstacles = torch.zeros(
            num_agents, num_wall_agents, 1, device=s.device
        )  # Shape: [num_agents, num_obstacles, 1]
        neighbor_features_obstacles = torch.cat(
            [s_diff_obstacles, eye_obstacles], dim=2
        )

        if include_d_norm:
            d_norm_obstacles = (
                distances_obstacles.unsqueeze(2) - r
            )  # Shape: [num_agents, num_obstacles, 1]
            neighbor_features_obstacles = torch.cat(
                [neighbor_features_obstacles, d_norm_obstacles], dim=2
            )

        # Concatenate agent neighbors and obstacle neighbors
        neighbor_features = torch.cat(
            [neighbor_features_agents, neighbor_features_obstacles], dim=1
        )  # Shape: [num_agents, k + num_obstacles, features]
    else:
        neighbor_features = neighbor_features_agents

    # Return neighbor features and indices (indices correspond to agent neighbors)
    return neighbor_features, indices
