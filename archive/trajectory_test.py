import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def calculate_curvature(rx, ry):
    """Calculate curvature at each point on the trajectory."""
    rx, ry = np.array(rx), np.array(ry)
    dx = np.gradient(rx)
    dy = np.gradient(ry)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature formula
    curvature = np.abs(ddx * dy - dx * ddy) / (dx**2 + dy**2)**1.5
    curvature = np.nan_to_num(curvature)  # Handle divide by zero in straight lines
    return curvature

def dynamic_downsampling(rx, ry, target_num_points):
    """Perform dynamic downsampling based on curvature."""
    # Step 1: Calculate curvature
    curvature = calculate_curvature(rx, ry)

    
    
    # Normalize curvature to sum to 1 (probability distribution for resampling)
    curvature_sum = np.sum(curvature)
    curvature_norm = curvature / curvature_sum if curvature_sum != 0 else curvature

    
    
    # Step 2: Cumulative sum for non-uniform sampling (higher resampling in high curvature regions)
    cumulative_sum = np.cumsum(curvature_norm)
    print(f'curvature_norm: {curvature_norm.shape}')
    print(f'curvature_norm: {curvature_norm}')
    print(f'cumulative_sum: {cumulative_sum.shape}')
    print(f'cumulative_sum: {cumulative_sum}')
    
    # Step 3: Resample points based on the cumulative sum
    u = np.linspace(0, 1, target_num_points)
    
    # Ensure indices are within bounds
    resample_indices = np.searchsorted(cumulative_sum, u, side='right')
    resample_indices = np.clip(resample_indices, 0, len(rx) - 1)

    # Ensure we always keep the first and last points
    resample_indices = np.unique(np.concatenate(([0], resample_indices, [len(rx) - 1])))

    # Downsampled points
    rx_downsampled = rx[resample_indices]
    ry_downsampled = ry[resample_indices]
    
    return rx_downsampled, ry_downsampled

# Example rx, ry (your points from the planner)
# rx = np.array([0, 1, 2, 3, 4, 5])
# ry = np.array([0, 0.5, 0.5, 0.2, -0.1, -0.3])

rx = np.array([8.0, 7.875, 7.75, 7.625, 7.5, 7.375, 7.25, 7.125, 7.0, 6.875, 6.75, 6.625, 6.5, 6.375, 6.25, 6.125, 6.0, 5.875, 5.75, 5.625, 5.5, 5.375, 5.25, 5.125, 5.0, 4.875, 4.75, 4.625, 4.5, 4.375, 4.25, 4.125, 4.0, 3.875, 3.75, 3.625, 3.5, 3.375, 3.25, 3.125, 3.0, 2.875, 2.75, 2.625, 2.5, 2.375, 2.25, 2.125, 2.0])
ry = np.array([8.0, 7.875, 7.75, 7.625, 7.5, 7.375, 7.25, 7.125, 7.0, 6.875, 6.75, 6.625, 6.5, 6.375, 6.25, 6.125, 6.0, 5.875, 5.75, 5.625, 5.5, 5.375, 5.25, 5.125, 5.0, 4.875, 4.75, 4.625, 4.5, 4.375, 4.25, 4.125, 4.0, 3.875, 3.75, 3.625, 3.5, 3.375, 3.25, 3.125, 3.0, 2.875, 2.75, 2.625, 2.5, 2.375, 2.25, 2.125, 2.0])

# Perform dynamic downsampling
target_num_points = 10  # Target number of points
rx_new, ry_new = dynamic_downsampling(rx, ry, target_num_points)

# Plot the original and downsampled points for comparison
plt.plot(rx, ry, 'o-', label='Original points')
plt.plot(rx_new, ry_new, 'x-', label='Downsampled points')
plt.legend()
plt.show()

# You can now use rx_new and ry_new as your dynamically downsampled trajectory
trajectory = list(zip(rx_new, ry_new))
