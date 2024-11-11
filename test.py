import numpy as np
from vis import *
from core import *


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter




fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)

agent_states, agent_goals, wall_agent_states, wall_agent_goals, border_points, wall_points = generate_social_mini_game_data(theta=0.7)

plot_single_state_with_wall_separate(
                ax,
                agent_states,
                agent_goals,
                0,
                border_points,
                wall_points,
                wall_agent_state=wall_agent_states,
                agent_size=30,
            )

plt.show()