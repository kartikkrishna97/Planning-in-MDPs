import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch



def Plotting(V, policy):
    # Define matrices
    values = np.array(V)
    policies = policy

    # Plot heatmap
    plt.figure(figsize=(16, 12))  # Adjust figure size for better visibility
    sns.heatmap(values, annot=True, fmt=".2f", cmap="YlGnBu", cbar=False)

    # Arrow configuration
    arrow_configs = {
        'd': {'dx': 0, 'dy': 1},   # Adjust arrow directions for a 50x50 grid
        'r': {'dx': 1, 'dy': 0},
        'l': {'dx': -1, 'dy': 0},
        't': {'dx': 0, 'dy': -1}
    }

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            policy = policies[i][j]
            if policy and policy in arrow_configs:
                dx = arrow_configs[policy]['dx'] / 25  # Scale the arrow length for a 50x50 grid
                dy = arrow_configs[policy]['dy'] / 25
                # Compute start and end points of the arrow
                start = (j + 0.5, i + 0.5)
                end = (j + 0.5 + dx, i + 0.5 + dy)
                # Create and add the arrow patch
                arrow = FancyArrowPatch(start, end, arrowstyle="-|>", color="black", mutation_scale=15)
                plt.gca().add_patch(arrow)

    # Adjust the ticks and labels
    plt.xticks(np.arange(0, values.shape[1], 5) + 0.5, np.arange(0, values.shape[1], 5) + 1)
    plt.yticks(np.arange(0, values.shape[0], 5) + 0.5, np.arange(0, values.shape[0], 5) + 1, rotation=0)

    # Add title
    plt.title('Values and Heatmap for Policy Iteration for Large Map')

    plt.savefig('Large Map')




def Plotting_small(V, policy):
# Define matrices
    values = np.array(V)

    policies = policy
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(values, annot=True, fmt=".2f", cmap="YlGnBu", cbar=False)
    # sns.heatmap(values, annot=True, fmt=".2f", cbar=False)

    # Arrow configuration
    # Instead of (dx, dy), we use explicit end-points for better control
    arrow_configs = {
        'd': {'dx': 0, 'dy': 0.3},
        'r': {'dx': 0.3, 'dy': 0},
        'l': {'dx': -0.3, 'dy': 0},
        't': {'dx': 0, 'dy': -0.3}
    }

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            policy = policies[i][j]
            if policy and policy in arrow_configs:
                dx = arrow_configs[policy]['dx']
                dy = arrow_configs[policy]['dy']
                # Compute start and end points of the arrow
                start = (j + 0.5, i + 0.5)
                end = (j + 0.5 + dx, i + 0.5 + dy)
                # Create and add the arrow patch
                arrow = FancyArrowPatch(start, end, arrowstyle="-|>", color="black", mutation_scale=15)
                plt.gca().add_patch(arrow)

    # Adjust the ticks and labels
    plt.xticks(np.arange(values.shape[1]) + 0.5, np.arange(1, values.shape[1] + 1))
    plt.yticks(np.arange(values.shape[0]) + 0.5, np.arange(1, values.shape[0] + 1), rotation=0)

    # Add title
    plt.title('Values and Heatmap for Policy Iteration for small Map')
    plt.savefig('Policy Iteration')

