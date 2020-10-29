#! /usr/bin/env python

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, patches


def plot_world(g, start, finish, wind, trajectory=None):
    """
    Plot track including car's trajectory
    :param g: Binary numpy array where 1 is on track, 0 is off
    :param start: Start line, list of tuples
    :param finish: Finish line, list of tuples
    :param trajectory: [(y,x,vy,vx,action,greedy action), reward] Car's trajectory
    """
    grid = g.copy()

    # Plot start and finish points
    grid[start[0], start[1]] = 2
    grid[finish[0], finish[1]] = 3

    # Plot entire trajectory
    if trajectory is not None:
        for t in trajectory:
            grid[t[0], t[1]] = 4

    # create discrete colormap
    cmap = colors.ListedColormap(['blue', 'green', 'purple', 'yellow'])
    labels = ['Grid', 'Start', 'Finish', 'Path']
    bounds = [1, 2, 3, 4, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap=cmap, norm=norm)
    ax.set_yticks([])
    ax.set_xticks(list(range(len(wind))))
    ax.set_xticklabels(wind)
    # Create legend
    handles = []
    for c in cmap.colors:
        handles.append(patches.Patch(color=c))
    ax.legend(handles=handles, labels=labels)

    plt.show()


if __name__ == "__main__":
    grid = np.ones((7, 10))
    start = np.array([3, 0])
    finish = np.array([3, 7])
    wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    plot_world(grid, start, finish, wind)