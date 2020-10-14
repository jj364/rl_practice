import matplotlib.pyplot as plt
from matplotlib import colors, patches


def show_track(t, start, finish, trajectory=None):
    """
    Plot track including car's trajectory
    :param t: Binary numpy array where 1 is on track, 0 is off
    :param start: Start line, list of tuples
    :param finish: Finish line, list of tuples
    :param trajectory: [(y,x,vy,vx,action,greedy action), reward] Car's trajectory
    """
    track = t.copy()
    for s in start:
        track[s[0], s[1]] = 2

    for f in finish:
        if f[1] < track.shape[1]:
            track[f[0], f[1]] = 3

    # Plot entire trajectory
    if trajectory is not None:
        for t in trajectory[::-1]:
            track[t[0][0], t[0][1]] = 4

        # Plot where trajectory intersects finish line
        [vy, vx] = trajectory[-1][0][2:4]
        [y, x] = trajectory[-1][0][0:2]
        if vy >= vx:
            for dy in range(1, vy+1):
                ty = y + dy
                tx = x + round(dy*vx/vy)
                if (ty, tx) in finish:
                    break
        else:
            for dx in range(1, vx+1):
                tx = x + dx
                ty = y + round(dx*vy/vx)
                if (ty, tx) in finish:
                    break
        if (ty, tx) not in finish:  # Catch error in finish line position
            (ty, tx) = finish[-1]
        track[ty, tx] = 4

    # create discrete colormap
    cmap = colors.ListedColormap(['red', 'blue', 'green', 'purple', 'yellow'])
    labels = ['Off Track', 'Track', 'Start', 'Finish', 'Path']
    bounds = [0, 1, 2, 3, 4, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(track, cmap=cmap, norm=norm)
    # Create legend
    handles = []
    for c in cmap.colors:
        handles.append(patches.Patch(color=c))
    ax.legend(handles=handles, labels=labels)

    plt.show()