import matplotlib.pyplot as plt
from matplotlib import colors


def show_track(track, start, finish, trajectory=None):

    for s in start:
        track[s[0], s[1]] = 2

    for f in finish:
        if f[1] < track.shape[1]:
            track[f[0], f[1]] = 3

    if trajectory is not None:
        for t in trajectory[::-1]:
            track[t[0][0], t[0][1]] = 4
            if [t[0][0], t[0][1]] in start:
                break
        # track[trajectory[-1][0][0], 9] = 4

    # create discrete colormap
    cmap = colors.ListedColormap(['red', 'blue', 'green', 'purple', 'yellow'])
    bounds = [0, 1, 2, 3, 4, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(track, cmap=cmap, norm=norm)

    plt.show()