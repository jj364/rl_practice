import numpy as np
import random
from scipy.ndimage.morphology import binary_dilation

# Each array is [dvy, dvx]
ACTIONS = [[-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 0], [0, 1],
            [1, -1], [1, 0], [1, 1]]
ROWS = 10
COLS = 10
SPEEDS = 5


class Track:
    def __init__(self):
        self.track = None
        self.start = None
        self.finish = None
        self.bound = None

    def create_track(self, structure='rectangles'):
        w = 10
        h = 10
        if structure == 'rectangles':  # Create from 2 rectangle
            self.track = np.zeros((h, w))  # 1 is on track, 0 is off
            # Create track from intersection of 2 rectangles
            r1 = [[0, h], [0, 5]]
            r2 = [[6, h], [0, w]]
            self.track[r1[0][0]:r1[0][1], r1[1][0]:r1[1][1]] = 1
            self.track[r2[0][0]:r2[0][1], r2[1][0]:r2[1][1]] = 1

            # Generate start and finish coordinates
            self.start = [[0, i] for i in np.where(self.track[0, :] == 1)[0]][1:-1]
            self.finish = []
            for j in range(5):
                for i in np.where(self.track[:, w - 1] == 1)[0]:
                    self.finish += [[i, w - 1 + j]]

            # Calculate boundary
            k = np.ones((3, 3), dtype=int)
            pad_track = np.pad(self.track, pad_width=1, mode='constant', constant_values=0)
            self.bound = np.logical_and(np.array(binary_dilation(pad_track == 0, k), dtype=int), pad_track)
            self.bound = self.bound[1:-1, 1:-1]  # Remove padding
            for s in self.start: self.bound[s[0], s[1]] = False
            for f in np.where(self.track[:, w - 1] == 1): self.bound[f, COLS-1] = False


class Car:
    def __init__(self):
        self.coord = None
        self.vx = None
        self.vy = None
        self.reward = None
        self.target_policy = None
        self.behaviour_policy = None
        self.Q = np.random.rand(ROWS, COLS, SPEEDS, SPEEDS, len(ACTIONS))
        self.C = np.zeros((ROWS, COLS, SPEEDS, SPEEDS, len(ACTIONS)))
        self.e = 0.1  # Epsilon for soft behaviour policy

    def init_car(self, track):
        self.update_target_policy(track.track)
        self.start_episode(track.start)

    def update_target_policy(self, track):
        # Create greedy policy for target
        self.target_policy = np.zeros((ROWS, COLS, SPEEDS, SPEEDS),dtype=int)
        for r in range(ROWS):
            for c in range(COLS):
                if track[r, c]:
                    for i in range(SPEEDS):
                        for j in range(SPEEDS):
                            self.target_policy[r, c, i, j] = int(np.argmax(self.Q[r, c, i, j]))

    def start_episode(self, track_start):
        self.reward = 0
        self.start_car(track_start)

    def start_car(self, track_start):
        self.vx = 0
        self.vy = 0
        c = np.random.choice(len(track_start))
        self.coord = [track_start[c][0], track_start[c][1]]

    def choose_action(self):
        poss_actions = []  # List possible actions given constraints
        for a in ACTIONS:
            if 0 <= self.vy + a[0] < 5 and 0 <= self.vx + a[1] < 5 and not \
                    (self.vy + a[0] == 0 and self.vx + a[1] == 0):
                poss_actions.append(a)
        n_poss = len(poss_actions)
        randfloat = random.random()  # Generate float to determine whether greedy action is taken

        if randfloat < self.e:  # Choose exploratory action
            action = poss_actions[np.random.choice(n_poss)]
        else:
            action = ACTIONS[self.target_policy[self.coord[0], self.coord[1], self.vy, self.vx]]
            if action not in poss_actions:
                action = poss_actions[np.random.choice(n_poss)]
        self.vy += action[0]
        self.vx += action[1]

    def move_car(self, track):
        self.reward = 1
        self.coord[0] += self.vy
        self.coord[1] += self.vx

        if self.coord in track.finish:
            outcome = 'Finish'
            self.reward = 0
        elif not 0 <= self.coord[0] <= 9 or not 0 <= self.coord[1] <= 9 or \
                track.bound[self.coord[0], self.coord[1]]:
            # on or through boundary
            outcome = 'Collision'
            self.start_car(track.start)
        else:
            outcome = 'Continue'
        return outcome


t = Track()
t.create_track()
c = Car()
c.init_car(t)

trajectory =[]
while True:
    c.choose_action()
    state = [c.coord[0], c.coord[1], c.vy, c.vx]
    result = c.move_car(t)
    trajectory.append(state+[c.reward])
    if result == 'Finish':
        break
print(trajectory)
print(len(trajectory))
