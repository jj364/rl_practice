import numpy as np
import random
from tqdm import tqdm
from scipy.ndimage.morphology import binary_dilation

from plot_track import show_track

# Each array is [dvy, dvx]
ACTIONS = [[-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 0], [0, 1],
            [1, -1], [1, 0], [1, 1]]
ROWS = 20
COLS = 20
W_RECT_1 = 15
H_RECT_2 = 15
SPEEDS = 5


class Track:
    def __init__(self):
        self.track = None
        self.start = None
        self.finish = None
        self.bound = None

    def create_track(self, structure='rectangles'):
        if structure == 'rectangles':  # Create from 2 rectangle
            self.track = np.zeros((ROWS, COLS))  # 1 is on track, 0 is off
            # Create track from intersection of 2 rectangles
            r1 = [[0, ROWS], [0, W_RECT_1]]  # [x1, x2], [y1, y2]
            r2 = [[ROWS-H_RECT_2, ROWS], [0, COLS]]
            self.track[r1[0][0]:r1[0][1], r1[1][0]:r1[1][1]] = 1
            self.track[r2[0][0]:r2[0][1], r2[1][0]:r2[1][1]] = 1

            # Generate start and finish coordinates
            self.start = [[0, i] for i in np.where(self.track[0, :] == 1)[0]][1:-1]
            self.finish = []
            for j in range(5):
                for i in np.where(self.track[:, COLS - 1] == 1)[0]:
                    self.finish += [[i, COLS - 1 + j]]

            # Calculate boundary
            k = np.ones((3, 3), dtype=int)
            pad_track = np.pad(self.track, pad_width=1, mode='constant', constant_values=0)
            self.bound = np.logical_and(np.array(binary_dilation(pad_track == 0, k), dtype=int), pad_track)
            self.bound = self.bound[1:-1, 1:-1]  # Remove padding
            for s in self.start: self.bound[s[0], s[1]] = False
            for f in np.where(self.track[:, COLS - 1] == 1): self.bound[f, COLS-1] = False


class Car:
    def __init__(self):
        self.coord = None
        self.vx = None
        self.vy = None
        self.action = None
        self.reward = None
        self.target_policy = None
        self.behaviour_policy = None
        self.Q = np.random.rand(ROWS, COLS, SPEEDS, SPEEDS, len(ACTIONS)) - 2
        self.C = np.zeros((ROWS, COLS, SPEEDS, SPEEDS, len(ACTIONS)))  # Weight sum
        self.e = 0.1  # Epsilon for soft behaviour policy
        self.gamma = 1  # Discount factor
        self.G = 0  # Returns
        self.W = 1  # Importance scaling factor
        self.greedy = None

    def init_car(self, track):
        self.update_target_policy(track.track)
        self.start_episode(track.start)

    def update_target_policy(self, track):
        # Create greedy policy for target
        self.target_policy = np.zeros((ROWS, COLS, SPEEDS, SPEEDS), dtype=int)
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

    def find_poss_actions(self):
        poss_actions = []  # List possible actions given constraints
        for a in ACTIONS:
            if (0 <= self.vy + a[0] < 5) and (0 <= self.vx + a[1] < 5) and not \
                    (self.vy + a[0] == 0 and self.vx + a[1] == 0):
                poss_actions.append(a)
        return poss_actions

    def choose_action(self, explore=True):
        poss_actions = self.find_poss_actions()
        n_poss = len(poss_actions)
        if explore:
            randfloat = random.random()  # Generate float to determine whether greedy action is taken
        else:
            randfloat = 1  # Greedy policy, for evaluation

        if randfloat > self.e and \
                ACTIONS[self.target_policy[self.coord[0], self.coord[1], self.vy, self.vx]] in poss_actions:  #
            action = ACTIONS[self.target_policy[self.coord[0], self.coord[1], self.vy, self.vx]]
            self.greedy = True
        else:
            # Choose exploratory action
            action = poss_actions[np.random.choice(n_poss)]
            self.greedy = False

        self.action = ACTIONS.index(action)
        self.vy += action[0]
        self.vx += action[1]

    def move_car(self, track):
        self.coord[0] += self.vy
        self.coord[1] += self.vx

        if self.coord in track.finish:
            outcome = 'Finish'
            self.reward = 0
        elif not 0 <= self.coord[0] < ROWS or not 0 <= self.coord[1] < COLS or \
                track.track[self.coord[0], self.coord[1]] != 1:
            # on or through boundary
            outcome = 'Collision'
            self.reward = -50
            self.start_car(track.start)
        else:
            self.reward = -1
            outcome = 'Continue'
        return outcome

    def update_vals(self, traj, start):
        self.G = 0
        self.W = 1

        # print('----', len(traj))
        for (y, x, act, vy, vx, greedy), reward in traj[::-1]:
            self.G = self.gamma*self.G + reward
            try:
                self.C[y, x, vy, vx, act] += self.W
            except RuntimeWarning:
                print(self.W)
            # print(y, x, (self.W * (self.G - self.Q[y, x, vy, vx, act])) / (self.C[y, x, vy, vx, act]))
            self.Q[y, x, vy, vx, act] += (self.W*(self.G - self.Q[y, x, vy, vx, act]))/(self.C[y, x, vy, vx, act])
            # print(self.Q[y, x, vy, vx], np.argmax(self.Q[y, x, vy, vx]))
            self.target_policy[y, x, vy, vx] = int(np.argmax(self.Q[y, x, vy, vx]))

            if act != self.target_policy[y, x, vy, vx]:
                break  # Start next episode
            else:  # Update importance ratio
                # print(act, self.target_policy[y, x, vy, vx])
                self.W /= (1 - self.e + self.e/9)


t = Track()
t.create_track()
show_track(t.track, t.start, t.finish)
c = Car()
c.init_car(t)
ei = 0

for ep in tqdm(range(50000)):
    c.start_episode(t.start)
    trajectory = []
    ei += 1
    while True:

        if ei % 10000 == 0:
            c.choose_action(explore=False)
        else:
            c.choose_action()
        state = (c.coord[0], c.coord[1], c.action, c.vy, c.vx, c.greedy)
        result = c.move_car(t)
        trajectory.append([state, c.reward])

        if result == 'Finish':
            break

    if ei%10000 == 0:
        print(ei, len(trajectory))
        print(trajectory)
        t.create_track()
        show_track(t.track, t.start, t.finish, trajectory)
        continue

    c.update_vals(trajectory, t.start)


