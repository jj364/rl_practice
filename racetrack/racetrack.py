import numpy as np
import random
import math
from tqdm import tqdm
from scipy.ndimage.morphology import binary_dilation

from plot_track import show_track

# Space of possible actions - each array is [dvy, dvx]
ACTIONS = [[-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 0], [0, 1],
            [1, -1], [1, 0], [1, 1]]
# Remaining constants
ROWS = 50  # Height of track
COLS = 50  # Width of track
W_RECT_1 = 6  # Create track from 2 rectangles - 1st is vertical with this width
H_RECT_2 = 6  # 2nd is horizontal with this height
SPEEDS = 3  # Speeds must be between 0 and N-1


class Track:
    """
    Class to create a track for optimising a left turn
    """
    def __init__(self):
        self.track = None
        self.start = None
        self.finish = None
        self.bound = None

    def create_track(self, structure='two_rectangles', n_rectangles=4):
        """
        Construct a track by placing 2 overlapping rectangles in an L shape
        :param structure: Method of constructing track, currently only rectangle implemented
        """
        if structure == 'two_rectangles':  # Create from 2 rectangle
            self.track = np.zeros((ROWS, COLS))  # 1 is on track, 0 is off
            # Create track from intersection of 2 rectangles
            r1 = [[0, ROWS], [0, W_RECT_1]]  # [x1, x2], [y1, y2]
            r2 = [[ROWS-H_RECT_2, ROWS], [0, COLS]]
            self.track[r1[0][0]:r1[0][1], r1[1][0]:r1[1][1]] = 1
            self.track[r2[0][0]:r2[0][1], r2[1][0]:r2[1][1]] = 1

            # Generate start and finish coordinates
            self.start = [[0, i] for i in np.where(self.track[0, :] == 1)[0]][1:-1]
            self.finish = []
            for j in range(5):  # Need to extend finish beyond finish line as speed can be > 1
                for i in np.where(self.track[:, COLS - 1] == 1)[0]:
                    self.finish += [[i, COLS - 1 + j]]

        elif structure == 'random':
            if n_rectangles < 2:
                raise Exception("Choose at least 2 rectangles to construct track")

            n_start = random.randint(5, 10)  # Generate start/finish line size
            n_finish = random.randint(5, 10)
            x = 0
            y = 0
            rectangles = []  # structure is [[x1,x2],[y1,y2]]

            for r in range(n_rectangles):  # create n overlapping rectangles
                # create rectangle width and height
                if x == 0:
                    dx = n_start
                else:
                    dx = random.randint(8, 15)
                if r == n_rectangles-1:
                    dy = n_finish
                else:
                    dy = random.randint(8, 15)
                rectangles.append([[x, x+dx], [y, y+dy]])
                if r != n_rectangles - 1:
                    x += random.randint(int(dx*0.25), int(dx*0.75))
                    y += random.randint(int(dy*0.25), int(dy*0.75))

            self.track = np.zeros((y+dy, x+dx))
            for r in rectangles:
                self.track[r[1][0]:r[1][1], r[0][0]:r[0][1]] = 1
            self.start = [[0, i] for i in range(n_start)]
            self.finish = []
            for j in range(5):
                for i in range(n_finish):
                    self.finish += [[y+dy-i-1, x+dx+j-1]]


class Car:
    """
    Class for the car which runs on the above track - passed in as variable to become attribute
    """
    def __init__(self, track):
        self.coord = None
        self.vx = None
        self.vy = None
        self.action = None
        self.reward = None
        self.target_policy = None
        self.behaviour_policy = None
        self.Q = np.random.rand(ROWS, COLS, SPEEDS, SPEEDS, len(ACTIONS)) - 100  # Randomly initialise state-action
        # values
        self.C = np.zeros((ROWS, COLS, SPEEDS, SPEEDS, len(ACTIONS)))  # Weight sum
        self.e = 0.1  # Epsilon for soft behaviour policy
        self.gamma = 0.99  # Discount factor
        self.track = track

    def create_target_policy(self):
        """
        Create greedy policy for target depending on state
        """
        self.target_policy = np.zeros((ROWS, COLS, SPEEDS, SPEEDS), dtype=int)
        for r in range(ROWS):
            for c in range(COLS):
                if self.track.track[r, c]:
                    for i in range(SPEEDS):
                        for j in range(SPEEDS):
                            self.target_policy[r, c, i, j] = \
                                np.random.choice(np.flatnonzero(self.Q[r, c, i, j] == self.Q[r, c, i, j].max()))

    def start_car(self, start_pos=None):
        """
        Initialise car parameters at start of episode
        :param start_pos: int: Optional starting position of car on start line. Index in self.track.start
        """
        self.vx = 0  # Speed zero
        self.vy = 0
        if start_pos is None:
            start_pos = np.random.choice(len(self.track.start))  # Random position on startline
        self.coord = [self.track.start[start_pos][0], self.track.start[start_pos][1]]

    def find_poss_actions(self):
        """
        Select possible actions given car's attributes and constraints on speed
        :return: list of possible actions
        """
        poss_actions = []  # List possible actions given constraints
        for a in ACTIONS:  # Speed cant be 0 and each velocity component must be between 0 and 5
            if (0 <= self.vy + a[0] < SPEEDS) and (0 <= self.vx + a[1] < SPEEDS) and not \
                    (self.vy + a[0] == 0 and self.vx + a[1] == 0):
                poss_actions.append(a)
        return poss_actions

    def choose_action(self, explore=True):
        """
        Choose possible action according to epsilon-greedy behaviour policy.
        Behaviour policy is to choose target policy with probability e and random choice with probability 1-e
        :param explore: Binary parameter to decide whether to take deterministic policy - False is used for evaluation
        """
        poss_actions = self.find_poss_actions()  # Due to constraints only certain actions are possible
        n_poss = len(poss_actions)
        if explore:
            randfloat = random.random()  # Generate float to determine whether greedy action is taken
        else:
            randfloat = 1  # Greedy policy, for evaluation

        self.greedy_action = self.target_policy[self.coord[0], self.coord[1], self.vy, self.vx]
        #  Follow greedy policy with probability epsilon IF move is possible
        if randfloat > self.e and \
                ACTIONS[self.target_policy[self.coord[0], self.coord[1], self.vy, self.vx]] in poss_actions:  #
            action = ACTIONS[self.target_policy[self.coord[0], self.coord[1], self.vy, self.vx]]
        #  Randomly choose exploratory action
        else:
            self.not_poss += 1
            action = poss_actions[np.random.choice(n_poss)]

        # Update car attributes
        self.action = ACTIONS.index(action)

    def move_car(self):
        """
        Move car according to chosen action
        :return: Outcome of action
        """
        self.vy += ACTIONS[self.action][0]
        self.vx += ACTIONS[self.action][1]
        self.coord[0] += self.vy
        self.coord[1] += self.vx

        if self.coord in self.track.finish:
            outcome = 'Finish'
            self.reward = -1  # Episode complete
        elif not (0 <= self.coord[0] < ROWS) or not (0 <= self.coord[1] < COLS) or \
                self.track.track[self.coord[0], self.coord[1]] != 1:
            # on or through boundary
            outcome = 'Collision'
            self.reward = -50  # Big penalty for hitting boundary
            self.start_car()  # Restart car on line
        else:
            # On track, standard reward of -1
            self.reward = -1
            outcome = 'Continue'

        return outcome

    def update_vals(self, traj):
        """
        Update learning values after each episode
        :param traj: Trajectory of car including position, action and velocity (y, x, action_number, vy, vx)
        """
        G = 0  # Returns
        W = 1  # Importance scaling factor

        # Update policy and state-action values for each step in (reversed) trajectory
        n_steps = len(traj)
        for step in range(n_steps-1 , -1, -1):
            (y, x, vy, vx, act, greedy_act), reward = traj[step]
            # print(y, x, vy, vx, act, greedy_act)
            G = self.gamma*G + reward
            self.C[y, x, vy, vx, act] += W
            # print(W, G, self.Q[y, x, vy, vx, act], self.C[y, x, vy, vx, act], (W*(G - self.Q[y, x, vy, vx, \
            #                 act]))/(self.C[y, x, vy, vx, act]))
            self.Q[y, x, vy, vx, act] += (W*(G - self.Q[y, x, vy, vx, act]))/(self.C[y, x, vy, vx, act])
            self.target_policy[y, x, vy, vx] = np.random.choice(np.flatnonzero(self.Q[y, x, vy, vx] == self.Q[y, x, vy, vx].max()))

            # if self.target_policy[y, x, vy, vx] != greedy_act:
            #     print('O', y, x, vy, vx, greedy_act, self.target_policy[y, x, vy, vx])

            if act != self.target_policy[y, x, vy, vx]:
                # print('OFF POL', G)
                break  # If action is off-policy then importance ratio is 0 so end episode
            else:  # Update importance ratio
                W /= (1 - self.e + self.e/9)
                # print('!',W)

    def generate_episode(self, evaluate=False, specify_start=None):
        self.not_poss = 0
        self.start_car(specify_start)  # Initialise car variables at start of episode
        ep_trajectory = []
        while True:  # Iterate until finish is reached

            if evaluate:  # Show policy path every 10k episodes
                self.choose_action(explore=False)
            else:
                self.choose_action()

            # Add car attributes to trajectory
            state = (self.coord[0], self.coord[1], self.vy, self.vx, self.action, self.greedy_action)
            result = self.move_car()
            ep_trajectory.append([state, c.reward])

            if result == 'Finish':
                break
        # print(self.not_poss/len(ep_trajectory))
        return ep_trajectory


# Instantiate racetrack  object and create the track
t = Track()
t.create_track(structure='random', n_rectangles=5)
show_track(t.track, t.start, t.finish)
quit()

# Create car object and initialise target policy
c = Car(t)
c.create_target_policy()
for ep in tqdm(range(50000)):  # Train for 50k episodes

    if ep % 10000 == 0 and ep != 0:  # Visualise policy every 10k
        for i in range(len(c.track.start)):
            print('--------------------------------------------------------')
            trajectory = c.generate_episode(evaluate=True, specify_start=i)
            print(len(trajectory), trajectory[-5:])
            t.create_track()  # Clean previous paths
            show_track(t.track, t.start, t.finish, trajectory)
    else:  # Generate episode as normal with e-greedy behaviour policy
        trajectory = c.generate_episode()
        c.update_vals(trajectory)     # Update policy

# Once done let's observe final policy
for i in range(len(c.track.start)):
    trajectory = c.generate_episode(evaluate=True, specify_start=i)
    print(len(trajectory), trajectory)
    t.create_track()  # Clean previous paths
    show_track(t.track, t.start, t.finish, trajectory)


