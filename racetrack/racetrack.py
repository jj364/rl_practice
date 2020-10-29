import numpy as np
import random
import time
from numba import jit
from tqdm import tqdm

from plot_track import show_track

# Space of possible actions - each array is [dvy, dvx]
ACTIONS = [[-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 0], [0, 1],
            [1, -1], [1, 0], [1, 1]]

SPEEDS = 5  # Speeds must be between 0 and N-1


class Track:
    """
    Class to create a track for optimising a left turn
    """
    def __init__(self):
        self.track = None
        self.start = None
        self.finish = None
        self.bound = None
        self.shape = None

    def create_track(self, n_rectangles=4):
        """
        Construct a track by placing partially overlapping rectangles in an L shape
        :param n_rectangles: int Number of overlapping rectangles. Higher n, bigger track
        """
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
                dy = random.randint(15, 20)
            elif r == n_rectangles-1:
                dy = n_finish
                dx = random.randint(15, 20)
            else:
                dx = random.randint(dx, 15)
                dy = random.randint(8, dy)
            rectangles.append([[x, x+dx], [y, y+dy]])
            if r != n_rectangles - 1:
                x += random.randint(int(dx*0.25), int(dx*0.75))
                y += random.randint(int(dy*0.25), int(dy*0.75))

        self.track = np.zeros((y+dy, x+dx))
        self.shape = self.track.shape
        for r in rectangles:
            self.track[r[1][0]:r[1][1], r[0][0]:r[0][1]] = 1
        self.start = [(0, i) for i in range(n_start)]
        self.finish = [(y + dy - i - 1, x + dx - 1) for i in range(n_finish)]


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
        # Randomly initialise state-action values
        self.Q = np.random.rand(track.shape[0], track.shape[1], SPEEDS, SPEEDS, len(ACTIONS)) - 100
        self.C = np.zeros((track.shape[0], track.shape[1], SPEEDS, SPEEDS, len(ACTIONS)))  # Weight sum
        self.e = 0.1  # Epsilon for soft behaviour policy
        self.gamma = 0.95  # Discount factor
        self.track = track
        self.greedy_action = None

    def create_target_policy(self):
        """
        Create greedy policy for target depending on state
        """
        self.target_policy = np.zeros((self.track.shape[0], self.track.shape[1], SPEEDS, SPEEDS), dtype=int)
        for r in range(self.track.shape[0]):
            for c in range(self.track.shape[1]):
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

        else:  # Randomly choose exploratory action
            action = poss_actions[np.random.choice(n_poss)]

        # Update car attributes
        self.action = ACTIONS.index(action)

    def move_car(self):
        """
        Move car according to chosen action
        :return: string Outcome of action
        """
        self.vy += ACTIONS[self.action][0]
        self.vx += ACTIONS[self.action][1]

        # generate small trajectory to prevent corner cutting - probably a better way of doing this!
        t = []
        if self.vy >= self.vx:
            for dy in range(1, self.vy+1):
                y = self.coord[0] + dy
                x = self.coord[1] + round(dy*self.vx/self.vy)
                t.append((y, x))
        else:
            for dx in range(1, self.vx+1):
                x = self.coord[1] + dx
                y = self.coord[0] + round(dx*self.vy/self.vx)
                t.append((y, x))

        self.coord[0] += self.vy
        self.coord[1] += self.vx

        self.reward = -1  # Default reward
        if len(set(t) & set(self.track.finish)) != 0:
            outcome = 'Finish'  # Episode complete
        elif not (0 <= self.coord[0] < self.track.shape[0]) or not (0 <= self.coord[1] < self.track.shape[1]) or \
                len([i for i in t if self.track.track[i[0], i[1]] != 1]) != 0:
            # off track
            outcome = 'Collision'
            self.reward = -50  # Big penalty for hitting boundary
            self.start_car()  # Restart car on line
        else:
            outcome = 'Continue'  # On track, standard reward of -1

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
            (y, x, vy, vx, act, greedy_act), reward = traj[step]  # Unpack trajectory step
            G = self.gamma*G + reward  # Return from action
            self.C[y, x, vy, vx, act] += W  # Update cumulative weight sum
            self.Q[y, x, vy, vx, act] += (W*(G - self.Q[y, x, vy, vx, act]))/(self.C[y, x, vy, vx, act])

            # Update target policy
            self.target_policy[y, x, vy, vx] = np.random.choice(np.flatnonzero(self.Q[y, x, vy, vx] \
                                                                               == self.Q[y, x, vy, vx].max()))

            if act != self.target_policy[y, x, vy, vx]:
                break  # If action is off-policy then importance ratio is 0 so end episode
            else:  # Update importance ratio
                W /= (1 - self.e + self.e/9)

    @jit(nopython=True)
    def generate_episode(self, evaluate=False, specify_start=None):
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
        return ep_trajectory


# Instantiate racetrack  object and create the track
t = Track()
t.create_track(n_rectangles=10)
show_track(t.track, t.start, t.finish)  # Show initial track

# Create car object and initialise target policy
c = Car(t)
c.create_target_policy()
t0 = time.time()
for ep in tqdm(range(10000)):  # Train for 50k episodes

    if ep % 30000 == 0 and ep != 0:  # Visualise policy every so often
        print('\n###########################################')
        print('VISUALISING PATHS')
        for i in range(len(c.track.start)):  # Iterate through each possible start position
            trajectory = c.generate_episode(evaluate=True, specify_start=i)
            print(f"Start Pos: {i}, Path length: {len(trajectory)}")
            show_track(t.track, t.start, t.finish, trajectory)

    else:  # Generate episode as normal with e-greedy behaviour policy
        trajectory = c.generate_episode()
        c.update_vals(trajectory)     # Update policy

print(f'Optimisation time = {time.time()-t0} s')

# Once done let's observe final policy
print('\n###########################################')
print('VISUALISING FINAL POLICY')
for i in range(len(c.track.start)):
    trajectory = c.generate_episode(evaluate=True, specify_start=i)
    print(f"Start Pos: {i}, Path length: {len(trajectory)}")
    show_track(t.track, t.start, t.finish, trajectory)


