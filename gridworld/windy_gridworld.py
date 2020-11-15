#! /usr/bin/env python
import time
import numpy as np
from plot import plot_world

GRIDSIZE = [7, 10]
START_STATE = np.array([3, 0])  # Start point
TERMINAL_STATE = np.array([3, 7])  # Location where game terminates
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]  # Vertical push by wind in each column

# Choose regular or Kingmove Action space
ACTIONS_REG = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # Can only move up, down, left, right
ACTIONS_KINGMOVE = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]]  # Above with king move


def step(state, action):
    """
    Given current state and selected action determine next state and reward for that action
    :param state: Current state
    :param action: Selected action
    :return: New state, Action reward
    """
    if np.array_equal(state, TERMINAL_STATE):
        return state, 0.0  # Terminal state reached, reward zero

    wind = np.array([-1*WIND[state[1]], 0])  # Add effect of wind - takes effect when you try to leave space
    next_state = state + action + wind  # Net effect

    if not 0 <= next_state[0] < GRIDSIZE[0] or not 0 <= next_state[1] < GRIDSIZE[1]:
        # outside of grid, state stays same
        if WIND[state[1]] != 0:
            state = np.array([state[0], next_state[1]])  # Need to account for wind here, y stays same, x moves
        reward = -1.0
    elif np.array_equal(next_state, TERMINAL_STATE):
        state = next_state  # Terminal state, reward 0
        reward = 0.0
    else:
        reward = -1.0
        state = next_state

    return state, reward


class World:
    """
    Gridworld class for finding optimal policy to traverse map
    Rewards of -1 are given for each movement except if you reach terminal state where reward is 0
    """

    def __init__(self, probability=0.25, gamma=1.0, alpha=0.5, epsilon=0.1, actions='reg'):
        self.Q = None
        self.V = None
        self.grid = None
        self.gamma = gamma  # Discount
        self.probability = probability  # initial
        self.epsilon = epsilon  # e greedy policy
        self.is_opt = False
        self.state = None
        self.alpha = alpha
        self.trajectory = None

        if actions.lower() == 'reg':
            self.actions = ACTIONS_REG
        elif actions.lower() == 'king':
            self.actions = ACTIONS_KINGMOVE
        else:
            raise ValueError('Please choose "reg" or "king" for action space...')

    def create_world(self, policy='uniform'):
        """
        Create Gridworld given fixed dimensions
        :param policy: optional parameter to set initial policy. Not yet implemented
        """
        self.grid = np.reshape(np.array(range(GRIDSIZE[0] * GRIDSIZE[1])), (GRIDSIZE[0], GRIDSIZE[1]))
        self.Q = np.ndarray((GRIDSIZE[0], GRIDSIZE[1], len(self.actions)))  # Create policy for each option a
        if policy == 'uniform':
            self.Q.fill(self.probability)

        self.Q[TERMINAL_STATE[0], TERMINAL_STATE[1], :] = 0  # Zero for terminal state

    def follow_policy(self, plot=True):
        """
        Follow the optimised policy to reach the terminal state and print each step
        """
        self.reset_episode()
        self.trajectory = []
        while not np.array_equal(self.state, TERMINAL_STATE):
            a0 = self.select_action(test=True)
            print(f"State: {self.state} Action: {self.actions[a0]}")
            self.state, _ = step(self.state, np.array(self.actions[a0]))
            self.trajectory.append(list(self.state))
        print("Terminal State Reached")

        if plot:
            plot_world(np.ones(GRIDSIZE), START_STATE, TERMINAL_STATE, WIND, self.trajectory[:-1])

    def reset_episode(self):
        self.state = START_STATE

    def select_action(self, s=None, test=False):
        """
        Choose action from Q function - randomly so if multiple actions are considered equally good
        :param s: optional state parameter - used for performing lookahead
        :param test: bool parameter to choose policy when testing/training
        :return: action number (index in list of possible actions)
        """
        if s is not None:
            [y, x] = s
        else:
            [y, x] = self.state

        if test:
            greedy = 1.0  # When testing follow policy, not e-greedy.
        else:
            greedy = np.random.rand()
        if greedy > self.epsilon:
            return np.random.choice(np.flatnonzero(self.Q[y, x, :] == self.Q[y, x, :].max()))
        else:
            return np.random.randint(0, len(self.actions))

    def episode(self):
        """
        Run episode from start to Terminal state and update Q function as you go
        """
        self.reset_episode()

        while not np.array_equal(self.state, TERMINAL_STATE):  # Apply TD learning with one step lookahead
            a0 = self.select_action()  # Choose greedy action from current state
            s1, r1 = step(self.state, np.array(self.actions[a0]))  # Calculate new state and reward
            a1 = self.select_action(s1)  # Choose greedy action for next state (one step lookahead)
            # Update Q for current state using reward from current action and discounted Q value from next state
            self.Q[self.state[0], self.state[1], a0] += self.alpha*(r1 + self.gamma*self.Q[s1[0], s1[1], a1]
                                                                    - self.Q[self.state[0], self.state[1], a0])
            self.state = s1  # Update state

    def episode_nsarsa(self, n):
        n = n
        self.reset_episode()
        a = [self.select_action()]  # Action t0
        s = [self.state]  # State t0
        r = [0]  # All rewards added are t+1 so need to start with nonzero array
        T = np.iinfo(np.int64).max  # large number
        t = 0  # Time counter
        while not np.array_equal(self.state, TERMINAL_STATE):  # N step sarsa
            if t < T:

                # Calculate and store next state and reward from action
                si, ri = step(self.state, np.array(self.actions[a[-1]]))
                self.state = si
                s.append(si)
                r.append(ri)

                # Store action if not terminal
                if np.array_equal(self.state, TERMINAL_STATE):
                    T = t + 1
                else:
                    a.append(self.select_action(s[-1]))

            tau = t - n + 1  # Counter to track whether n-step lookahead has occurred
            # Once n step sarsa has been done we need to update weights
            if tau >= 0:
                i = tau + 1
                j = min(tau + n, T) + 1  # Plus 1 to allow for inclusive range below
                g = sum([self.gamma**(ind-tau-1)*r[ind] for ind in range(i, j)])

                if tau + n < T:
                    g += self.gamma**n * self.Q[s[tau+n][0], s[tau+n][1], a[tau+n]]

                self.Q[s[tau][0], s[tau][1], a[tau]] += self.alpha*(g - self.Q[s[tau][0], s[tau][1], a[tau]])

            t += 1  # Update time


if __name__ == "__main__":
    w = World(actions='reg', gamma=0.99)
    w.create_world()
    n_episodes = 10000

    # Time n-step sarsa
    t0 = time.time()
    for e in range(n_episodes):
        w.episode_nsarsa(5)
    t1 = time.time()-t0
    w.follow_policy()  # Print optimal policy

    w.create_world()  # Must reset state-action function
    # Time one-step sarsa
    t2 = time.time()
    for e in range(n_episodes):
        w.episode()
    t3 = time.time() - t2
    w.follow_policy()  # Print optimal policy

    print(f'N-step Sarsa: {t1:.2f}s  One-step sarsa: {t3:.2f}s')
