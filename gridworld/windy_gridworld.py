#! /usr/bin/env python

import numpy as np
from collections import OrderedDict

# from numpy.core._multiarray_umath import ndarray

GRIDSIZE = [7, 10]
TERMINAL_STATE = np.array([3, 7])  # Location where game terminates
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]  # Vertical push by wind in each column

# Choose regular or Kingmove Action space
# ACTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # Can only move up, down, left, right
ACTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]]  # Above with king move


def step(state, action):
    """
    Given current state and selected action determine next state and reward for that action
    :param state: Current state
    :param action: Selected action
    :return: New state, Action reward
    """
    if np.array_equal(state, TERMINAL_STATE):
        return state, 0.0

    wind = np.array([-1*WIND[state[1]], 0])  # Add effect of wind
    next_state = state + action + wind

    if not 0 <= next_state[0] < GRIDSIZE[0] or not 0 <= next_state[1] < GRIDSIZE[1]:
        # outside of grid, state stays same
        if WIND[state[1]] != 0:
            state = np.array([state[0], next_state[1]])  # Need to account for wind here, y stays same, x moves
        reward = -1.0
    elif np.array_equal(next_state, TERMINAL_STATE):
        state = next_state
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

    def __init__(self, probability=0.25, gamma=1.0, alpha=0.5):
        self.Q = None
        self.V = None
        self.grid = None
        self.gamma = gamma  # Discount
        self.probability = probability  # initial
        self.is_opt = False
        self.state = None
        self.alpha = alpha

    def create_world(self, policy='uniform'):
        """
        Create Gridworld given fixed dimensions
        :param policy: optional parameter to set initial policy. Not yet implemented
        """
        self.grid = np.reshape(np.array(range(GRIDSIZE[0] * GRIDSIZE[1])), (GRIDSIZE[0], GRIDSIZE[1]))
        self.Q = np.ndarray((GRIDSIZE[0], GRIDSIZE[1], len(ACTIONS)))  # Create policy for each option a
        if policy == 'uniform':
            self.Q.fill(self.probability)

        self.Q[TERMINAL_STATE[0], TERMINAL_STATE[1], :] = 0  # Zero for terminal state

    def follow_policy(self):
        self.reset_episode()
        print(self.state)
        while not np.array_equal(self.state, TERMINAL_STATE):
            a0 = self.select_action()
            self.state, _ = step(self.state, np.array(ACTIONS[a0]))
            print(self.state)

    def reset_episode(self):
        self.state = np.array([3, 0])

    def select_action(self, s=None):
        if s is not None:
            [y, x] = s
        else:
            [y, x] = self.state
        return np.random.choice(np.flatnonzero(self.Q[y, x, :] == self.Q[y, x, :].max()))

    def episode(self):
        self.reset_episode()

        while not np.array_equal(self.state, TERMINAL_STATE):
            a0 = self.select_action()
            s1, r1 = step(self.state, np.array(ACTIONS[a0]))
            a1 = self.select_action(s1)
            self.Q[self.state[0], self.state[1], a0] += self.alpha*(r1 + self.gamma*self.Q[s1[0], s1[1], a1]
                                                                    - self.Q[self.state[0], self.state[1], a0])
            self.state = s1


if __name__ == "__main__":
    w = World()
    w.create_world()
    n_episodes = 1000

    # Alternate between determining value function and improving policy until optimal
    for e in range(n_episodes):
        w.episode()
    w.follow_policy()  # Print optimal policy
