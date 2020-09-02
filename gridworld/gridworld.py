#! /usr/bin/env python

import numpy as np
from collections import OrderedDict

GRIDSIZE = [4, 4]
TERMINAL_STATES = [np.array([0,0]),np.array([GRIDSIZE[0]-1,GRIDSIZE[0]-1])]
ACTIONS = OrderedDict({'l': np.array([0,-1]), 'r': np.array([0,1]), 'u': np.array([-1,0]), 'd': np.array([1,0])})


def step(state, action):

    if any((state == t).all() for t in TERMINAL_STATES):
        return state, 0.0

    next_state = state + action

    if next_state[0] in [-1,GRIDSIZE[0]] or next_state[1] in [-1,GRIDSIZE[1]]:
        # outside of grid
        reward = -1.0
    elif any((state == t).all() for t in TERMINAL_STATES):
        state = next_state
        reward = 0.0
    else:
        reward = -1.0
        state = next_state

    return state, reward


class World:
    """
    Gridworld class with optional policy setting
    To begin with policy is equal prob for each of four possible actions
    """

    def __init__(self, probability=0.25, discount=1.0):
        self.policy = np.ndarray((GRIDSIZE[0],GRIDSIZE[1],4))  # Create policy for each option at each state
        self.V = None
        self.grid = None
        self.discount = discount
        self.probability = probability

    def create_world(self, policy='uniform'):
        self.grid = np.reshape(np.array(range(GRIDSIZE[0]*GRIDSIZE[1])), (GRIDSIZE[0],GRIDSIZE[1]))

        if policy == 'uniform':
            self.policy.fill(self.probability)
            pass

    def find_val(self):
        # Start with value fns of all zero
        self.V = np.zeros((GRIDSIZE[0],GRIDSIZE[1]))

        # Iterate until value function converges
        while True:
            new_val = np.zeros((GRIDSIZE[0],GRIDSIZE[1]))
            # iterate over states
            for r in range(GRIDSIZE[0]):
                for c in range(GRIDSIZE[1]):
                    for ai, action in enumerate(ACTIONS.values()):
                        state = np.array([r, c])
                        next_state, reward = step(np.array([r, c]), action)
                        new_val[r, c] += self.policy[state[0],state[1],ai]*(reward + self.discount*self.V[next_state[0],
                                                                                               next_state[1]])
                        pass

            if np.sum(np.abs(self.V-new_val)) < 0.001:
                return new_val
            else:
                self.V = new_val


if __name__ == "__main__":
    w = World()
    w.create_world()
    v = w.find_val()
    print(np.around(v,1))
