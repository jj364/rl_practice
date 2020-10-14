#! /usr/bin/env python

import numpy as np
from collections import OrderedDict

# from numpy.core._multiarray_umath import ndarray

GRIDSIZE = [5, 5]
TERMINAL_STATES = [np.array([0, 0]), np.array([GRIDSIZE[0] - 1, GRIDSIZE[0] - 1])]  # Locations where game terminates
ACTIONS = OrderedDict({'l': np.array([0, -1]), 'r': np.array([0, 1]), 'u': np.array([-1, 0]), 'd': np.array([1, 0])})
# Can only move up, down, left, right


def step(state, action):
    """
    Given current state and selected action determine next state and reward for that action
    :param state: Current state
    :param action: Selected action
    :return: New state, Action reward
    """
    if any((state == t).all() for t in TERMINAL_STATES):
        return state, 0.0

    next_state = state + action

    if next_state[0] in [-1, GRIDSIZE[0]] or next_state[1] in [-1, GRIDSIZE[1]]:
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
    Gridworld class for finding optimal policy to traverse map
    Rewards of -1 are given for each movement except if you reach top left or bottom right where reward = 0
    """

    def __init__(self, probability=0.25, discount=1.0):
        self.policy = np.ndarray((GRIDSIZE[0], GRIDSIZE[1], 4))  # Create policy for each option at each state
        self.V = None
        self.grid = None
        self.discount = discount
        self.probability = probability
        self.is_opt = False

    def create_world(self, policy='uniform'):
        """
        Create Gridworld given fixed dimensions
        :param policy: optional parameter to set initial policy. Not yet implemented
        """
        self.grid = np.reshape(np.array(range(GRIDSIZE[0] * GRIDSIZE[1])), (GRIDSIZE[0], GRIDSIZE[1]))

        if policy == 'uniform':
            self.policy.fill(self.probability)

    def find_val(self):
        """
        Iterative function to determine value function of a policy
        """
        # Start with value fns of all zero
        self.V = np.zeros((GRIDSIZE[0], GRIDSIZE[1]))

        # Iterate until value function converges
        while True:
            new_val = np.zeros((GRIDSIZE[0], GRIDSIZE[1]))
            # iterate over states
            for r in range(GRIDSIZE[0]):
                for c in range(GRIDSIZE[1]):
                    # Iterate over actions
                    for ai, action in enumerate(ACTIONS.values()):
                        state = np.array([r, c])
                        next_state, reward = step(np.array([r, c]), action)

                        # Value function calculates likely total reward from being at a location
                        # Policy should be chosen by determining the path through states which maximises reward
                        new_val[r, c] += self.policy[state[0], state[1], ai] * (reward +
                                                                                self.discount * self.V[
                                                                                    next_state[0], next_state[1]])

            if np.sum(np.abs(self.V - new_val)) < 0.001:
                return new_val
            else:
                self.V = new_val

    def policy_iteration(self, value_func):
        """
        Method to determine improved policy in gridworld. Starting from a value function iterate over states and
        actions to find an improved policy
        :param value_func: Converged value function from before
        :return Bool: True if policy optimal, False otherwise
        """
        temp_policy = np.zeros((GRIDSIZE[0], GRIDSIZE[1], 4))
        value_func = np.around(value_func, 1)  # Round value function to find actions which are equally good

        # Iterate over all states
        for r in range(GRIDSIZE[0]):
            for c in range(GRIDSIZE[1]):
                state_rewards = []
                # Iterate over all actions from current location to determine best action
                for ai, action in enumerate(ACTIONS.values()):
                    next_state, _ = step(np.array([r, c]), action)  # Get next state for each action
                    reward = value_func[next_state[0], next_state[1]]
                    state_rewards.append(reward)
                best_actions = np.where(np.array(state_rewards) == max(state_rewards))  # Create list of best actions
                # for given state - multiple actions can be equally good
                for a in best_actions:
                    temp_policy[r, c, a] = 1/len(best_actions[0])  # Normalise probability

        # If new policy is same as old then we have found optimal policy
        if np.array_equal(temp_policy, self.policy):
            return True
        else:
            self.policy = temp_policy
            return False


if __name__ == "__main__":
    w = World()
    w.create_world()

    # Alternate between determining value function and improving policy until optimal
    while True:
        v = w.find_val()
        is_opt = w.policy_iteration(v)
        if is_opt:
            break

    print(np.around(v, 1))
    print(w.policy)