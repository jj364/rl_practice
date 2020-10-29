#! /usr/bin/env python

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

    def __init__(self, probability=0.25, gamma=1.0, alpha=0.5, actions='reg'):
        self.Q = None
        self.V = None
        self.grid = None
        self.gamma = gamma  # Discount
        self.probability = probability  # initial
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
            a0 = self.select_action()
            print(f"State: {self.state} Action: {self.actions[a0]}")
            self.state, _ = step(self.state, np.array(self.actions[a0]))
            self.trajectory.append(list(self.state))
        print("Terminal State Reached")

        if plot:
            plot_world(np.ones(GRIDSIZE), START_STATE, TERMINAL_STATE, WIND, self.trajectory[:-1])


    def reset_episode(self):
        self.state = START_STATE

    def select_action(self, s=None):
        """
        Choose action from Q function - randomly so if multiple actions are considered equally good
        :param s: optional state parameter - used for performing one-step lookahead
        :return: action number
        """
        if s is not None:
            [y, x] = s
        else:
            [y, x] = self.state
        return np.random.choice(np.flatnonzero(self.Q[y, x, :] == self.Q[y, x, :].max()))

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


if __name__ == "__main__":
    w = World(actions='reg')
    w.create_world()
    n_episodes = 1000

    # Alternate between determining value function and improving policy until optimal
    for e in range(n_episodes):
        w.episode()
    w.follow_policy()  # Print optimal policy
