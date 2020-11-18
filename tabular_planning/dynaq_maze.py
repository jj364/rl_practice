#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from plot import plot_world

GRIDSIZE = [6, 9]
START_STATE = np.array([2, 0])  # Start point
TERMINAL_STATE = np.array([0, 8])  # Location where game terminates
BLOCKS = [[1, 2], [2, 2], [3, 2], [0, 7], [1, 7], [2, 7], [4, 5]]  # Obstructions in the path
ACTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1]]  # Can only move up, down, left, right


def step(state, action):
    """
    Given current state and selected action determine next state and reward for that action
    :param state: Current state
    :param action: Selected action
    :return: New state, Action reward
    """
    if np.array_equal(state, TERMINAL_STATE):
        return state, 1.0  # Terminal state reached

    next_state = state + action

    if not 0 <= next_state[0] < GRIDSIZE[0] or not 0 <= next_state[1] < GRIDSIZE[1] or list(next_state) in BLOCKS:
        # out of bounds or hit obstacle
        return state, 0.0
    elif np.array_equal(next_state, TERMINAL_STATE):
        return next_state, 1.0
    else:
        return next_state, 0.0


class Grid:
    def __init__(self, probability=0.25, gamma=0.95, alpha=0.5, epsilon=0.1):
        self.Q = None
        self.model = {}
        self.grid = None
        self.gamma = gamma  # Discount
        self.probability = probability  # initial
        self.epsilon = epsilon  # e greedy policy
        self.is_opt = False
        self.state = None
        self.alpha = alpha
        self.trajectory = None
        self.actions = ACTIONS

    def create_world(self, policy='uniform'):
        """
        Create Gridworld given fixed dimensions
        :param policy: optional parameter to set initial policy. Not yet implemented
        """
        self.grid = np.reshape(np.array(range(GRIDSIZE[0] * GRIDSIZE[1])), (GRIDSIZE[0], GRIDSIZE[1]))
        self.Q = np.zeros((GRIDSIZE[0], GRIDSIZE[1], len(self.actions)))

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
            plot_world(np.ones(GRIDSIZE), START_STATE, TERMINAL_STATE, BLOCKS, self.trajectory[:-1])

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

    def episode(self, n):
        """
        Run episode from start to Terminal state and update Q function and model as you go
        :param n: Number of model update steps
        """
        self.reset_episode()
        steps = 0
        while not np.array_equal(self.state, TERMINAL_STATE):
            steps += 1
            a0 = self.select_action()  # Choose epsilon-greedy action from current state
            s1, r1 = step(self.state, np.array(self.actions[a0]))  # Calculate new state and reward

            # Update Q for current state using reward from current action and discounted Q value from next state
            self.Q[self.state[0], self.state[1], a0] += self.alpha*(r1 + self.gamma*np.max(self.Q[s1[0], s1[1], :])
                                                                    - self.Q[self.state[0], self.state[1], a0])

            # update model
            l_state = tuple(self.state)
            if l_state not in self.model.keys():
                self.model[l_state] = {}
            self.model[l_state][a0] = (r1, tuple(s1))

            # update q from model - these are the planning steps
            for _ in range(n):
                state = list(self.model.keys())[np.random.randint(0, len(self.model.keys()))]
                action = list(self.model[state].keys())[np.random.randint(0, len(self.model[state].keys()))]

                (model_r, model_s) = self.model[state][action]
                self.Q[state[0], state[1], action] += self.alpha * (
                            model_r + self.gamma * np.max(self.Q[model_s[0], model_s[1], :])
                            - self.Q[state[0], state[1], action])

            self.state = s1  # Update state

        return steps  # Return number of timesteps needed to reach goal for plotting


if __name__ == "__main__":
    w = Grid(alpha=0.5, gamma=0.95, epsilon=0.1)
    w.create_world()
    # Create the world and run for the desired number of episodes
    n_episodes = 50
    n_steps = np.zeros(n_episodes)

    # Start off with 5 planning steps
    for e in range(n_episodes):
        n_steps[e] = w.episode(5)

    w.follow_policy()  # Follow policy, plot learned path

    fig = plt.figure()
    plt.plot(list(range(n_episodes)), n_steps, label="5")

    # Now try with 50 planning steps
    w.create_world()
    n_steps = np.ndarray((n_episodes))

    for e in range(n_episodes):
        n_steps[e] = w.episode(50)
    plt.plot(list(range(n_episodes)), n_steps, label="50")

    w.follow_policy()

    # Direct learning - 0 planning steps
    w.create_world()
    n_steps = np.ndarray((n_episodes))

    for e in range(n_episodes):
        n_steps[e] = w.episode(0)

    w.follow_policy()  # Plot learned path

    plt.plot(list(range(n_episodes)), n_steps, label="0")

    plt.legend()
    plt.title("Convergence of policy with different numbers of planning steps")
    plt.show()