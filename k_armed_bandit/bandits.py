#!/usr/bin/env python

"""
Simple k-armed bandit problem for reinforcement learning.
Each time step choose one action from k options
Using sample averages to estimate reward of each action.
Epsilon-greedy action selection tested with several epsilon values
All code written by Jasper James
Problem adapted from 'Reinforcement Learning' by Sutton & Barto
"""

import random
import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    """
    Bandit class for creating actions. Each Bandit object is a possible action

    Args:
        std_r (float): Actual reward for action is chosen from Gaussian dist about mean reward with standard dev std_r
    """
    def __init__(self, std_r):
        self.q_star = None  # Actual value
        self.est_q = None  # Estimated reward
        self.reward = None  # Actual reward recieved from action
        self.std_r = std_r
        self.n_selections = 0

    def init_q(self, mean_q, std_q, est_q):
        """
        Initialise action properties by choosing true reward from Gaussian distribution
        :param mean_q: Mean of Gaussian from which to choose true action reward
        :param std_q: Standard dev of Gaussian
        :param est_q: Initial estimate of action reward -- here we start at zero
        """
        self.q_star = np.random.normal(mean_q, std_q)  # Choose actual reward from gaussian
        self.est_q = est_q  # Set initial estimate for reward

    def update_reward(self):
        """
        Generate reward from the selected action
        Reward is Gaussian where mean is the actual reward of action (q_star) and standard dev is std_r
        Update estimated agent reward by doing weighted average of all rewards for this action
        :return: reward generated from Gaussian distribution
        """
        self.n_selections += 1
        self.reward = np.random.normal(self.q_star, self.std_r)  # Generate random reward close to actual
        self.est_q = self.est_q + (self.reward-self.est_q)/self.n_selections  # Update estimated value
        return self.reward


def create_bandits(n_bandits=10, mean_q=0, std_q=1, std_r=1, est_q=0):
    """
    Generate all the actions (bandits)
    :param n_bandits: Number of possible actions
    :param mean_q: See Bandit Class init_q method
    :param std_q: See Bandit Class init_q method
    :param std_r: See Bandit Class description
    :param est_q: Initial estimate of action reward
    :return:
    """
    agents = []  # Bandits called agents here to avoid name shadowing
    for _ in range(n_bandits):  # Instantiate all bandits
        b = Bandit(std_r)
        b.init_q(mean_q, std_q, est_q)
        agents.append(b)
    return agents


def select_action(agents, epsilon):  # Bandits called agents here to avoid name shadowing
    """
    Time step - Either choose action estimated to give highest reward or randomly select exploratory action
    :param agents: All possible actions
    :param epsilon: Probability of exploring by choosing random action
    :return:
    """
    random_float = random.random()  # Choose random float in range [0,1]
    if random_float < epsilon:  # With probability epsilon explore random action
        # Greedy action epsilon = 0 so greedy action always taken
        action = random.randint(0, len(agents) - 1)
    else:
        est_rewards = [b.est_q for b in agents]
        max_rewards = np.argwhere(np.array(est_rewards) == max(est_rewards))  # Check how many max rewards are the same
        if len(max_rewards) == 1:
            action = max_rewards[0][0] # Choose action with highest estimated reward
        else:  # Randomly select one of the top rewards
            action = random.choice([i[0] for i in max_rewards])

    return agents[action].update_reward()  # Return actual reward given


def plot_rewards(data, es, n_tests):
    """
    Plot rewards from stored data
    """
    [plt.plot(data[i] / n_tests, label='Epsilon = ' + str(es[i])) for i in range(len(es))]
    plt.legend(loc='lower right')
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.title("Average reward from bandits using an epsilon-greedy approach")
    plt.show()


if __name__ == '__main__':

    """ Change the following three variables to explore different situations """
    n_tests = 1000  # Number of different bandits to test
    n_steps = 2000  # Test each set of bandits this many times
    epsilons = [0, 0.01, 0.1]  # Test 3 values of epsilon
    """"""
    rewards = np.zeros((len(epsilons), n_steps))  # Array to store awards for plotting

    """ Loop over different strategies, sets of bandits and actions"""
    for ie, e in enumerate(epsilons):
        for test in range(n_tests):
            bandits = create_bandits()  # Create bandits, use defaults
            for s in range(n_steps):
                rewards[ie, s] += select_action(agents=bandits, epsilon=e)
            # [print("Est: %f  Actual: %f" % (b.est_q, b.q_star)) for b in bandits]

    plot_rewards(rewards, epsilons, n_tests)

    """
    This is just a very simple implementation of the k-armed bandit but shows the effect of different epsilon values.
    Problem could be enhanced by testing time-varying actual reward values for each action or any of the other 
    options in chapter 1 of Sutton & Barto
    """
