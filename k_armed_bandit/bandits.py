import numpy as np
import random

class Bandit:
    def __init__(self, mean_r, std_r):
        self.q_star = None  # Actual value
        self.est_q = None
        self.reward = None
        self.mean_r = mean_r
        self.std_r = std_r
        self.n_selections = 0

    def init_q(self, mean_q, std_q, est_q):
        self.q_star = np.random.normal(mean_q, std_q)  # Choose actual reward from gaussian
        self.est_q = est_q  # Set initial estimate for reward

    def update_reward(self):
        self.n_selections += 1
        self.reward = np.random.normal(self.q_star, self.std_r)  # Generate random reward close to actual
        self.est_q = self.est_q + (self.reward-self.est_q)/self.n_selections  # Update estimated value

def create_bandits(n_bandits=10, mean_q=0, std_q=1, mean_r=0, std_r=1, est_q=0):
    bandits = []
    for _ in range(n_bandits):  # Instantiate all bandits
        b = Bandit(mean_r, std_r)
        b.init_q(mean_q, std_q, est_q)
        bandits.append(b)

def select_action(bandits, epsilon):
    random_float = random.random()
    if random_float <= epsilon:  # With probability epsilon explore random action

    else:
        action =




if __name__ == '__main__':
