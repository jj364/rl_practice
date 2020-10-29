# rl_practice
Repo for teaching myself Reinforcement Learning

All content is written by myself (Jasper James) but mostly shadows examples given in the book *Reinforcement Learning* 2nd Edition by Sutton & Barto so credit for the examples goes to them.

## k_armed_bandit
This folder contains a simple implementation of a *k-armed bandit* problem

Each time step one action from a possible *k* is chosen 

Action selection uses an <img src="https://render.githubusercontent.com/render/math?math=\epsilon">-greedy approach where with probability <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> a random exploratory action is taken and otherwise the action with the highest estimated reward is selected.

Rewards for each action are estimated using sample average. Average actual reward recieved from each time this action was selected.

## Gridworld - Generalised Policy Iteration
Contains two examples of learning using the *gridworld* environment. Agent is on a grid with certain action space and must find optimal trajectory to the terminal state.

In the *vanilla* gridworld Generalised Policy Iteration (GPI) is used to find the optimal path from start to finish. The entire state-action space is sampled for the grid.

In *windy* gridworld a Southerly wind pushes the agent up the grid to add complexity to the environment. Temporal difference learning is used to find the optimal path from start to finish. 

## Racetrack - Off-policy Monte Carlo
Off-policy Monte Carlo control is used to teach a car to make a turn on a racetrack with restrictions on maximum speed.

A racetrack is randomly generated and the car learns the optimal policy over several thousand episodes. The state-action space is too large to sample it all so Monte Carlo sampling is needed.
