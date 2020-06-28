# rl_practice
Repo for teaching myself Reinforcement Learning

All content is written by myself (Jasper James) but mostly shadows examples given in the book *Reinforcement Learning* 2nd Edition by Sutton & Barto so a lot of credit goes to them.

## k_armed_bandit
This folder contains a simple implementation of a *k-armed bandit* problem

Each time step one action from a possible *k* is chosen 

Action selection uses an <img src="https://render.githubusercontent.com/render/math?math=\epsilon">-greedy approach where with probability <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> a random exploratory action is taken and otherwise the action with the highest estimated reward is selected.

Rewards for each action are estimated using sample average. Average actual reward recieved from each time this action was selected.

