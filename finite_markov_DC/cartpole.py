#!/usr/bin/env python

"""
At this point it makes more sense to use established environments so I'm loading openai-gym
"""
import gym

env = gym
env.reset()
action = env.action_space.sample() # Random action
for i in range(1000):
    env.render()
    observation, reward, done, info = env.step(action)

    if observation[2] > 0:
        action = 1
    else:
        action = 0
    print(i, reward)
    if done:
        break

env.close()