#!/usr/bin/env python

"""
At this point it makes more sense to use established environments so I'm loading openai-gym
"""
import gym

env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())  # Take random action
env.close()