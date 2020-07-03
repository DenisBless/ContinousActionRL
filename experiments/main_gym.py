import gym
import numpy as np

env = gym.make("FetchReach-v1")
env.reset()
env.step(np.array([ 0 ,0 ,0, 0]))