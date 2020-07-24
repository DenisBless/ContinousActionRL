import os
import pathlib
import datetime

import numpy as np
from gym.spaces import Box
import torch
import time
import os.path
from os import path
import gym
import gym_point

import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from common.actor_critic_models import Actor, Critic
from common.replay_buffer import SharedReplayBuffer
from common.arg_parser import ArgParser
from common.utils import get_hparam_dict
from sp_carl.learner import Learner
from sp_carl.sampler import Sampler
from sp_carl.evaluator import Evaluator

CPU = 'cpu'
CUDA = 'cuda:0'

SAVE_MODEL_EVERY = 10
parser = ArgParser()

# # Swimmer
# NUM_ACTIONS = 2
# NUM_OBSERVATIONS = 8
# ACTION_SPACE = Box(low=np.array([-1., -1.]),
#                    high=np.array([1., 1.]))
# OBS_SPACE = None  # observation space is unbounded
# EPISODE_LENGTH = 1000

# # Pendulum
# NUM_ACTIONS = 1
# NUM_OBSERVATIONS = 3
# ACTION_SPACE = Box(low=np.array([-2.]),
#                    high=np.array([2.]))
# OBS_SPACE = Box(low=np.array([-1., -1., -8.]),
#                 high=np.array([1., 1., 8.]))
# EPISODE_LENGTH = 200

# # Point Env
# NUM_ACTIONS = 2
# NUM_OBSERVATIONS = 2
# ACTION_SPACE = Box(low=np.array([-1., -1.]),
#                    high=np.array([1., 1.]))
# OBS_SPACE = Box(low=np.array([-1., -1.]),
#                 high=np.array([1., 1.]))
# EPISODE_LENGTH = 80

# Hopper
NUM_ACTIONS = 3
NUM_OBSERVATIONS = 11
ACTION_SPACE = Box(low=np.array([-1., -1., -1]),
                   high=np.array([1., 1., 1.]))
OBS_SPACE = None  # observation space is unbounded
EPISODE_LENGTH = 1000


if __name__ == '__main__':
    device = CUDA

    model_root_dir = str(pathlib.Path(__file__).resolve().parents[1]) + "/models/"
    model_save_path = model_root_dir + "18_51__24_07/"

    args = parser.parse_args()

    actor = Actor(num_actions=NUM_ACTIONS, num_obs=NUM_OBSERVATIONS, log_std_init=np.log(args.init_std))
    actor.load_state_dict(torch.load(model_save_path + "actor_40"))

    # logger.add_graph(actor, torch.zeros(3))

    # env = gym.make("PointEnv-v0")
    # env = gym.make("Swimmer-v2")
    env = gym.make("Hopper-v2")
    evaluator = Evaluator(actor=actor, argp=args, logger=None, env=env, render=True)

    actor = actor.to(device)

    evaluator.eval(0)

