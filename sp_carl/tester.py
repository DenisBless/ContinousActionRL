import os
import datetime
import torch
import pathlib
import numpy as np
from common.actor_critic_models import Actor
from sp_carl.evaluator import Evaluator
from common.arg_parser import ArgParser
from gym.spaces.box import Box

parser = ArgParser()

# Pendulum
NUM_ACTIONS = 1
NUM_OBSERVATIONS = 3
ACTION_SPACE = Box(low=np.array([-2.]),
                   high=np.array([2.]))
OBS_SPACE = Box(low=np.array([-1., -1., -8.]),
                high=np.array([1., 1., 8.]))
EPISODE_LENGTH = 200

if __name__ == '__main__':
    args = parser.parse_args()

    model_dir = str(pathlib.Path(__file__).resolve().parents[1]) + "/models/" + "14_32__20_07/actor_230"

    actor = Actor(num_actions=NUM_ACTIONS, num_obs=NUM_OBSERVATIONS, log_std_init=np.log(args.init_std))

    actor.load_state_dict(torch.load(model_dir))

    evaluator = Evaluator(actor=actor, argp=args, render=True)
    evaluator.num_samples = 50
    evaluator.eval()
