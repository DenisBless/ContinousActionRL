import os
import pathlib
import datetime

import numpy as np
from gym.spaces import Box
import torch
import time
import os.path
from os import path

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

# Hopper
# NUM_ACTIONS = 3
# NUM_OBSERVATIONS = 11
# ACTION_SPACE = Box(low=np.array([-1., -1., -1]),
#                    high=np.array([1., 1., 1.]))
# OBS_SPACE = None  # observation space is unbounded
# EPISODE_LENGTH = 1000


# Swimmer
# NUM_ACTIONS = 2
# NUM_OBSERVATIONS = 8
# ACTION_SPACE = Box(low=np.array([-1., -1.]),
#                    high=np.array([1., 1.]))
# OBS_SPACE = None  # observation space is unbounded
# EPISODE_LENGTH = 1000

# Pendulum
# NUM_ACTIONS = 1
# NUM_OBSERVATIONS = 3
# ACTION_SPACE = Box(low=np.array([-2.]),
#                    high=np.array([2.]))
# OBS_SPACE = Box(low=np.array([-1., -1., -8.]),
#                 high=np.array([1., 1., 8.]))
# EPISODE_LENGTH = 200

# Reacher
NUM_ACTIONS = 3
NUM_OBSERVATIONS = 9
ACTION_SPACE = Box(low=np.array([-1., -1., -1]),
                   high=np.array([1., 1., 1.]))
EPISODE_LENGTH = 360

def work(replay_buffer, actor, parser_args):
    worker = Sampler(replay_buffer=replay_buffer,
                     actor=actor,
                     argp=parser_args)
    worker.run()


if __name__ == '__main__':
    device = CUDA if torch.cuda.is_available() else CPU

    model_root_dir = str(pathlib.Path(__file__).resolve().parents[1]) + "/models/"
    if not os.path.isdir(model_root_dir):
        os.mkdir(model_root_dir)
    model_dir = model_root_dir + datetime.datetime.now().strftime("%H_%M__%d_%m")
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lock = mp.Lock()
    args = parser.parse_args()
    logger = SummaryWriter()
    # logger.add_hparams(get_hparam_dict(args), {'mean reward': 0})

    actor = Actor(num_actions=NUM_ACTIONS, num_obs=NUM_OBSERVATIONS, log_std_init=np.log(args.init_std))

    # logger.add_graph(actor, torch.zeros(3))

    actor.share_memory()

    critic = Critic(num_actions=NUM_ACTIONS, num_obs=NUM_OBSERVATIONS)

    critic.share_memory()

    print(args.batch_size)

    shared_replay_buffer = SharedReplayBuffer(capacity=args.replay_buffer_size,
                                              trajectory_length=EPISODE_LENGTH,
                                              num_actions=NUM_ACTIONS,
                                              num_obs=NUM_OBSERVATIONS,
                                              batch_size=args.batch_size,
                                              cv=lock)

    learner = Learner(actor=actor,
                      critic=critic,
                      replay_buffer=shared_replay_buffer,
                      device=device,
                      num_actions=NUM_ACTIONS,
                      num_obs=NUM_OBSERVATIONS,
                      argp=args,
                      logger=logger)

    evaluator = Evaluator(actor=actor, argp=args, logger=logger)

    n = 0
    while n < args.num_runs:

        assert actor.is_shared() and critic.is_shared()
        actor = actor.to(CPU)
        critic = critic.to(CPU)

        if args.num_worker == 1:
            t1 = time.time()
            work(replay_buffer=shared_replay_buffer, actor=actor, parser_args=args)
            t2 = time.time()

        elif args.num_worker > 1:
            processes = [mp.Process(target=work, args=(shared_replay_buffer, actor, args))
                         for _ in range(args.num_worker)]

            t1 = time.time()

            for p in processes:
                p.start()

            for p in processes:
                p.join()

            t2 = time.time()

        else:
            raise ValueError("Error, the number of workers has to be positive.")

        print("Sampling Nr. ", n + 1, "finished after ", t2 - t1)

        actor = actor.to(CUDA) if torch.cuda.is_available() else actor.to(CPU)
        critic = critic.to(CUDA) if torch.cuda.is_available() else critic.to(CPU)

        learner.learn()

        print("Learning Nr. ", n + 1, "finished after ", time.time() - t2)

        evaluator.eval()

        n += 1

        # Saving the model parameters
        if n % SAVE_MODEL_EVERY == 0:
            torch.save(actor.state_dict(), model_dir + "/actor_" + str(n))
            torch.save(critic.state_dict(), model_dir + "/critic_" + str(n))
