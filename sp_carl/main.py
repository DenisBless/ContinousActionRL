import os
import numpy as np
from gym.spaces import Box
import torch
import time

import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from common.actor_critic_models import Actor, Critic
from common.replay_buffer import SharedReplayBuffer
from common.arg_parser import ArgParser
from sp_carl.learner import Learner
from sp_carl.sampler import Sampler
from sp_carl.evaluator import Evaluator

CPU = 'cpu'
CUDA = 'cuda:0'

parser = ArgParser()

# Swimmer
NUM_ACTIONS = 2
NUM_OBSERVATIONS = 8
ACTION_SPACE = Box(low=np.array([-1., -1.]),
                   high=np.array([1., 1.]))
OBS_SPACE = None  # observation space is unbounded
EPISODE_LENGTH = 1000


def work(replay_buffer, actor, parser_args):
    worker = Sampler(replay_buffer=replay_buffer,
                     actor=actor,
                     argp=parser_args)
    worker.run()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ""  # Disable CUDA

    device = CUDA if torch.cuda.is_available() else CPU

    lock = mp.Lock()
    logger = SummaryWriter()
    args = parser.parse_args()

    actor = Actor(num_actions=NUM_ACTIONS, num_obs=NUM_OBSERVATIONS, log_std_init=np.log(args.init_std))

    actor.share_memory()

    critic = Critic(num_actions=NUM_ACTIONS, num_obs=NUM_OBSERVATIONS)

    critic.share_memory()

    shared_replay_buffer = SharedReplayBuffer(capacity=args.replay_buffer_size,
                                              trajectory_length=EPISODE_LENGTH,
                                              num_actions=NUM_ACTIONS,
                                              num_obs=NUM_OBSERVATIONS,
                                              batch_size=args.batch_size,
                                              lock=lock)

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
