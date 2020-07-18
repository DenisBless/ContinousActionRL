import argparse
import os
import pathlib
import numpy as np
from gym.spaces import Box

import torch.multiprocessing as mp
from common.actor_critic_models import Actor, Critic
from common.replay_buffer import SharedReplayBuffer
from sp_carl.agent import Agent

CPU = 'cpu'
CUDA = 'cuda:0'

# from mp_carl.agent import Agent
# from mp_carl.parameter_server import ParameterServer
# from mp_carl.shared_replay_buffer import SharedReplayBuffer

parser = argparse.ArgumentParser(description='algorithm arguments')

# Swimmer
NUM_ACTIONS = 2
NUM_OBSERVATIONS = 8
ACTION_SPACE = Box(low=np.array([-1., -1.]),
                   high=np.array([1., 1.]))
OBS_SPACE = None  # observation space is unbounded
EPISODE_LENGTH = 1000


def work(replay_buffer, actor, critic, parser_args):
    worker = Agent(shared_replay_buffer=replay_buffer,
                   actor=actor,
                   critic=critic,
                   parser_args=parser_args)
    worker.run()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ""  # Disable CUDA

    lock = mp.Lock()
    args = parser.parse_args()

    actor = Actor(num_actions=NUM_ACTIONS,
                  num_obs=NUM_OBSERVATIONS,
                  log_std_init=np.log(parser_args.init_std)).to(CPU)

    critic = Critic(num_actions=NUM_ACTIONS,
                    num_obs=NUM_OBSERVATIONS).to(CPU)

    shared_replay_buffer = SharedReplayBuffer(capacity=args.replay_buffer_size,
                                              trajectory_length=EPISODE_LENGTH,
                                              num_actions=NUM_ACTIONS,
                                              num_obs=NUM_OBSERVATIONS,
                                              lock=lock)
    n = 0
    while n < N:

        if args.num_worker == 1:
            work(replay_buffer=shared_replay_buffer, actor=actor, critic=critic, parser_args=args,)

        elif args.num_worker > 1:
            processes = [mp.Process(target=work, args=(shared_replay_buffer, actor, critic, args))
                         for _ in range(args.num_worker)]
            for p in processes:
                p.start()

            for p in processes:
                p.join()

        else:
            raise ValueError("Error, the number of workers has to be positive.")




        n += 1
