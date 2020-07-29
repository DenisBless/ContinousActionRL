import argparse
import os
import pathlib
import numpy as np
from gym.spaces import Box

import torch.multiprocessing as mp
from mp_carl.agent import Agent
from mp_carl.parameter_server import ParameterServer
from common.replay_buffer import SharedReplayBuffer

parser = argparse.ArgumentParser(description='algorithm arguments')

# Algorithm parameter
# parser.add_argument('--num_worker', type=int, default=os.cpu_count(),
parser.add_argument('--num_workers', type=int, default=2,
                    help='Number of workers training the agent in parallel.')
parser.add_argument('--num_grads', type=int, default=200,
                    help='Number of gradients collected before updating the networks.')
parser.add_argument('--update_targnets_every', type=int, default=10,
                    help='Number of learning steps before the target networks are updated.')
parser.add_argument('--learning_steps', type=int, default=2000,
                    help='Total number of learning timesteps before sampling trajectories.')
parser.add_argument('--num_runs', type=int, default=5000,
                    help='Number of learning iterations.')
parser.add_argument('--actor_lr', type=float, default=2e-4,
                    help='Learning rate for the actor network.')
parser.add_argument('--critic_lr', type=float, default=2e-4,
                    help='Learning rate for the critic network.')
parser.add_argument('--init_std', type=float, default=1,
                    help='Initial standard deviation of the actor.')

parser.add_argument('--global_gradient_norm', type=float, default=0.5,
                    help='Enables gradient clipping with a specified global parameter L2 norm')
parser.add_argument('--num_expectation_samples', type=int, default=1,
                    help='Number of action samples for approximating the value function from the Q function.')
parser.add_argument('--entropy_reg', type=float, default=0,
                    help='Scaling of entropy term in the actor loss function')
parser.add_argument('--trust_region_coeff', type=float, default=0,
                    help='Scaling of the KL-div. between the old action distribution and the current in actor loss.')
parser.add_argument('--action_mean_scale', type=float, default=2,
                    help='Scales the output of the actor net to [-action_mean_scale, action_mean_scale].')
parser.add_argument('--action_bound', type=float, default=2.,
                    help='Clips the action in the range [-action_bound, action_bound].')
parser.add_argument('--replay_buffer_size', type=int, default=300,
                    help='Size of the replay buffer.')
parser.add_argument('--logging', type=bool, default=True,
                    help='Whether to log data or not.')

# Environment parameter
# parser.add_argument('--episode_length', type=int, default=200,
#                     help='Length of a episode.')
parser.add_argument('--num_eval_trajectories', type=int, default=1,
                    help='Number of trajectories used for evaluating the policy.')
parser.add_argument('--num_trajectories', type=int, default=10,
                    help='Number of trajectories sampled before entering the learning phase.')
parser.add_argument('--dt', type=float, default=1e-3,
                    help='Time between steps in the mujoco pyhsics simulation (in seconds).')
parser.add_argument('--percentage', type=float, default=2e-2,
                    help='Percent scaling of the action which is in [-1, 1] (in meters)')
parser.add_argument('--control_timesteps', type=int, default=100,
                    help='Number of steps in the simulation between before the next action is executed.')
parser.add_argument('--action_smoothing', type=bool, default=True,
                    help='Uses number <control_timsteps> to smooth between two action from the agent.')

parser.add_argument('--render', type=bool, default=False,
                    help='If true, the environment is rendered.')

parser.add_argument('--log_interval', type=int, default=10,
                    help='Interval of the logger writing data to the tensorboard.')

# Number of action samples for approximating the value function from the Q function
NUM_EXPECTATION_SAMPLES = 1
MODEL_SAVE_path = str(pathlib.Path(__file__).resolve().parents[1]) + "/models/"
SAVE_MODEL_EVERY = 10
LOG_INTERVAL = 10

# Pendulum
# NUM_ACTIONS = 1
# NUM_OBSERVATIONS = 3
# ACTION_SPACE = Box(low=np.array([-2.]),
#                    high=np.array([2.]))
# OBS_SPACE = Box(low=np.array([-1., -1., -8.]),
#                 high=np.array([1., 1., 8.]))
# EPISODE_LENGTH = 200

# HalfCheetah
# NUM_ACTIONS = 6
# NUM_OBSERVATIONS = 17
# ACTION_SPACE = Box(low=np.array([-1., -1., -1., -1., -1., -1.]),
#                    high=np.array([1., 1., 1., 1., 1., 1.]))
# OBS_SPACE = None  # observation space is unbounded
# EPISODE_LENGTH = 1000

# Swimmer
NUM_ACTIONS = 2
NUM_OBSERVATIONS = 8
ACTION_SPACE = Box(low=np.array([-1., -1.]),
                   high=np.array([1., 1.]))
OBS_SPACE = None  # observation space is unbounded
EPISODE_LENGTH = 1000


def work(param_server, replay_buffer, parser_args, condition):
    worker = Agent(param_server=param_server,
                   shared_replay_buffer=replay_buffer,
                   parser_args=parser_args,
                   condition=condition)
    worker.run()


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = ""  # Disable CUDA

    cv = mp.Condition()
    args = parser.parse_args()
    shared_param_server = ParameterServer(actor_lr=args.actor_lr,
                                          critic_lr=args.critic_lr,
                                          num_actions=NUM_ACTIONS,
                                          num_obs=NUM_OBSERVATIONS,
                                          cv=cv,
                                          arg_parser=args)

    shared_replay_buffer = SharedReplayBuffer(capacity=args.replay_buffer_size,
                                              trajectory_length=EPISODE_LENGTH,
                                              num_actions=NUM_ACTIONS,
                                              num_obs=NUM_OBSERVATIONS,
                                              cv=cv)

    if args.num_workers == 1:
        work(param_server=shared_param_server, replay_buffer=shared_replay_buffer, parser_args=args, condition=cv)

    elif args.num_workers > 1:
        processes = [mp.Process(target=work, args=(shared_param_server, shared_replay_buffer, args, cv))
                     for _ in range(args.num_workers)]
        for p in processes:
            p.start()

        for p in processes:
            p.join()

    else:
        raise ValueError("Error, the number of workers has to be positive.")


# TODO:
"""
We have a bug: Even though we share the gradients of the actor and the critic, when adding gradients to them, 
the threads are adding separate gradients. This can best be observed when setting the number of gradients to a large 
number and uncomment the /G in the parameter server.
"""