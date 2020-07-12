import os
import os.path as osp
import sys
import gym
import pathlib
import torch
import time
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), '..')))

from continous_action_RL.actor_critic_networks import Actor, Critic
from continous_action_RL.networks_continuous import ContinuousCritic, ContinuousActor
from continous_action_RL.off_policy.replay_buffer import ReplayBuffer
from collections import deque
from continous_action_RL.sampler import Sampler
from continous_action_RL.evaluator import Evaluator
from continous_action_RL.logger import Logger
from continous_action_RL.off_policy.off_policy_learner import OffPolicyLearner


def check_gpu(gpu_device):
    if type(gpu_device) is not str:
        gpu_device = str(gpu_device)
    # Make sure we can use gpu
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_device
    use_gpu = torch.cuda.is_available() and use_gpu
    print('Use GPU: %s' % gpu_device if use_gpu else use_gpu)


    return use_gpu

if __name__ == '__main__':

    """
    Information on the Environment:
    Actions - Dim: (1,); Value Range: [-2, 2]
    Observations - Dim (3,)
    Constant Trajectory Length of 200
    Goal: Pendulum should stay upright
    """
    env = gym.make("Pendulum-v0")

    # PARAMETER
    GPU_DEVICE = 6

    NUM_OBSERVATIONS = env.observation_space.shape[0]
    NUM_ACTIONS = env.action_space.shape[0]
    TRAJECTORY_LENGTH = 200
    NUM_EVAL_TRAJECTORIES = 5
    NUM_TRAJECTORIES = 32
    BATCH_SIZE = 32
    UPDATE_TARGNETS_EVERY = 16
    NUM_TRAINING_ITERATIONS = 32
    TOTAL_TIMESTEPS = 100
    ACTOR_LEARNING_RATE = 2e-4
    CRITIC_LEARNING_RATE = 2e-4
    GRADIENT_CLIPPING_VALUE = None
    NUM_EXPECTATION_SAMPLES = 1
    ENTROPY_REGULARIZATION_ON = False
    ENTROPY_REGULARIZATION = 1e-5
    ACTION_STD_LOW = 1e-1
    ACTION_STD_HIGH = 1
    ACTION_MEAN_SCALE = 2
    ACTION_BOUNDS = (-2, 2)
    REPLAY_BUFFER_SIZE = 512
    LOG_EVERY = 10
    SAVE_MODEL_EVERY = 5
    MODEL_SAVE_PATH = "./models/"

    use_gpu = check_gpu(GPU_DEVICE)

    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    # logger = Logger(log_every=LOG_EVERY)
    logger = None

    actor = ContinuousActor(use_gpu=use_gpu)

    critic = ContinuousCritic(use_gpu=use_gpu)

    actor = actor.cuda() if use_gpu else actor
    critic = critic.cuda() if use_gpu else critic

    sampler = Sampler(actor, env, None, replay_buffer,
                      num_trajectories=NUM_TRAJECTORIES,
                      continuous=True,
                      writer=logger,
                      use_gpu=use_gpu)

    learner = OffPolicyLearner(actor=actor,
                               critic=critic,
                               trajectory_length=TRAJECTORY_LENGTH,
                               actor_lr=ACTOR_LEARNING_RATE,
                               critic_lr=CRITIC_LEARNING_RATE,
                               expectation_samples=NUM_EXPECTATION_SAMPLES,
                               entropy_regularization_on=ENTROPY_REGULARIZATION_ON,
                               entropy_regularization=ENTROPY_REGULARIZATION,
                               gradient_clip_val=GRADIENT_CLIPPING_VALUE,
                               update_targnets_every=UPDATE_TARGNETS_EVERY,
                               num_training_iter=NUM_TRAINING_ITERATIONS,
                               minibatch_size=BATCH_SIZE,
                               logger=logger,
                               use_gpu=use_gpu)

    evaluator = Evaluator(env=env,
                          actor=actor,
                          critic=critic,
                          save_path=MODEL_SAVE_PATH,
                          num_trajectories=NUM_EVAL_TRAJECTORIES,
                          save_model_every=SAVE_MODEL_EVERY,
                          logger=logger,
                          render=False,
                          use_gpu=use_gpu)

    for t in range(TOTAL_TIMESTEPS):
        tm = time.time()
        print("-" * 10, t, "-" * 10)
        sampler.sample()
        print("Sampling Nr. ", t + 1, " finished in ", time.time() - tm, " seconds.")
        tm = time.time()
        learner.learn(replay_buffer)
        print("Learning Nr. ", t + 1, " finished in ", time.time() - tm, " seconds.")
        evaluator.evaluate()
