from continous_action_RL.actor_critic_networks import Actor, Critic
from continous_action_RL.off_policy.replay_buffer import ReplayBuffer
from continous_action_RL.sampler import Sampler
from continous_action_RL.evaluator import Evaluator
from continous_action_RL.logger import Logger
from continous_action_RL.off_policy.off_policy_learner import OffPolicyLearner
import gym
import pathlib
import torch
import time

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
    NUM_OBSERVATIONS = env.observation_space.shape[0]
    NUM_ACTIONS = env.action_space.shape[0]
    TRAJECTORY_LENGTH = 200

    MODEL_NAME = "actor_100"
    MODEL_LOAD_PATH = "./models/"
    NUM_EVAL_TRAJECTORIES = 10
    ACTION_STD_LOW = 1e-1
    ACTION_STD_HIGH = 1
    ACTION_MEAN_SCALE = 2
    ACTION_BOUNDS = (-2, 2)
    TOTAL_TIMESTEPS = 10

    actor = Actor(num_actions=NUM_ACTIONS,
                  num_obs=NUM_OBSERVATIONS,
                  mean_scale=ACTION_MEAN_SCALE,
                  std_low=ACTION_STD_LOW,
                  std_high=ACTION_STD_HIGH,
                  action_bound=ACTION_BOUNDS)

    critic = Critic(num_actions=NUM_ACTIONS, num_obs=NUM_OBSERVATIONS)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    actor.load_state_dict(torch.load(MODEL_LOAD_PATH + MODEL_NAME))
    actor = actor.to(device)
    critic = critic.to(device)

    evaluator = Evaluator(env=env,
                          actor=actor,
                          critic=critic,
                          num_trajectories=NUM_EVAL_TRAJECTORIES,
                          save_model_every=None,
                          logger=None,
                          render=True)

    for t in range(TOTAL_TIMESTEPS):
        evaluator.evaluate()
