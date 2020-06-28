from continous_action_RL.actor_critic_networks import Actor, Critic
from continous_action_RL.evaluater import Evaluator
import gym
import torch
from pathlib import Path

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
    NUM_EVAL_TRAJECTORIES = 50
    NUM_TRAJECTORIES = 32
    BATCH_SIZE = 32
    UPDATE_TARGNETS_EVERY = 50
    NUM_TRAINING_ITERATIONS = 100
    TOTAL_TIMESTEPS = 1000
    ACTOR_LEARNING_RATE = 2e-4
    CRITIC_LEARNING_RATE = 2e-4
    ENTROPY_REGULARIZATION_ON = False
    ENTROPY_REGULARIZATION = 1e-5
    ACTION_STD_LOW = 1e-2
    ACTION_STD_HIGH = 1
    ACTION_MEAN_SCALE = 2
    ACTION_BOUNDS = (-2, 2)
    REPLAY_BUFFER_SIZE = 5000
    LOG_EVERY = 10
    SAVE_MODEL_EVERY = 10

    actor = Actor(num_actions=NUM_ACTIONS,
                  num_obs=NUM_OBSERVATIONS,
                  mean_scale=ACTION_MEAN_SCALE,
                  std_low=ACTION_STD_LOW,
                  std_high=ACTION_STD_HIGH,
                  action_bound=ACTION_BOUNDS)

    critic = Critic(num_actions=NUM_ACTIONS, num_obs=NUM_OBSERVATIONS)

    actor.load_state_dict(torch.load(str(Path(__file__).resolve().parents[1]) + "/continous_action_RL/models/actor_10"))

    evaluator = Evaluator(env=env,
                          actor=actor,
                          critic=critic,
                          save_path=None,
                          num_trajectories=NUM_EVAL_TRAJECTORIES,
                          save_model_every=SAVE_MODEL_EVERY,
                          logger=None,
                          render=True)

    for t in range(TOTAL_TIMESTEPS):
        evaluator.evaluate()
