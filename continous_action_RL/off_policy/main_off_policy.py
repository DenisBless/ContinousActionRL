from continous_action_RL.actor_critic_networks import Actor, Critic
from continous_action_RL.off_policy.replay_buffer import *
from continous_action_RL.sampler import Sampler
from continous_action_RL.logger import Logger
from continous_action_RL.off_policy.off_policy_learner import OffPolicyLearner
import gym

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
    NUM_TRAJECTORIES = 300
    BATCH_SIZE = 64
    TRAJECTORY_LENGTH = 200  # environment dependent
    UPDATE_TARGNETS_EVERY = 20
    ACTOR_LEARNING_RATE = 2e-4
    CRITIC_LEARNING_RATE = 2e-4
    ACTION_STD_LOW = 1E-2
    ACTION_STD_HIGH = 1
    ACTION_MEAN_SCALE = 2
    ACTION_BOUNDS = (-2, 2)
    REPLAY_BUFFER_SIZE = 5000
    LOG_EVERY = 10

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    logger = Logger(log_every=LOG_EVERY)

    actor = Actor(num_actions=NUM_ACTIONS,
                  num_obs=NUM_OBSERVATIONS,
                  mean_scale=ACTION_MEAN_SCALE,
                  std_low=ACTION_STD_LOW,
                  std_high=ACTION_STD_HIGH,
                  action_bound=ACTION_BOUNDS,
                  logger=logger)

    critic = Critic(num_actions=NUM_ACTIONS, num_obs=NUM_OBSERVATIONS)

    sampler = Sampler(env=env,
                      num_trajectories=NUM_TRAJECTORIES,
                      actor_network=actor,
                      replay_buffer=replay_buffer,
                      render=False,
                      logger=logger)

    learner = OffPolicyLearner(actor=actor,
                               critic=critic,
                               trajectory_length=TRAJECTORY_LENGTH,
                               actor_lr=ACTOR_LEARNING_RATE,
                               critic_lr=CRITIC_LEARNING_RATE,
                               update_targnets_every=UPDATE_TARGNETS_EVERY,
                               minibatch_size=BATCH_SIZE,
                               logger=logger)

    for t in range(5000):
        print("-" * 10, t, "-" * 10)
        sampler.collect_trajectories()
        learner.learn(replay_buffer)
