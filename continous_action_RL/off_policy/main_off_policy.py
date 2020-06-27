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
    num_obs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    num_trajectories = 300
    mini_batch_size = 64
    trajectory_length = 200  # environment dependent
    log_dir = "/logs/"

    replay_buffer = ReplayBuffer(5000)

    logger = Logger(log_every=10)

    actor = Actor(num_actions=num_actions,
                  num_obs=num_obs,
                  mean_scale=2,
                  std_low=1e-2,
                  std_high=1,
                  action_bound=(-2, 2),
                  logger=logger)

    critic = Critic(num_actions=num_actions, num_obs=num_obs)

    sampler = Sampler(env=env,
                      num_trajectories=num_trajectories,
                      actor_network=actor,
                      replay_buffer=replay_buffer,
                      render=False,
                      logger=logger)

    learner = OffPolicyLearner(actor=actor,
                               critic=critic,
                               trajectory_length=trajectory_length,
                               actor_lr=2.5e-4,
                               critic_lr=2.5e-4,
                               update_targnets_every=20,
                               minibatch_size=mini_batch_size,
                               logger=logger)

    for t in range(5000):
        print("-" * 10, t, "-" * 10)
        sampler.collect_trajectories()
        learner.learn(replay_buffer)

# Todo: When using an environment with multiple continuous actions, we need to use a Multivariate Normal Dist.
# Todo: How do we compute the expectations wrt to the policy?
# Todo: Do we detach the action and action probs when sampling?
# Todo: The delta in the retrace loss in LbP paper is computed between "i" and "j". In the original paper it os
#  computed between "i" and "i + 1". Which one is correct?
