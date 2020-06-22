from continous_action_RL.actor_critic_networks import Actor, Critic
from continous_action_RL.replay_buffer import *
from continous_action_RL.actor import Actor as Sampler
from continous_action_RL.off_policy_learner import OffPolicyLearner
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
    num_trajectories = 150
    trajectory_length = 200  # environment dependent

    # Fill the replay buffer
    replay_buffer = ReplayBuffer(5000)
    actor = Actor(num_actions=num_actions,
                  num_obs=num_obs,
                  mean_scale=1,
                  std_low=0.01,
                  std_high=1,
                  action_bound=(-2, 2))

    critic = Critic(num_actions=num_actions, num_obs=num_obs)

    sampler = Sampler(env=env,
                      num_trajectories=num_trajectories,
                      actor_network=actor,
                      replay_buffer=replay_buffer,
                      render=False)

    learner = OffPolicyLearner(actor=actor,
                               critic=critic,
                               trajectory_length=trajectory_length,
                               actor_lr=2.5e-4,
                               critic_lr=2.5e-4,
                               update_targnets_every=10,
                               minibatch_size=32)

    for _ in range(1000):
        sampler.collect_trajectories()
        learner.learn(replay_buffer)


# Todo: When using an environment with multiple continuous actions, we need to use a Multivariate Normal Dist.
# Todo: How do we compute the expectations wrt to the policy?
# Todo: Do we detach the action and action probs when sampling?
# Todo: The delta in the retrace loss in LbP paper is computed between "i" and "j". In the original paper it os
#  computed between "i" and "i + 1". Which one is correct?
