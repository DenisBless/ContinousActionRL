from continous_action_RL.actor_critic_networks import Actor, Critic
from continous_action_RL.replay_buffer import *
from continous_action_RL.actor import Actor as Sampler
from continous_action_RL.on_policy_learner import OnPolicyLearner
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
    num_training_iter = 2000
    num_trajectories = 32
    trajectory_length = 200  # environment dependent

    # Fill the replay buffer
    replay_buffer = ReplayBuffer(32)
    actor = Actor(num_actions=num_actions,
                  num_obs=num_obs,
                  mean_scale=2,
                  std_low=0.01,
                  std_high=1,
                  action_bound=(-2, 2))

    critic = Critic(num_actions=num_actions, num_obs=num_obs)

    sampler = Sampler(env=env,
                      num_trajectories=num_trajectories,
                      actor_network=actor,
                      replay_buffer=replay_buffer,
                      render=False)

    learner = OnPolicyLearner(actor=actor,
                              critic=critic,
                              actor_lr=1e-3,
                              critic_lr=1e-3,
                              trajectory_length=trajectory_length)

    for _ in range(num_training_iter):
        replay_buffer.clear()
        trajectories = sampler.collect_trajectories()
        learner.learn(replay_buffer)

# Todo: When using an environment with multiple continuous actions, we need to use a Multivariate Normal Dist.
