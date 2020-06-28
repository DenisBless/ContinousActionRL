from continous_action_RL.actor_critic_networks import Actor, Critic
from continous_action_RL.off_policy.replay_buffer import ReplayBuffer
from continous_action_RL.sampler import Sampler
from continous_action_RL.evaluator import Evaluator
from continous_action_RL.logger import Logger
from continous_action_RL.off_policy.off_policy_learner import OffPolicyLearner
import pathlib
import gym

if __name__ == '__main__':

    """
    Information on the Environment:
    Actions - Dim: (1,); Value Range: [-1, 1]
    Varying Trajectory Length of 1000 
    Goal: Mountain car hits the top
    """
    env = gym.make("MountainCarContinuous-v0")

    # PARAMETER
    NUM_OBSERVATIONS = env.observation_space.shape[0]
    NUM_ACTIONS = env.action_space.shape[0]
    TRAJECTORY_LENGTH = 1000
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
    ACTION_MEAN_SCALE = 1
    ACTION_BOUNDS = (-1, 1)
    REPLAY_BUFFER_SIZE = 5000
    LOG_EVERY = 10
    SAVE_MODEL_EVERY = 10
    MODEL_SAVE_PATH = str(pathlib.Path().absolute()) + "/models/"

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    logger = Logger(log_every=LOG_EVERY)

    actor = Actor(num_actions=NUM_ACTIONS,
                  num_obs=NUM_OBSERVATIONS,
                  mean_scale=ACTION_MEAN_SCALE,
                  std_low=ACTION_STD_LOW,
                  std_high=ACTION_STD_HIGH,
                  action_bound=ACTION_BOUNDS)

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
                               entropy_regularization_on=ENTROPY_REGULARIZATION_ON,
                               entropy_regularization=ENTROPY_REGULARIZATION,
                               update_targnets_every=UPDATE_TARGNETS_EVERY,
                               num_training_iter=NUM_TRAINING_ITERATIONS,
                               minibatch_size=BATCH_SIZE,
                               logger=logger)

    evaluator = Evaluator(env=env,
                          actor=actor,
                          critic=critic,
                          save_path=MODEL_SAVE_PATH,
                          num_trajectories=NUM_EVAL_TRAJECTORIES,
                          save_model_every=SAVE_MODEL_EVERY,
                          logger=logger,
                          render=False)

    for t in range(TOTAL_TIMESTEPS):
        print("-" * 10, t, "-" * 10)
        sampler.collect_trajectories()
        learner.learn(replay_buffer)
        evaluator.evaluate()

