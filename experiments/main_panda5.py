from continous_action_RL.actor_critic_networks import Actor, Critic
from continous_action_RL.off_policy.replay_buffer import ReplayBuffer
from continous_action_RL.sampler import Sampler
from continous_action_RL.evaluator import Evaluator
from continous_action_RL.logger import Logger
from continous_action_RL.off_policy.off_policy_learner import OffPolicyLearner
from ALR_SF.SimulationFramework.simulation.src.gym_sf.mujoco.mujoco_envs.reach_env.reach_env import ReachEnv
import pathlib
import torch
import time

if __name__ == '__main__':

    TRAJECTORY_LENGTH = 100
    env = ReachEnv(max_steps=TRAJECTORY_LENGTH,
                   control='mocap',
                   coordinates='relative',
                   action_smoothing=True,
                   step_limitation='percentage',
                   percentage=0.02,
                   dt=1e-3,
                   control_timesteps=100,
                   randomize_objects=False,
                   render=False)

    # PARAMETER
    NUM_OBSERVATIONS = env.observation_dim
    NUM_ACTIONS = env.agent.action_dimension
    NUM_EVAL_TRAJECTORIES = 1
    NUM_TRAJECTORIES = 150
    BATCH_SIZE = 32
    NUM_BUFFER_FRONTLOAD = 0
    # NUM_TRAJECTORIES = 16
    # BATCH_SIZE = 16
    # NUM_BUFFER_FRONTLOAD = 0
    UPDATE_TARGNETS_EVERY = 50
    NUM_TRAINING_ITERATIONS = 2000
    TOTAL_TIMESTEPS = 1000
    ACTOR_LEARNING_RATE = 1e-4
    CRITIC_LEARNING_RATE = 1e-4
    GRADIENT_CLIPPING_VALUE = 0.5
    NUM_EXPECTATION_SAMPLES = 1
    ENTROPY_REGULARIZATION_ON = False
    ENTROPY_REGULARIZATION = 0
    TRUST_REGION_COEFF = 0
    ACTION_STD_LOW = 5e-1
    ACTION_STD_HIGH = 1
    ACTION_MEAN_SCALE = 1
    ACTION_BOUNDS = (-1, 1)
    REPLAY_BUFFER_SIZE = 5000
    LOG_EVERY = 10
    SAVE_MODEL_EVERY = 10
    MODEL_SAVE_PATH = str(pathlib.Path(__file__).resolve().parents[1]) + "/models/"

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    logger = Logger(log_every=LOG_EVERY)

    actor = Actor(num_actions=NUM_ACTIONS,
                  num_obs=NUM_OBSERVATIONS,
                  mean_scale=ACTION_MEAN_SCALE,
                  std_low=ACTION_STD_LOW,
                  std_high=ACTION_STD_HIGH,
                  action_bound=ACTION_BOUNDS)

    critic = Critic(num_actions=NUM_ACTIONS, num_obs=NUM_OBSERVATIONS)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    actor = actor.to(device)
    critic = critic.to(device)

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
                               expectation_samples=NUM_EXPECTATION_SAMPLES,
                               entropy_regularization_on=ENTROPY_REGULARIZATION_ON,
                               entropy_regularization=ENTROPY_REGULARIZATION,
                               trust_region_coeff=TRUST_REGION_COEFF,
                               gradient_clip_val=GRADIENT_CLIPPING_VALUE,
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

    print("Cuda used:", torch.cuda.is_available())
    for i in range(NUM_BUFFER_FRONTLOAD):
        # Filling the buffer
        sampler.collect_trajectories()

    for t in range(TOTAL_TIMESTEPS):
        tm = time.time()
        print("-" * 10, t, "-" * 10)
        sampler.collect_trajectories()
        print("Sampling Nr. ", t + 1, " finished in ", time.time() - tm, " seconds.")
        tm = time.time()
        learner.learn(replay_buffer)
        print("Learning Nr. ", t + 1, " finished in ", time.time() - tm, " seconds.")
        evaluator.evaluate()
