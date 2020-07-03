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

    TRAJECTORY_LENGTH = 300
    env = ReachEnv(max_steps=TRAJECTORY_LENGTH,
                   control='mocap',
                   coordinates='relative',
                   step_limitation='norm',
                   vector_norm=0.02,
                   dt=1e-2,
                   control_timesteps=3,
                   randomize_objects=False,
                   render=True)

    # PARAMETER
    NUM_OBSERVATIONS = env.get_observation_dimension()
    NUM_ACTIONS = env.agent.get_action_dimension()
    NUM_EVAL_TRAJECTORIES = 10
    NUM_TRAJECTORIES = 200
    BATCH_SIZE = 64
    # NUM_TRAJECTORIES = 300
    # BATCH_SIZE = 128
    UPDATE_TARGNETS_EVERY = 10
    NUM_TRAINING_ITERATIONS = 40
    TOTAL_TIMESTEPS = 1000
    ACTOR_LEARNING_RATE = 2e-4
    CRITIC_LEARNING_RATE = 2e-4
    GRADIENT_CLIPPING_VALUE = None
    NUM_EXPECTATION_SAMPLES = 1
    ENTROPY_REGULARIZATION_ON = False
    ENTROPY_REGULARIZATION = 1e-5
    ACTION_STD_LOW = 1e-1
    ACTION_STD_HIGH = 1
    ACTION_MEAN_SCALE = 1
    ACTION_BOUNDS = (-1, 1)
    REPLAY_BUFFER_SIZE = 10000
    LOG_EVERY = 10
    SAVE_MODEL_EVERY = 10
    MODEL_SAVE_PATH = str(pathlib.Path(__file__).resolve().parents[1]) + "/models/"

    actor = Actor(num_actions=NUM_ACTIONS,
                  num_obs=NUM_OBSERVATIONS,
                  mean_scale=ACTION_MEAN_SCALE,
                  std_low=ACTION_STD_LOW,
                  std_high=ACTION_STD_HIGH,
                  action_bound=ACTION_BOUNDS)

    critic = Critic(num_actions=NUM_ACTIONS, num_obs=NUM_OBSERVATIONS)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    actor.load_state_dict(torch.load(MODEL_SAVE_PATH + "actor_200"))

    actor = actor.to(device)
    critic = critic.to(device)

    evaluator = Evaluator(env=env,
                          actor=actor,
                          critic=critic,
                          save_path=MODEL_SAVE_PATH,
                          num_trajectories=NUM_EVAL_TRAJECTORIES,
                          save_model_every=SAVE_MODEL_EVERY,
                          render=False)

    print("Cuda used:", torch.cuda.is_available())
    for t in range(TOTAL_TIMESTEPS):
        evaluator.evaluate()
