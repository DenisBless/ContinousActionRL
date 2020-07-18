import torch
import gym
from common.replay_buffer import SharedReplayBuffer


class Evaluator:
    def __init__(self,
                 actor: torch.nn.Module,
                 logger,
                 argp):

        self.actor = actor
        self.num_samples = argp.num_evals

        self.logger = logger

        self.env = gym.make("Swimmer-v2")
        # self.env = gym.make("Pendulum-v0")
        # self.env = gym.make("HalfCheetah-v2")

    def eval(self):
        for i in range(self.num_samples):
            rewards = []

            obs = torch.tensor(self.env.reset(), dtype=torch.float)
            done = False
            while not done:
                mean, log_std = self.actor.forward(obs)
                action, _ = self.actor.action_sample(mean, log_std)
                next_obs, reward, done, _ = self.env.step(action.detach().cpu())
                next_obs = torch.tensor(next_obs, dtype=torch.float)
                rewards.append(reward)
                obs = next_obs

            self.logger.add_scalar(scalar_value=sum(rewards) / len(rewards), tag="mean reward")
