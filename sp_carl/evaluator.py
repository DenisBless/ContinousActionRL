import torch
import gym
from common.replay_buffer import SharedReplayBuffer


class Evaluator:
    def __init__(self,
                 actor: torch.nn.Module,
                 argp,
                 logger=None,
                 render: bool = False,
                 env=None):

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.actor = actor
        self.num_samples = argp.num_evals
        self.render = render

        self.logger = logger
        self.env = env

        # self.env = gym.make("Swimmer-v2")
        # self.env = gym.make("Pendulum-v0")
        # self.env = gym.make("HalfCheetah-v2")

    def eval(self, n_iter):
        r = []

        for i in range(self.num_samples):
            rewards = []

            obs = torch.tensor(self.env.reset(), dtype=torch.float).to(self.device)
            done = False
            while not done:
                mean, _ = self.actor.forward(obs)
                action = mean.to(self.device)
                # action, _ = self.actor.action_sample(mean, torch.ones_like(mean))
                next_obs, reward, done, _ = self.env.step(action.detach().cpu().numpy().clip(min=self.env.action_space.low, max=self.env.action_space.high))
                next_obs = torch.tensor(next_obs, dtype=torch.float).to(self.device)
                rewards.append(reward)

                obs = next_obs
                if self.render:
                    self.env.render()

            disc_ret = 0
            gamma = 0.99
            for t, rr in enumerate(rewards):
                disc_ret += gamma ** t * rr

            r.append(disc_ret)

        if self.logger is not None:
            self.logger.add_scalar(scalar_value=sum(r) / len(r), tag="mean discounted return", global_step=n_iter)
