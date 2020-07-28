import torch
import gym


class Evaluator:
    def __init__(self,
                 actor: torch.nn.Module,
                 argp,
                 logger=None,
                 render: bool = False):

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.actor = actor
        self.num_samples = argp.num_evals
        self.render = render

        self.logger = logger

        self.env = gym.make("Swimmer-v2")
        # self.env = gym.make("Hopper-v2")
        # self.env = gym.make("Pendulum-v0")
        # self.env = gym.make("HalfCheetah-v2")

    def eval(self):
        r = []
        for i in range(self.num_samples):
            rewards = []

            obs = torch.tensor(self.env.reset(), dtype=torch.float).to(self.device)
            done = False
            while not done:
                mean, _ = self.actor.forward(obs)
                mean = mean.to(self.device)
                action, _ = self.actor.action_sample(mean, torch.ones_like(mean))
                next_obs, reward, done, _ = self.env.step(action.detach().cpu())
                next_obs = torch.tensor(next_obs, dtype=torch.float).to(self.device)
                rewards.append(reward)
                obs = next_obs
                if self.render:
                    self.env.render()
            print(sum(rewards) / len(rewards))
            r.append(sum(rewards) / len(rewards))

        if self.logger is not None:
            self.logger.add_scalar(scalar_value=max(r), tag="mean reward")
            print(max(r))
