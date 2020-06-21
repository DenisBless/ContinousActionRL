import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


class Critic(torch.nn.Module):
    def __init__(self,
                 num_actions,
                 num_obs,
                 hidden_size1=64,
                 hidden_size2=64):
        super(Critic, self).__init__()

        self.num_actions = num_actions
        self.input = torch.nn.Linear(num_actions + num_obs, hidden_size1)
        self.hidden = torch.nn.Linear(hidden_size1, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, 1)

    def forward(self, action, observation):
        x = self.input(torch.cat((action, observation), dim=2))  # dim 2 are the input features
        x = F.elu(x)
        x = self.hidden(x)
        x = F.elu(x)
        x = self.output(x)
        return x


class Actor(torch.nn.Module):
    def __init__(self,
                 num_actions,
                 num_obs,
                 hidden_size1=64,
                 hidden_size2=64,
                 mean_scale=1,
                 std_low=0.01,
                 std_high=1,
                 action_bound=None):
        super(Actor, self).__init__()
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.mean_scale = mean_scale
        self.std_low = std_low
        self.std_high = std_high
        self.action_bound = action_bound
        self.input = torch.nn.Linear(num_obs, hidden_size1)
        self.hidden = torch.nn.Linear(hidden_size1, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, 2 * num_actions)

    def forward(self, observation):
        x = self.input(observation)
        x = F.elu(x)
        x = self.hidden(x)
        x = F.elu(x)
        x = self.output(x)
        x = F.tanh(x)
        mean, std = self.get_normal_params(x)
        action_sample = self.action_sample(mean, std).detach()
        action_log_prob = self.get_log_prob(action_sample, mean, std)
        return action_sample, action_log_prob

    def action_sample(self, mean, std):
        eps = Normal(loc=torch.zeros_like(mean), scale=torch.ones_like(std)).sample()
        sample = std * eps + mean
        return sample if self.action_bound is None else sample.clamp(min=self.action_bound[0], max=self.action_bound[1])

    def get_normal_params(self, x):
        # mean is between [-mean_scale, mean_scale]
        mean = self.mean_scale * x[0] if x.dim() == 1 else x[:, :, 0]

        # standard deviation is between [std_low, std_high]
        std_unscaled = x[1] if x.dim() == 1 else x[:, :, 1]
        std = (0.5 * (self.std_high - self.std_low)) * std_unscaled + 0.5 * (self.std_high + self.std_low)
        return mean, std

    def get_log_prob(self, action_sample, mean, std):
        t1 = - ((mean - action_sample)**2) / (2 * std**2)
        t2 = - torch.sqrt(torch.tensor(2 * np.pi, dtype=torch.float) * std)
        return t1 + t2

