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
        self.hidden1 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, 1)

    def forward(self, action, observation):
        """
        Critic network. Approx. the discounted sum of rewards for the current state s_t when taking action a_t.

        Args:
            action: a_t
            observation: s_t

        Returns:
            Q(s_t, a_t)
        """
        assert action.dim() == observation.dim(), \
            "Error, dimension mismatch. Dimensions: " \
            "action: " + str(action.dim()) + " observation: " + str(observation.dim())

        x = F.elu(self.input(torch.cat((action, observation), dim=2)))  # dim 2 are the input features
        x = F.elu(self.hidden1(x))
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
        self.hidden1 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, 2 * num_actions)

    def forward(self, observation):
        x = F.elu(self.input(observation))
        x = F.elu(self.hidden1(x))
        x = torch.tanh(self.output(x))
        mean, std = self.get_normal_params(x)
        return mean, std

    def action_sample(self, mean, std):
        """
        Computes 𝔼_π[log N(a|μ(x), σ(x)^2)], 𝔼_π[log N(a|μ(x), σ(x)^2)]
        Args:
            mean: μ(x)
            std: σ(x)

        Returns:
            a ~ π(•|s), log N(a|μ(x), σ(x)^2)
        """
        if self.training:
            eps = Normal(loc=torch.zeros_like(mean), scale=torch.ones_like(std)).sample()
        else:  # setting the variance to zero when evaluating the model
            eps = Normal(loc=torch.zeros_like(mean), scale=torch.zeros_like(std)).sample()

        action_sample = std * eps + mean
        log_probs = self.get_log_prob(action_sample, mean, std)

        if self.action_bound:
            return action_sample.clamp(min=self.action_bound[0], max=self.action_bound[1]), log_probs
        else:
            return action_sample, log_probs

    def get_normal_params(self, x):
        """
        Computes mean μ(x) and std σ(x) where x is the output of the neural network.
        Args:
            x: output of the neural network

        Returns:
            μ(x), σ(x)
        """
        # mean is between [-mean_scale, mean_scale]
        mid = x.shape[-1] // 2
        mean = self.mean_scale * x[:mid] if x.dim() == 1 else self.mean_scale * x[:, :, :mid]

        # standard deviation is between [std_low, std_high]
        std_unscaled = x[mid:] if x.dim() == 1 else x[:, :, mid:]
        std = (0.5 * (self.std_high - self.std_low)) * std_unscaled + 0.5 * (self.std_high + self.std_low)

        return mean, std

    @staticmethod
    def get_log_prob(action_sample, mean, std):
        """
        Computes log N(a|μ(x), σ(x)^2) where a ~ π(•|s)
        Args:
            action_sample: a ~ π(•|s)
            mean: μ(x)
            std: σ(x)

        Returns:
            log N(a|μ(x), σ(x)^2)
        """
        assert action_sample.shape == mean.shape == std.shape, \
            "Error, shape mismatch. Shapes: action_sample: " \
            + str(action_sample.shape) + " mean: " + str(mean.shape) + " std: " + str(std.shape)

        t1 = - 0.5 * (((mean - action_sample) / std) ** 2)
        t2 = - torch.sqrt(torch.tensor(2 * np.pi, dtype=torch.float)) * std
        return t1 + t2
