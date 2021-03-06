import numpy as np
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
from torch.distributions import Normal


class Critic(torch.nn.Module):
    def __init__(self,
                 num_actions,
                 num_obs,
                 hidden_size1=64,
                 hidden_size2=64,
                 layer_norm=False):
        super(Critic, self).__init__()

        self.num_actions = num_actions
        self.layer_norm = layer_norm
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
        x = F.layer_norm(x, normalized_shape=list(x.shape)) if self.layer_norm else x
        x = F.elu(self.hidden1(x))
        x = F.layer_norm(x, normalized_shape=list(x.shape)) if self.layer_norm else x
        x = self.output(x)
        return x

    def copy_params(self, source_network):
        for param, source_param in zip(self.parameters(), source_network.parameters()):
            param.data.copy_(source_param.data)

    def copy_gradients(self, source_network):
        for param, source_param in zip(self.parameters(), source_network.parameters()):
            param._grad = source_param.grad


class Actor(torch.nn.Module):
    def __init__(self,
                 num_actions,
                 num_obs,
                 hidden_size1=64,
                 hidden_size2=64,
                 out_layer='linear',
                 mean_scale=1,
                 std_low=0.01,
                 std_high=1,
                 action_bound=None,
                 layer_norm=False):

        super(Actor, self).__init__()
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.layer_norm = layer_norm
        self.mean_scale = mean_scale
        self.out_layer = out_layer
        self.std_low = std_low
        self.std_high = std_high
        self.action_bound = action_bound
        self.input = torch.nn.Linear(num_obs, hidden_size1)
        self.hidden1 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.output = torch.nn.Linear(hidden_size2, 2 * num_actions)
        self.hardtanh = torch.nn.Hardtanh(min_val=-1, max_val=1)

    def forward(self, observation):
        x = F.elu(self.input(observation))
        x = F.layer_norm(x, normalized_shape=list(x.shape)) if self.layer_norm else x
        x = F.elu(self.hidden1(x))
        if self.out_layer == 'tanh':
            x = F.layer_norm(x, normalized_shape=list(x.shape)) if self.layer_norm else x
            x = torch.tanh(self.output(x))
            mean, std = self.get_normal_params(x)
        elif self.out_layer == 'linear':
            x = self.output(x)
            mean = x[:self.num_actions] if x.dim() == 1 else x[:, :, :self.num_actions]
            mean = mean.clamp(min=-self.mean_scale, max=self.mean_scale)
            std = x[self.num_actions:] if x.dim() == 1 else x[:, :, self.num_actions:]
            std = std.clamp(min=self.std_low, max=self.std_high)
        else:
            raise ValueError("Error, choose a valid output layer.")
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

        #
        # if self.action_bound:
        #     action_sample = action_sample.clamp(min=self.action_bound[0], max=self.action_bound[1])

        log_probs = self.get_log_prob(action_sample, mean, std)

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
            log N(a|μ(x), σ(x)^2) = - log[(√2π)σ] - 1/2 (x - μ/σ)^2
        """
        assert action_sample.shape == mean.shape == std.shape, \
            "Error, shape mismatch. Shapes: action_sample: " \
            + str(action_sample.shape) + " mean: " + str(mean.shape) + " std: " + str(std.shape)

        t1 = - 0.5 * torch.pow(((mean - action_sample) / std), exponent=2)
        t2 = - torch.log(torch.sqrt(torch.tensor(2 * np.pi, dtype=torch.float)) * std)
        return t1 + t2

    def copy_params(self, source_network):
        for param, source_param in zip(self.parameters(), source_network.parameters()):
            param.data.copy_(source_param.data)

    def copy_gradients(self, source_network):
        for param, source_param in zip(self.parameters(), source_network.parameters()):
            param._grad = source_param.grad

    # def sample(self, mean, std):
    #     dist = torch.distributions.Normal(loc=mean, scale=std)
    #     return dist.rsample()
    #
    # def log_probs(self, dist):
    #     return dist.log_prob()


class ParameterManager:
    def __init__(self, num_actions,
                 num_observations,
                 mean_scale,
                 action_std_low,
                 action_std_high,
                 action_bound):

        self.actor = Actor(num_actions=num_actions,
                           num_obs=num_observations,
                           mean_scale=mean_scale,
                           std_low=action_std_low,
                           std_high=action_std_high,
                           action_bound=(-action_bound, action_bound))

        self.actor.share_memory()

        self.avg_actor = Actor(num_actions=num_actions,
                               num_obs=num_observations,
                               mean_scale=mean_scale,
                               std_low=action_std_low,
                               std_high=action_std_high,
                               action_bound=(-action_bound, action_bound))

        self.avg_actor.share_memory()
        self.avg_actor.copy_params(source_network=self.actor)

        self.critic = Critic(num_actions=num_actions, num_obs=num_observations)

        self.critic.share_memory()

        self.avg_critic = Critic(num_actions=num_actions, num_obs=num_observations)
        self.avg_critic.share_memory()
        self.avg_critic.copy_params(source_network=self.critic)
