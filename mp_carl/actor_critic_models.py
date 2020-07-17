import torch
from gym.spaces import Box
from typing import Tuple, List
from torch.distributions.normal import Normal


def normalize_actions(action: torch.Tensor, action_bounds: Box) -> torch.Tensor:
    ...


def denormalize_actions(action: torch.Tensor, lower_bounds: List, upper_bounds: List) -> torch.Tensor:
    ...


def normalize_obs(observations: torch.Tensor, lower_bounds: List, upper_bounds: List) -> torch.Tensor:
    ...


class Base(torch.nn.Module):
    def __init__(self,
                 num_actions: int,
                 num_obs: int):
        super(Base, self).__init__()
        self.num_actions = num_actions
        self.num_obs = num_obs

    def copy_params(self, source_network: torch.nn.Module) -> None:
        for param, source_param in zip(self.parameters(), source_network.parameters()):
            param.data.copy_(source_param.data)

    def freeze_net(self) -> None:
        for params in self.parameters():
            params.requires_grad = False

    @staticmethod
    def init_weights(module: torch.nn.Module, gain: float = 1) -> None:
        if type(module) == torch.nn.Linear:
            torch.nn.init.orthogonal_(module.weight, gain=gain)
            module.bias.data.fill_(0.0)


class Actor(Base):
    def __init__(self,
                 num_actions: int,
                 num_obs: int,
                 actor_layers: List = None,
                 log_std_init: float = -2.,
                 eps: float = 1e-6,
                 logger=None):

        super(Actor, self).__init__(num_actions, num_obs)
        self.log_std_init = log_std_init
        self.eps = eps
        self.action_dim = num_actions
        self.logger = logger

        if actor_layers is None:
            actor_layers = [64, 64]

        actor_base = [num_obs] + actor_layers + [num_actions]
        actor_modules = []
        for i in range(len(actor_base) - 1):
            actor_modules.append(torch.nn.Linear(actor_base[i], actor_base[i + 1]))
            if i is not len(actor_base) - 2:
                actor_modules.append(torch.nn.ReLU())

        self.model = torch.nn.Sequential(*actor_modules)
        self.init_weights(self.model)

        self.log_std = torch.nn.Parameter(torch.ones(self.action_dim) * self.log_std_init, requires_grad=True)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.model(obs)
        return mean, self.log_std.clamp(min=-3)  # Lower bound on the variance

    def action_sample(self, mean: torch.Tensor, log_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.normal_dist(mean, log_std)
        normal_action = dist.rsample()  # rsample() employs reparameterization trick
        action = torch.tanh(normal_action)
        normal_log_prob = dist.log_prob(normal_action)
        log_prob = torch.sum(normal_log_prob, dim=-1) - torch.sum(torch.log((1 - action.pow(2) + self.eps)), dim=-1)
        # return 2 * action, log_prob  # todo change to normalization
        return action, log_prob

    def get_log_prob(self, actions: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor,
                     normal_actions: torch.Tensor = None) -> torch.Tensor:
        # actions = actions / 2  # todo use normalize instead
        if normal_actions is None:
            normal_actions = self.inverseTanh(actions)

        normal_log_probs = self.normal_dist(mean, log_std).log_prob(normal_actions)
        log_probs = torch.sum(normal_log_probs, dim=-1) - torch.sum(torch.log(1 - actions.pow(2) + self.eps), dim=-1)
        return log_probs

    @staticmethod
    def normal_dist(mean: torch.Tensor, log_std: torch.Tensor) -> Normal:
        return Normal(loc=mean, scale=torch.ones_like(mean) * log_std.exp())

    def inverseTanh(self, action: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(action.dtype).eps  # The smallest representable number such that 1.0 + eps != 1.0

        return self.atanh(action.clamp(min=-1. + eps, max=1. - eps))

    @staticmethod
    def atanh(action: torch.Tensor) -> torch.Tensor:
        return 0.5 * (action.log1p() - (-action).log1p())


class Critic(Base):
    def __init__(self,
                 num_actions: int,
                 num_obs: int,
                 critic_layers: List = None):

        super(Critic, self).__init__(num_actions, num_obs)
        if critic_layers is None:
            critic_layers = [64, 64]

        critic_base = [num_actions + num_obs] + critic_layers + [1]

        critic_modules = []
        for i in range(len(critic_base) - 1):
            critic_modules.append(torch.nn.Linear(critic_base[i], critic_base[i + 1]))
            if i is not len(critic_base) - 2:
                critic_modules.append(torch.nn.ReLU())

        self.model = torch.nn.Sequential(*critic_modules)
        self.init_weights(self.model)

    def forward(self, action, obs):
        x = torch.cat([action, obs], dim=-1)  # todo remove /2 by normalize_action()
        return self.model(x)
