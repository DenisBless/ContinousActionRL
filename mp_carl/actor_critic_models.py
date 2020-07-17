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
        """
        Copy the parameters from the source network to the current network.

        Args:
            source_network: Network to copy parameters from

        Returns:
            No return value
        """
        for param, source_param in zip(self.parameters(), source_network.parameters()):
            param.data.copy_(source_param.data)

    def freeze_net(self) -> None:
        """
        Deactivate gradient
            Computation for the network

        Returns:
            No return value
        """
        for params in self.parameters():
            params.requires_grad = False

    @staticmethod
    def init_weights(module: torch.nn.Module) -> None:
        """
        Orthogonal initialization of the weights. Sets initial bias to zero.

        Args:
            module: Network to initialize weights.

        Returns:
            No return value

        """
        if type(module) == torch.nn.Linear:
            torch.nn.init.orthogonal_(module.weight)
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
        """
        Creates an action sample from the policy network. The output of the network is assumed to be gaussian
        distributed. Let u be a random variable with distribution p(u|s). Since we want our actions to be bound in
        [-1, 1] we apply the tanh function to u, that is a = tanh(u). By change of variable, we have:

        π(a|s) = p(u|s) |det(da/du)| ^-1. Since da/du = diag(1 - tanh^2(u)). We obtain the log likelihood as
        log π(a|s) = log p(u|s) - ∑_i 1 - tanh^2(u_i)

        Args:
            mean: μ(s)
            log_std: log σ(s) where u ~ N(•|μ(s), σ(s))

        Returns:
            action sample a = tanh(u) and log prob log π(a|s)
        """
        dist = self.normal_dist(mean, log_std)
        normal_action = dist.rsample()  # rsample() employs reparameterization trick
        action = torch.tanh(normal_action)
        normal_log_prob = dist.log_prob(normal_action)
        log_prob = torch.sum(normal_log_prob, dim=-1) - torch.sum(torch.log((1 - action.pow(2) + self.eps)), dim=-1)
        # return 2 * action, log_prob  # todo change to normalization
        return action, log_prob

    def get_log_prob(self, actions: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor,
                     normal_actions: torch.Tensor = None) -> torch.Tensor:
        """
        Returns the log prob of a given action a = tanh(u) and u ~ N(•|μ(s), σ(s)) according to

        log π(a|s) = log p(u|s) - ∑_i 1 - tanh^2(u_i).

        If u is not given we can reconstruct it with u = tanh^-1(a), since tanh is bijective.

        Args:
            actions: a = tanh(u)
            mean: μ(s)
            log_std: log σ(s)
            normal_actions: u ~ N(•|μ(s), σ(s))

        Returns:
            log π(a|s)
        """
        # actions = actions / 2  # todo use normalize instead
        if normal_actions is None:
            normal_actions = self.inverseTanh(actions)

        normal_log_probs = self.normal_dist(mean, log_std).log_prob(normal_actions)
        log_probs = torch.sum(normal_log_probs, dim=-1) - torch.sum(torch.log(1 - actions.pow(2) + self.eps), dim=-1)
        assert not torch.isnan(log_probs).any()
        return log_probs

    @staticmethod
    def normal_dist(mean: torch.Tensor, log_std: torch.Tensor) -> Normal:
        """
        Returns a normal distribution.

        Args:
            mean: μ(s)
            log_std: log σ(s) where u ~ N(•|μ(s), σ(s))

        Returns:
            N(u|μ(s), σ(s))
        """
        return Normal(loc=mean, scale=torch.ones_like(mean) * log_std.exp())

    def inverseTanh(self, action: torch.Tensor) -> torch.Tensor:
        """
        Computes the inverse of the tanh for the given action
        Args:
            action: a = tanh(u)

        Returns:
            u = tanh^-1(a)
        """
        eps = torch.finfo(action.dtype).eps  # The smallest representable number such that 1.0 + eps != 1.0
        atanh = self.atanh(action.clamp(min=-1. + eps, max=1. - eps))
        assert not torch.isnan(atanh).any()
        return atanh

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
