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
                 var_low=0.01,
                 var_high=1,
                 action_bound=None):
        super(Actor, self).__init__()
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.mean_scale = mean_scale
        self.var_low = var_low
        self.var_high = var_high
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
        x = self.normal_dist(x)
        return x

    def normal_dist(self, x, ):
        # mean is between [-mean_scale, mean_scale]
        mean = self.mean_scale * x[0] if x.dim() == 1 else x[:, :, 0]

        # variance is between [var_low, var_high]
        var_unscaled = x[1] if x.dim() == 1 else x[:, :, 1]
        var = (0.5 * (self.var_high - self.var_low)) * var_unscaled + 0.5 * (self.var_high + self.var_low)

        return Normal(loc=mean, scale=torch.sqrt(var))

    def sample(self, state):
        policy = self.forward(state)
        return policy.sample() if self.action_bound is None else policy.sample().clamp(min=self.action_bound[0],
                                                                                       max=self.action_bound[1])

    def action_prob(self, action, observation):
        policy = self.forward(observation)
        return torch.exp(policy.log_prob(action.squeeze(-1)))
