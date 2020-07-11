import torch
import torch.nn as nn
import numpy as np
from .networks import SQXNet, TaskHeadActor, TaskHeadCritic


class ContinuousActor(SQXNet):
    """Class for policy (or actor) network"""

    def __init__(self,
                 state_dim=3,
                 base_hidden_size=32,
                 num_intentions=1,
                 head_input_size=16,
                 head_hidden_size=8,
                 head_output_size=4,
                 action_dim=1,
                 non_linear=torch.nn.ELU(),
                 net_type=TaskHeadActor,
                 layer_norm=True,
                 use_gpu=True):
        super(ContinuousActor, self).__init__(state_dim=state_dim,
                                              base_hidden_size=base_hidden_size,
                                              num_intentions=num_intentions,
                                              head_input_size=head_input_size,
                                              head_hidden_size=head_hidden_size,
                                              head_output_size=action_dim * 2,
                                              non_linear=non_linear,
                                              intention_net_type=net_type,
                                              layer_norm=layer_norm,
                                              use_gpu=use_gpu)
        self.logits = nn.Tanh()
        self.action_dim = action_dim

    def predict(self, x, task=None, action=None, sampling_batch=None, requires_grad=False, noise=True):
        x = self(x, task)
        x = self.logits(x)
        assert x.shape[-1] == self.action_dim * 2
        # Intention head determines parameters of Categorical distribution
        means = x.narrow(-1, 0, self.action_dim) * 2
        # TODO: Scale standard deviation
        standard_deviations = (x.narrow(-1, self.action_dim, self.action_dim) / 2 + 0.5) * 0.9 + 0.1
        dist = torch.distributions.Normal(means, standard_deviations)
        aux_dist = torch.distributions.Normal(torch.zeros_like(means), torch.ones_like(standard_deviations))
        if action is None:
            if requires_grad:
                assert sampling_batch is None
                action = aux_dist.sample() * standard_deviations + means
            elif sampling_batch is None:
                action = dist.sample()
            else:
                action = dist.sample([sampling_batch])
        if not noise:
            action = means
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob


class ContinuousCritic(SQXNet):
    """Class for Q-function (or critic) network"""

    def __init__(self,
                 num_intentions=1,
                 state_dim=4,
                 base_hidden_size=64,
                 head_input_size=64,
                 head_hidden_size=32,
                 head_output_size=1,
                 non_linear=torch.nn.ELU(),
                 net_type=TaskHeadCritic,
                 layer_norm=True,
                 use_gpu=True):
        super(ContinuousCritic, self).__init__(state_dim,
                                               base_hidden_size,
                                               num_intentions,
                                               head_input_size,
                                               head_hidden_size,
                                               head_output_size,
                                               non_linear,
                                               net_type,
                                               layer_norm,
                                               use_gpu)

    def forward(self, x, task=None):
        if task is not None or x.dim() == 3:
            return super().forward(x, task)
        elif x.dim() == 4:
            assert x.dim() == 4
            assert x.shape[-2] == self.num_intentions
            x_list = list()
            for i in range(self.num_intentions):
                x_list.append(super().forward(x.narrow(-2, i, 1), task=i).squeeze(-1))
            x = torch.cat(x_list, dim=-1)
            return x
        else:
            assert False

