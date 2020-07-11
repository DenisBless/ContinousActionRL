import torch
import torch.nn as nn
import numpy as np


class TaskHeadBase(torch.nn.Module):
    """Generic class for a single intention head (used within actor/critic networks)"""

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 non_linear,
                 use_gpu=True):
        super(TaskHeadBase, self).__init__()
        self.non_linear = non_linear
        self.use_gpu = use_gpu

        # Build the network
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.final_layer = nn.Linear(hidden_size, output_size)
        #self.init_weights()

    def init_weights(self):
        # Initialize the other layers with xavier (still constant 0 bias)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.constant_(self.layer1.bias, 0)
        nn.init.xavier_uniform_(self.final_layer.weight)
        nn.init.constant_(self.final_layer.bias, 0)

    def forward(self, x):
        x = self.non_linear(self.layer1(x))
        x = self.final_layer(x)
        return x


class TaskHeadCritic(TaskHeadBase):
    """Class for a single Task head within the Q-function (or critic) network"""

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 non_linear=torch.nn.ELU(),
                 use_gpu=True):
        super(TaskHeadCritic, self).__init__(input_size, hidden_size, output_size, non_linear,
                                             use_gpu)


class TaskHeadActor(TaskHeadBase):
    """Class for a single Task head within the policy (or actor) network"""

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 non_linear=nn.ELU(),
                 use_gpu=True):
        super(TaskHeadActor, self).__init__(input_size, hidden_size, output_size, non_linear,
                                            use_gpu)


class SQXNet(torch.nn.Module):
    """Generic class for actor and critic networks. The arch is very similar."""

    def __init__(self,
                 state_dim,
                 base_hidden_size,
                 num_intentions,
                 head_input_size,
                 head_hidden_size,
                 head_output_size,
                 non_linear,
                 intention_net_type,
                 layer_norm=False,
                 use_gpu=True):
        super(SQXNet, self).__init__()
        self.non_linear = non_linear
        self.layer_norm = layer_norm
        self.use_gpu = use_gpu
        self.num_intentions = num_intentions
        self.state_dim = state_dim

        # Build the base of the network
        self.layer1 = nn.Linear(state_dim, base_hidden_size)
        self.layer2 = nn.Linear(base_hidden_size, head_input_size)
        if self.layer_norm:
            self.ln1 = nn.LayerNorm(base_hidden_size)
        #self.init_weights()

        # Create the many intention nets heads

        self.intention_nets = []
        for _ in range(num_intentions):
            self.intention_nets.append(intention_net_type(input_size=head_input_size,
                                                          hidden_size=head_hidden_size,
                                                          output_size=head_output_size,
                                                          use_gpu=use_gpu,
                                                          non_linear=non_linear))

        self.intention_nets = nn.ModuleList(self.intention_nets)

    def init_weights(self):
        # Initialize the other layers with xavier (still constant 0 bias)
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.constant_(self.layer1.bias, 0)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.constant_(self.layer2.bias, 0)

    def forward(self, x, task=None):
        # Feed the input through the base layers of the model
        x = self.layer1(x)
        x = self.non_linear(x)
        if self.layer_norm:
            x = self.ln1(x)
        x = self.non_linear(self.layer2(x))
        if task is not None:  # single intention head
            x = self.intention_nets[task](x)
            return x
        else:
            # TODO: For Single output (discrete) actors. is this squeeze() a problem?
            x = [net(x) for net in self.intention_nets]
            x = torch.stack(x, dim=-2).squeeze(-1)
            return x

