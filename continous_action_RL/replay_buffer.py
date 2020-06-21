from collections import namedtuple
import torch
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'action_prob'))


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)


# class TorchReplayBuffer(object):
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0
#
#     def push(self, trajectory):
#         """Saves a transition."""
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = trajectory
#         self.position = (self.position + 1) % self.capacity
#
#     def sample(self, batch_size):
#         ...
#         # return random.sample(self.memory, batch_size)
#         # return torch.(self.memory, batch_size)
#
#     def __len__(self):
#         return len(self.memory)
