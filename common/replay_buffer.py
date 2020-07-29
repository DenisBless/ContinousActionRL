import torch
from torch.multiprocessing import Condition
import random


class ReplayBuffer(object):
    def __init__(self, num_obs: int, num_actions: int, trajectory_length: int, capacity: int, batch_size: int = 1):
        self.capacity = capacity
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.batch_size = batch_size

        self.state_memory = torch.zeros([capacity, trajectory_length, num_obs], dtype=torch.float32)
        self.action_memory = torch.zeros([capacity, trajectory_length, num_actions], dtype=torch.float32)
        self.reward_memory = torch.zeros([capacity, trajectory_length], dtype=torch.float32)
        self.log_prob_memory = torch.zeros([capacity, trajectory_length], dtype=torch.float32)

        self.position = torch.tensor(0)
        self.full = torch.tensor(0)

    def sample(self):
        if not self.full:
            idx = random.sample(range(self.position.item()), self.batch_size)
        else:
            idx = random.sample(range(self.capacity), self.batch_size)
        return self.state_memory[idx].squeeze(dim=0), self.action_memory[idx].squeeze(dim=0), \
               self.reward_memory[idx].squeeze(dim=0), self.log_prob_memory[idx].squeeze(dim=0)

    def push(self, states, actions, rewards, log_probs) -> None:
        self.state_memory[self.position] = states
        self.action_memory[self.position] = actions
        self.reward_memory[self.position] = rewards
        self.log_prob_memory[self.position] = log_probs

        self.position += 1

        if self.position >= self.capacity:
            self.full.fill_(1)
            self.position.zero_()

    def __len__(self):
        if self.full:
            return self.capacity
        else:
            return self.position


class SharedReplayBuffer(ReplayBuffer):
    def __init__(self,
                 num_obs: int,
                 num_actions: int,
                 trajectory_length: int,
                 capacity: int,
                 cv: Condition,
                 batch_size: int = 1):

        super(SharedReplayBuffer, self).__init__(num_obs, num_actions, trajectory_length, capacity, batch_size)

        self.cv = cv
        self.state_memory.share_memory_()
        self.action_memory.share_memory_()
        self.reward_memory.share_memory_()
        self.log_prob_memory.share_memory_()

        self.position.share_memory_()
        self.full.share_memory_()

    def push(self, states, actions, rewards, log_probs) -> None:
        with self.cv:
            assert self.position.is_shared()
            assert self.state_memory.is_shared()

            super().push(states, actions, rewards, log_probs)

    def sample(self):
        return super().sample()
