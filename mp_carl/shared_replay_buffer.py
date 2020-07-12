from collections import namedtuple
from multiprocessing import Manager, Lock
import random
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'action_prob'))


class SharedReplayBuffer(object):
    def __init__(self, num_actions, trajectory_length, num_obs, capacity, lock):
        self.capacity = capacity
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.lock = lock
        self.state_memory = torch.zeros([capacity, trajectory_length, num_obs], dtype=torch.float32)
        self.state_memory.share_memory_()
        self.action_memory = torch.zeros([capacity, trajectory_length, num_actions], dtype=torch.float32)
        self.action_memory.share_memory_()
        self.reward_memory = torch.zeros([capacity, trajectory_length], dtype=torch.float32)
        self.reward_memory.share_memory_()
        self.log_prob_memory = torch.zeros([capacity, trajectory_length, num_actions], dtype=torch.float32)
        self.log_prob_memory.share_memory_()
        self.position = torch.tensor(0)
        self.position.share_memory_()

        self.full = False

    def push(self, states, actions, rewards, log_probs):
        """Saves a transition."""
        with self.lock:
            self.state_memory[self.position] = states
            self.action_memory[self.position] = actions
            self.reward_memory[self.position] = rewards
            self.log_prob_memory[self.position] = log_probs
            self.position += 1
            if self.position == self.capacity - 1:
                self.full = True
                self.position = 0
            print(self.position)

    def sample(self):
        with self.lock:
            if not self.full:
                idx = random.sample(range(self.position), 1)
            else:
                idx = random.sample(range(self.capacity), 1)
        return self.state_memory[idx].squeeze(dim=0), self.action_memory[idx].squeeze(dim=0), \
               self.reward_memory[idx].squeeze(dim=0), self.log_prob_memory[idx].squeeze(dim=0)

    def clear(self):
        self.state_memory = torch.tensor([self.capacity, self.num_obs], dtype=torch.float32)
        self.state_memory.share_memory_()
        self.action_memory = torch.tensor([self.capacity, self.num_actions], dtype=torch.float32)
        self.action_memory.share_memory_()
        self.reward_memory = torch.tensor([self.capacity, 1], dtype=torch.float32)
        self.reward_memory.share_memory_()
        self.log_prob_memory = torch.tensor([self.capacity, self.num_actions], dtype=torch.float32)
        self.log_prob_memory.share_memory_()

    def __len__(self):
        with self.lock:
            return self.position


class SharedReplayBuffer2(object):
    def __init__(self, num_actions, trajectory_length, num_obs, capacity, lock):
        self.capacity = capacity
        self.num_actions = num_actions
        self.num_obs = num_obs
        self.lock = lock
        self.state_memory = Manager().list()
        self.action_memory = Manager().list()
        self.reward_memory = Manager().list()
        self.log_prob_memory = Manager().list()
        self.position = 0

        self.full = False

    def push(self, states, actions, rewards, log_probs):
        """Saves a transition."""
        with self.lock:
            if len(self.state_memory) < self.capacity:
                self.state_memory.append(None)
                self.action_memory.append(None)
                self.reward_memory.append(None)
                self.log_prob_memory.append(None)

            self.state_memory.append(states)
            self.action_memory.append(actions)
            self.reward_memory.append(rewards)
            self.log_prob_memory.append(log_probs)
            if self.position == self.capacity - 1:
                self.full = True
            self.position = (self.position + 1) % self.capacity
            # print(self.position)

    def sample(self):
        with self.lock:
            idx = random.sample(range(len(self.state_memory)), 1)[0]
            # print(self.state_memory[idx])
        return self.state_memory[idx], self.action_memory[idx], self.reward_memory[idx], self.log_prob_memory[idx]

    def clear(self):
        self.state_memory = torch.tensor([self.capacity, self.num_obs], dtype=torch.float32)
        self.state_memory.share_memory_()
        self.action_memory = torch.tensor([self.capacity, self.num_actions], dtype=torch.float32)
        self.action_memory.share_memory_()
        self.reward_memory = torch.tensor([self.capacity, 1], dtype=torch.float32)
        self.reward_memory.share_memory_()
        self.log_prob_memory = torch.tensor([self.capacity, self.num_actions], dtype=torch.float32)
        self.log_prob_memory.share_memory_()

    def __len__(self):
        with self.lock:
            return self.position
