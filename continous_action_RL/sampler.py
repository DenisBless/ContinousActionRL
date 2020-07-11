from collections import namedtuple
import random
import time


import torch
import torch.nn as nn
import numpy as np


Trajectory = namedtuple('Trajectory', ['states', 'actions', 'log_probs', 'rewards'])


class Sampler:

    def __init__(self,
                 actor,
                 env,
                 task_scheduler,
                 replay_buffer,
                 num_trajectories=10,
                 task_period=10,
                 use_gpu=False,
                 continuous=True,
                 reward_scaling_factor=1.0,
                 skip_steps=1,
                 writer=None):

        self.actor = actor
        self.env = env
        self.task_scheduler = task_scheduler
        self.replay_buffer = replay_buffer
        self.num_trajectories = num_trajectories
        self.task_period = task_period
        self.writer = writer
        self.step_counter = 0
        self.use_gpu = use_gpu
        self.continuous = continuous
        self.reward_scaling_factor = reward_scaling_factor
        self.skip_steps = skip_steps

    def sample(self):
        for trajectory_idx in range(self.num_trajectories):
            # print('Acting: trajectory %s of %s' % (trajectory_idx + 1, num_trajectories))
            self.actor.eval()
            observations, actions, log_probs, rewards = list(), list(), list(), list()
            # Reset environment and trajectory specific parameters
            #self.task_scheduler.reset()  # h in paper
            obs = self.env.reset()
            done = False
            num_steps = 0
            # Roll out
            while not done:
                # Sample a new task using the scheduler
                #if num_steps % self.task_period == 0:
                 #   self.task_scheduler.sample()
                # Get the action from current actor policy
                obs = torch.tensor(obs, dtype=torch.float).unsqueeze(dim=0)
                obs = obs.cuda() if self.use_gpu else obs
                action, log_prob = self.actor.predict(obs, task=0)
                # Execute action and collect rewards for each task
                gym_action = action.cpu().squeeze()
                if self.continuous and gym_action.dim() == 0:
                    gym_action = gym_action.unsqueeze(0)
                gym_reward = list()
                for _ in range(self.skip_steps):
                    obs_new, r, done, _ = self.env.step(gym_action)
                    gym_reward.append(r)
                # Modify the main task reward (the huge -100 and 100 values cause instability)
                # Reward is a vector of the reward for each task
                reward = np.mean(gym_reward)
                if self.writer:
                    for i, r in enumerate(reward):
                        self.writer.add_scalar('train/reward/%s' % i, r, self.step_counter)
                # group information into a step and add to current trajectory
                observations.append(obs.detach())
                actions.append(action.detach())
                log_probs.append(log_prob.detach())
                rewards.append(torch.tensor(reward).unsqueeze(-1))
                num_steps += 1
                self.step_counter += 1
                obs = obs_new
            # Add trajectory to replay buffer
            observations = torch.cat(observations).float()
            actions = torch.cat(actions).float()
            log_probs = torch.cat(log_probs).float()
            rewards = torch.stack(rewards).float()
            trajectory = Trajectory(observations, actions, log_probs, rewards)
            self.replay_buffer.append(trajectory)



