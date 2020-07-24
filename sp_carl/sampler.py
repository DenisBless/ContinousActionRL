import torch
import gym
from common.replay_buffer import SharedReplayBuffer
from torch.multiprocessing import current_process


class Sampler:
    def __init__(self,
                 actor: torch.nn.Module,
                 replay_buffer: SharedReplayBuffer,
                 argp,
                 logger=None, env=None):

        self.actor = actor
        self.replay_buffer = replay_buffer
        self.num_samples = argp.num_trajectories
        self.log_every = argp.log_interval

        self.logger = logger
        if argp.num_worker > 1:
            self.pid = current_process()._identity[0]  # process ID
        else:
            self.pid = 1

        # self.env = gym.make("Swimmer-v2")
        # self.env = gym.make("Pendulum-v0")
        # self.env = gym.make("HalfCheetah-v2")
        self.env = env

    def run(self):
        for i in range(self.num_samples):
            states, actions, rewards, action_log_probs, dones = [], [], [], [], []

            obs = torch.tensor(self.env.reset(), dtype=torch.float)
            # obs = self.env.reset()
            done = False
            while not done:
                mean, log_std = self.actor.forward(obs)
                action, action_log_prob = self.actor.action_sample(mean, log_std)
                action_np = action.detach().cpu().numpy().clip(min=self.env.action_space.low, max=self.env.action_space.high)
                next_obs, reward, done, _ = self.env.step(action_np)
                next_obs = torch.tensor(next_obs, dtype=torch.float)
                reward = torch.tensor(reward, dtype=torch.float)
                done = torch.tensor(done, dtype=torch.bool)
                # reward = reward.clone().detach()
                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                action_log_probs.append(action_log_prob)
                dones.append(done)
                obs = next_obs

                # if self.render:
                #     self.env.render()

            # turn lists into tensors
            states = torch.stack(states)
            actions = torch.stack(actions)
            rewards = torch.stack(rewards)
            action_log_probs = torch.stack(action_log_probs)
            dones = torch.stack(dones)

            if states.shape[0] < self.env._max_episode_steps:
                pad = self.env._max_episode_steps - states.shape[0]
                states = torch.nn.functional.pad(states, pad=(0, 0, 0, pad), mode='constant', value=0)
                actions = torch.nn.functional.pad(actions, pad=(0, 0, 0, pad), mode='constant', value=0)
                rewards = torch.nn.functional.pad(rewards, pad=(0, pad), mode='constant', value=0)
                action_log_probs = torch.nn.functional.pad(action_log_probs, pad=(0, pad), mode='constant', value=0)
                dones = torch.nn.functional.pad(dones, pad=(0, pad), mode='constant', value=True)

            # # turn lists into tensors
            # states = torch.tensor(states)
            # actions = torch.tensor(actions)
            # rewards = torch.tensor(rewards)
            # action_log_probs = torch.stack(action_log_probs)

            if self.pid == 1 and self.logger is not None and i % self.log_every == 0:
                self.logger.add_scalar(scalar_value=rewards.mean(), tag="Reward/train")

            self.replay_buffer.push(states, actions.detach(), rewards, action_log_probs.detach(), dones)
