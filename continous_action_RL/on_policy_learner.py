import copy

import torch
import torch.nn.functional as F
import numpy as np
from continous_action_RL.loss_fn import ActorLoss
from continous_action_RL.loss_fn import Retrace
from continous_action_RL.utils import Utils


class OnPolicyLearner:
    def __init__(self,
                 actor,
                 critic,
                 trajectory_length,
                 discount_factor=0.99,
                 actor_lr=2e-4,
                 critic_lr=2e-4):
        self.actor = actor
        self.critic = critic
        self.trajectory_length = trajectory_length
        self.discount_factor = discount_factor
        self.actor_opt = torch.optim.Adam(params=actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(params=critic.parameters(), lr=critic_lr)

        self.num_actions = actor.num_actions
        self.num_obs = actor.num_obs

        self.actor_loss = ActorLoss()
        self.critic_loss = torch.nn.MSELoss()

    def learn(self, replay_buffer):
        trajectories = replay_buffer.sample(len(replay_buffer))
        state_batch, action_batch, reward_batch, action_prob_batch \
            = Utils.create_batches(trajectories=trajectories,
                                   trajectory_length=self.trajectory_length,
                                   minibatch_size=len(replay_buffer),
                                   num_obs=self.num_obs,
                                   num_actions=self.num_actions)

        Q = self.critic.forward(action_batch, state_batch)
        _, action_log_prob = self.actor.forward(state_batch)

        rewards_t = reward_batch[:, :-1, :].squeeze(-1)
        Q_t = Q[:, :-1, :].squeeze(-1)
        Q_next_t = Q[:, 1:, :].squeeze(-1)
        action_log_prob_t = action_log_prob[:, :-1]

        # Critic update
        self.actor.eval()
        self.critic.train()
        self.critic_opt.zero_grad()
        # TODO the last reward is VERY important, we thus need to have the state_T+1
        TD_target = rewards_t + self.discount_factor * Q_next_t
        critic_loss = F.mse_loss(TD_target.squeeze(-1), Q_t.squeeze(-1))
        critic_loss.backward(retain_graph=True)
        self.critic_opt.step()

        # Actor update
        self.actor.train()
        self.critic.eval()
        self.actor_opt.zero_grad()
        advantage = rewards_t + self.discount_factor * Q_next_t.detach() - Q_t.detach()
        actor_loss = - (advantage * action_log_prob_t).mean()
        actor_loss.backward(retain_graph=True)
        self.actor_opt.step()

        print("actor_loss:", actor_loss.item())
        print("critic_loss:", critic_loss.item())
        print("reward:", torch.sum(reward_batch, dim=1)[-1].item())
        print("-" * 30)
