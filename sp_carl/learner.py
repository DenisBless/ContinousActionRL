import copy

import torch
from torch.nn.utils import clip_grad_norm_
from common.loss_fn import ActorLoss, Retrace
from common.replay_buffer import SharedReplayBuffer


class Learner:
    def __init__(self,
                 actor: torch.nn.Module,
                 critic: torch.nn.Module,
                 replay_buffer: SharedReplayBuffer,
                 device: str,
                 num_actions: int,
                 num_obs: int,
                 argp,
                 logger=None):

        self.actor = actor
        self.critic = critic

        self.target_actor = copy.deepcopy(self.actor).to(device)
        self.target_actor.freeze_net()
        self.target_critic = copy.deepcopy(self.critic).to(device)
        self.target_critic.freeze_net()

        self.replay_buffer = replay_buffer

        self.num_actions = num_actions
        self.num_obs = num_obs
        self.device = device

        self.logger = logger
        self.logging = argp.logging
        self.log_every = argp.log_interval

        self.actor_loss = ActorLoss(alpha=argp.entropy_reg)
        self.critic_loss = Retrace(num_actions=self.num_actions, reward_scale=argp.reward_scale)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), argp.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), argp.critic_lr)

        self.update_targnets_every = argp.update_targnets_every
        self.learning_steps = argp.learning_steps
        self.smoothing_coefficient = argp.smoothing_coefficient
        self.global_gradient_norm = argp.global_gradient_norm

    def learn(self) -> None:
        """
        Calculates gradients w.r.t. the actor and the critic and sends them to a shared parameter server. Whenever
        the server has accumulated G gradients, the parameter of the shared critic and actor are updated and sent
        to the worker. However, the parameters of the shared actor and critic are copied to the worker after each
        iteration since it is unknown to the worker when the gradient updates were happening.

        Returns:
            No return value
        """
        self.actor.train()
        self.critic.train()

        for i in range(self.learning_steps):

            # Update the target networks
            if i % self.update_targnets_every == 0:
                self.update_targnets(smoothing_coefficient=self.smoothing_coefficient)

            states, actions, rewards, behaviour_log_pr = self.replay_buffer.sample()
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            behaviour_log_pr = behaviour_log_pr.to(self.device)

            # Q(a_t, s_t)
            batch_Q = self.critic(torch.cat([actions, states], dim=-1))

            # Q_target(a_t, s_t)
            target_Q = self.target_critic(torch.cat([actions, states], dim=-1))

            # Compute 𝔼_π_target [Q(s_t,•)] with a ~ π_target(•|s_t), log(π_target(a|s)) with 1 sample
            mean, log_std = self.target_actor(states)
            mean, log_std = mean.to(self.device), log_std.to(self.device)

            action_sample, _ = self.target_actor.action_sample(mean, log_std)
            # action_sample = torch.tanh(mean)
            expected_target_Q = self.target_critic(torch.cat([action_sample, states], dim=-1))

            # log(π_target(a_t | s_t))
            target_action_log_prob = self.target_actor.get_log_prob(actions=actions, mean=mean, log_std=log_std)

            # a ~ π(•|s_t), log(π(a|s_t))
            current_mean, current_log_std = self.actor(states)
            current_actions, current_action_log_prob = self.actor.action_sample(current_mean, current_log_std)
            current_actions.to(self.device)

            # Reset the gradients
            self.critic.zero_grad()
            self.actor.zero_grad()

            # Critic update
            critic_loss = self.critic_loss(Q=batch_Q,
                                           expected_target_Q=expected_target_Q,
                                           target_Q=target_Q,
                                           rewards=rewards,
                                           target_policy_probs=target_action_log_prob,
                                           behaviour_policy_probs=behaviour_log_pr,
                                           logger=self.logger)

            self.critic.zero_grad()
            critic_loss.backward()
            if self.global_gradient_norm != -1:
                clip_grad_norm_(self.critic.parameters(), self.global_gradient_norm)
            self.critic_opt.step()

            # Q(a, s_t)
            current_Q = self.critic(torch.cat([current_actions, states], dim=-1))

            # Actor update
            actor_loss = self.actor_loss(Q=current_Q, action_log_prob=current_action_log_prob.unsqueeze(-1))
            self.actor.zero_grad()
            actor_loss.backward()
            if self.global_gradient_norm != -1:
                clip_grad_norm_(self.actor.parameters(), self.global_gradient_norm)
            self.actor_opt.step()

            # Keep track of different values
            if self.logging and i % self.log_every == 0:
                self.logger.add_scalar(scalar_value=actor_loss.item(), tag="Loss/Actor_loss")
                self.logger.add_scalar(scalar_value=critic_loss.item(), tag="Loss/Critic_loss")
                self.logger.add_scalar(scalar_value=current_log_std.exp().mean(), tag="Statistics/Action_std_mean")
                self.logger.add_scalar(scalar_value=current_log_std.exp().std(), tag="Statistics/Action_std_std")
                self.logger.add_scalar(scalar_value=batch_Q.mean(), tag="Statistics/Q")

                self.logger.add_scalar(scalar_value=self.critic.param_norm, tag="Critic/param norm")
                self.logger.add_scalar(scalar_value=self.critic.grad_norm, tag="Critic/grad norm")
                self.logger.add_scalar(scalar_value=self.actor.param_norm, tag="Actor/param norm")
                self.logger.add_scalar(scalar_value=self.actor.grad_norm, tag="Actor/grad norm")

                self.logger.add_histogram(values=current_mean, tag="Statistics/Action_mean")
                self.logger.add_histogram(values=rewards.sum(dim=-1), tag="Cumm Reward/Action_mean")
                # print(current_mean[:10])
                self.logger.add_histogram(values=current_actions, tag="Statistics/Action")

    def update_targnets(self, smoothing_coefficient=1.) -> None:
        """
        Update the target actor and the target critic by copying the parameter from the updated networks. If the
        smoothing coefficient is 1 then updates are hard otherwise the parameter update is smoothed according to.

        param' = (1 - smoothing_coefficient) * target param + smoothing_coefficient * param

        Returns:
            No return value
        """
        if smoothing_coefficient == 1:
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())
        else:
            assert 0 < smoothing_coefficient < 1
            with torch.no_grad():
                for a_param, a_target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                    a_target_param.data.mul_(1 - smoothing_coefficient)
                    torch.add(a_target_param.data, a_param.data, alpha=smoothing_coefficient,
                              out=a_target_param.data)

                for c_param, c_target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                    c_target_param.data.mul_(1 - smoothing_coefficient)
                    torch.add(c_target_param.data, c_param.data, alpha=smoothing_coefficient,
                              out=c_target_param.data)
