import copy
import random

import torch
from continous_action_RL.off_policy.actor_loss import ActorLoss
from continous_action_RL.off_policy.retrace_loss import Retrace
from continous_action_RL.utils import Utils


class OffPolicyLearner:
    def __init__(self,
                 actor,
                 critic,
                 trajectory_length,
                 discount_factor=0.99,
                 actor_lr=2e-4,
                 critic_lr=2e-4,
                 entropy_regularization_on=True,
                 entropy_regularization=1e-3,
                 gradient_clip_val=None,
                 num_training_iter=100,
                 update_targnets_every=20,
                 expectation_samples=10,
                 minibatch_size=8,
                 logger=None,
                 use_gpu=False
                 ):

        self.use_gpu = use_gpu

        self.actor = actor
        self.critic = critic

        self.logger = logger
        self.trajectory_length = trajectory_length
        self.discount_factor = discount_factor
        self.target_actor = copy.deepcopy(actor)
        self.target_critic = copy.deepcopy(critic)
        Utils.freeze_net(self.target_actor)
        Utils.freeze_net(self.target_critic)
        self.actor_opt = torch.optim.Adam(params=actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(params=critic.parameters(), lr=critic_lr)

        self.num_training_iter = num_training_iter
        self.update_targnets_every = update_targnets_every
        self.expectation_samples = expectation_samples
        self.minibatch_size = minibatch_size
        self.gradient_clip_val = gradient_clip_val

        self.num_actions = 1
        self.num_obs = 3

        self.actor_loss = ActorLoss()
        self.critic_loss = Retrace()

    @staticmethod
    def get_critic_input(actions, states) -> torch.Tensor:
        if actions.dim() == 4 and states.dim() == 3:
            states = states.unsqueeze(-2).expand(*([-1] * (actions.dim() - 2) + [1, -1]))

        assert actions.dim() == states.dim()
        critic_input = torch.cat([actions, states], dim=-1)
        return critic_input

    def get_batch(self, replay_buffer, size=1):
        """
        returns -- (states, actions, log_probs, rewards)
        """
        trajectories = random.choices(replay_buffer, k=size)
        tensors = [
            torch.stack([
                trajectory[tensor]
                for trajectory in trajectories
            ])
            for tensor in range(4)
        ]
        return tuple([t.cuda() if self.use_gpu else t for t in tensors])

    def learn(self, replay_buffer):
        """Update the actor and critic networks using trajectories from the replay buffer.

        Args:
            replay_buffer: Replay buffer containing trajectories.

        Returns:
            No return value
        """
        for i in range(self.num_training_iter):
            for j in range(self.update_targnets_every):

                self.actor.train()
                self.critic.train()

                state_batch, action_batch, action_prob_batch, reward_batch \
                    = self.get_batch(replay_buffer, self.minibatch_size)

                # Q(a_t, s_t)
                Q = self.critic.forward(self.get_critic_input(action_batch, state_batch))

                # Q_target(a_t, s_t)
                target_Q = self.target_critic.forward(self.get_critic_input(action_batch, state_batch))

                # Compute 𝔼_π_target [Q(s_t,•)] with a ~ π_target(•|s_t), log(π_target(a|s))
                action_sample, _ = self.target_actor.predict(state_batch)
                expected_target_Q = self.target_critic.forward(self.get_critic_input(action_sample, state_batch))

                # log(π_target(a_t | s_t))
                _, target_action_log_prob = self.target_actor.predict(state_batch, action=action_batch.unsqueeze(-2))

                # a ~ π(•|s_t), log(π(a|s))
                actions, action_log_prob = self.actor.predict(state_batch, requires_grad=True)

                # Q(a, s_t)
                current_Q = self.critic.forward(self.get_critic_input(actions, state_batch))

                # Critic update
                self.actor.eval()
                self.critic.train()
                self.critic_opt.zero_grad()

                critic_loss = self.critic_loss.forward(Q=Q.squeeze(-1),
                                                       expected_target_Q=expected_target_Q.squeeze(-1),
                                                       target_Q=target_Q.squeeze(-1),
                                                       rewards=reward_batch.squeeze(-1),
                                                       target_policy_probs=torch.exp(
                                                           target_action_log_prob.squeeze(-1)),
                                                       behaviour_policy_probs=torch.exp(action_prob_batch.squeeze(-1)),
                                                       recursive=True)

                critic_loss.backward(retain_graph=True)

                # Actor update
                self.actor.train()
                self.critic.eval()
                self.actor_opt.zero_grad()

                actor_loss = self.actor_loss.forward(current_Q.squeeze(-1), action_log_prob.squeeze(-1))
                actor_loss.backward()

                # Gradient update step with gradient clipping
                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip_val)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip_val)

                # Keep track of various values
                if self.logger is not None and j % self.logger.log_every == 0:
                    # self.logger.log_DNN_params(self.actor, name="Actor")
                    # self.logger.log_DNN_gradients(self.actor, name="Actor")
                    # self.logger.log_DNN_params(self.critic, name="Critic")
                    # self.logger.log_DNN_gradients(self.critic, name="Critic")

                    self.logger.add_scalar(scalar_value=actor_loss.item(), tag="Loss/Actor_loss", global_step=j)
                    self.logger.add_scalar(scalar_value=critic_loss.item(), tag="Loss/Critic_loss", global_step=j)
                    self.logger.add_scalar(scalar_value=std.mean().item(), tag="Action_std_mean", global_step=j)
                    self.logger.add_histogram(values=mean, tag="Statistics/Action_mean", global_step=j)
                    self.logger.add_histogram(values=std, tag="Statistics/Action_std", global_step=j)
                    if self.num_actions > 1:
                        self.logger.add_histogram(values=actions[:, :, 0], tag="Action/x", global_step=j)
                        self.logger.add_histogram(values=actions[:, :, 1], tag="Action/y", global_step=j)
                        self.logger.add_histogram(values=actions[:, :, 2], tag="Action/z", global_step=j)

                self.critic_opt.step()
                self.actor_opt.step()

            # Update the target networks
            self.update_targnets()

    def update_targnets(self):
        """
        Update the target actor and the target critic by copying the parameter from the updated networks.

        Returns:
            No return value
        """
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
