import copy

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
                 logger=None):

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.actor = actor
        self.critic = critic

        self.logger = logger
        self.trajectory_length = trajectory_length
        self.discount_factor = discount_factor
        self.target_actor = copy.deepcopy(actor).to(self.device)
        self.target_critic = copy.deepcopy(critic).to(self.device)
        Utils.freeze_net(self.target_actor)
        Utils.freeze_net(self.target_critic)
        self.actor_opt = torch.optim.Adam(params=actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(params=critic.parameters(), lr=critic_lr)

        self.num_training_iter = num_training_iter
        self.update_targnets_every = update_targnets_every
        self.expectation_samples = expectation_samples
        self.minibatch_size = minibatch_size
        self.gradient_clip_val = gradient_clip_val

        self.num_actions = actor.num_actions
        self.num_obs = actor.num_obs

        self.actor_loss = ActorLoss(entropy_regularization_on=entropy_regularization_on,
                                    alpha=entropy_regularization)
        self.critic_loss = Retrace()

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

                trajectories = replay_buffer.sample(self.minibatch_size)
                state_batch, action_batch, reward_batch, action_prob_batch \
                    = Utils.create_batches(trajectories=trajectories,
                                           trajectory_length=self.trajectory_length,
                                           minibatch_size=self.minibatch_size,
                                           num_obs=self.num_obs,
                                           num_actions=self.num_actions)

                # Q(a_t, s_t)
                Q = self.critic.forward(action_batch, state_batch)

                # Q_target(a_t, s_t)
                target_Q = self.target_critic.forward(action_batch, state_batch)

                # Compute 𝔼_π_target [Q(s_t,•)] with a ~ π_target(•|s_t), log(π_target(a|s))
                expected_target_Q = torch.zeros_like(reward_batch)
                mean, std = self.target_actor.forward(state_batch)
                mean = mean.to(self.device)
                std = std.to(self.device)
                for _ in range(self.expectation_samples):
                    action_sample, _ = self.target_actor.action_sample(mean, std)
                    expected_target_Q += self.target_critic.forward(action_sample, state_batch)
                expected_target_Q /= self.expectation_samples

                # log(π_target(a_t | s_t))
                target_action_log_prob = self.target_actor.get_log_prob(action_batch, mean, std)

                # a ~ π(•|s_t), log(π(a|s))
                actions, action_log_prob = self.actor.forward(state_batch)
                actions.to(self.device)

                # Q(a, s_t)
                current_Q = self.critic.forward(actions, state_batch)

                # Critic update
                self.actor.eval()
                self.critic.train()
                self.critic_opt.zero_grad()

                critic_loss = self.critic_loss.forward(Q=Q.squeeze(-1),
                                                       expected_target_Q=expected_target_Q.squeeze(-1),
                                                       target_Q=target_Q.squeeze(-1),
                                                       rewards=reward_batch.squeeze(-1),
                                                       target_policy_probs=torch.exp(target_action_log_prob.squeeze(-1)),
                                                       behaviour_policy_probs=torch.exp(action_prob_batch.squeeze(-1)),
                                                       recursive=True)

                critic_loss.backward(retain_graph=True)

                # Actor update
                self.actor.train()
                self.critic.eval()
                self.actor_opt.zero_grad()

                actor_loss = self.actor_loss.forward(Q=current_Q.squeeze(-1),
                                                     action_log_prob=action_log_prob.squeeze(-1))
                actor_loss.backward()

                # Keep track of various values
                if self.logger is not None and j % self.logger.log_every == 0:
                    # self.logger.log_DNN_params(self.actor, name="Actor")
                    self.logger.log_DNN_gradients(self.actor, name="Actor")
                    # self.logger.log_DNN_params(self.critic, name="Critic")
                    self.logger.log_DNN_gradients(self.critic, name="Critic")

                    self.logger.add_scalar(scalar_value=actor_loss.item(), tag="Actor_loss")
                    self.logger.add_scalar(scalar_value=critic_loss.item(), tag="Critic_loss")
                    self.logger.add_scalar(scalar_value=std.mean().item(), tag="Action_std")
                    self.logger.add_histogram(values=mean, tag="Action_mean")

                # Gradient update step with gradient clipping
                if self.gradient_clip_val is not None:
                    torch.nn.utils.clip_grad_value_(self.critic.parameters(), self.gradient_clip_val)
                    torch.nn.utils.clip_grad_value_(self.actor.parameters(), self.gradient_clip_val)

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
