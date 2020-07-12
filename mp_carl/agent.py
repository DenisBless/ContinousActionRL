import copy

from mp_carl.loss_fn import Retrace, ActorLoss
from mp_carl.actor_critic_networks import Actor, Critic
import torch
import numpy as np


class Agent:
    def __init__(self,
                 param_server,
                 shared_replay_buffer,
                 arg_parser,
                 logger=None):

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.param_server = param_server
        self.shared_replay_buffer = shared_replay_buffer
        self.logger = logger
        self.env = ...
        self.num_actions = self.env.action_space.shape[0]
        self.num_obs = self.env.observation_space.shape[0]

        self.actor = Actor(num_actions=self.num_actions, num_obs=self.num_obs).to(self.device)
        self.critic = Critic(num_actions=self.num_actions, num_obs=self.num_obs).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)

        self.actor_loss = ActorLoss()
        self.critic_loss = Retrace()

        self.num_trajectories = arg_parser.num_trajectories
        self.update_targnets_every = arg_parser.update_targnets_every
        self.learning_steps = arg_parser.num_learning_iter
        self.num_runs = arg_parser.num_runs
        self.global_gradient_norm = arg_parser.global_gradient_norm

    def run(self):
        for i in range(self.num_runs):
            self.sample()
            self.learn()
            self.evaluate()

    def sample(self):
        self.actor.copy_params(self.param_server.shared_actor)

        for i in range(self.num_trajectories):
            states, actions, rewards, action_log_probs = [], [], [], []

            obs = torch.tensor(self.env.reset(), dtype=torch.float).to(self.device)
            done = False
            while not done:
                mean, std = self.actor.forward(observation=obs)
                mean = mean.to(self.device)
                std = std.to(self.device)
                action, action_log_prob = self.actor.action_sample(mean, std)
                action = action.to(self.device)
                next_obs, reward, done, _ = self.env.step(action.detach().cpu().numpy())
                next_obs = torch.tensor(next_obs, dtype=torch.float).to(self.device)
                reward = torch.tensor(reward, dtype=torch.float).to(self.device)
                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                action_log_probs.append(action_log_prob)
                obs = next_obs

            # turn lists into tensors
            states = torch.stack(states).to(self.device)
            actions = torch.stack(actions).to(self.device)
            rewards = torch.stack(rewards).to(self.device)
            action_log_probs = torch.stack(action_log_probs).to(self.device)

            if self.logger is not None and i % self.logger.log_every == 0:
                self.logger.add_scalar(scalar_value=rewards.mean(), tag="Reward/train")

            # todo put? shat structure has the shared replay buffer
            self.shared_replay_buffer.push(states, actions.detach(), rewards, action_log_probs.detach())

    def learn(self):

        for i in range(self.learning_steps):
            self.actor.copy_params(self.param_server.shared_actor)
            self.critic.copy_params(self.param_server.shared_critic)

            # Update the target networks
            if i % self.update_targnets_every == 0:
                self.update_targnets()

            self.actor.train()
            self.critic.train()

            trajectory = self.shared_replay_buffer.sample()
            states = trajectory.states
            actions = trajectory.actions
            rewards = trajectory.rewards
            behaviour_log_pr = trajectory.log_probs

            # Q(a_t, s_t)
            Q = self.critic.forward(actions, states)

            # Q_target(a_t, s_t)
            target_Q = self.target_critic.forward(actions, states)

            # Compute 𝔼_π_target [Q(s_t,•)] with a ~ π_target(•|s_t), log(π_target(a|s))
            mean, std = self.target_actor.forward(states)
            mean, std = mean.to(self.device), std.to(self.device)
            action_sample, _ = self.target_actor.action_sample(mean, std)
            expected_target_Q = self.target_critic.forward(action_sample, states)

            # log(π_target(a_t | s_t))
            target_action_log_prob = self.target_actor.get_log_prob(actions, mean, std)

            # a ~ π(•|s_t), log(π(a|s))
            current_mean, current_std = self.actor.forward(states)
            current_actions, current_action_log_prob = self.actor.action_sample(current_mean, current_std)
            actions.to(self.device)

            # Q(a, s_t)
            current_Q = self.critic.forward(current_actions, states)

            # Critic update
            self.actor.eval()
            self.critic.train()

            critic_loss = self.critic_loss.forward(Q=Q.squeeze(-1),
                                                   expected_target_Q=expected_target_Q.squeeze(-1),
                                                   target_Q=target_Q.squeeze(-1),
                                                   rewards=rewards.squeeze(-1),
                                                   target_policy_probs=target_action_log_prob.squeeze(-1),
                                                   behaviour_policy_probs=behaviour_log_pr.squeeze(-1),
                                                   logger=self.logger)

            critic_loss.backward(retain_graph=True)

            # Actor update
            self.actor.train()
            self.critic.eval()

            actor_loss = self.actor_loss.forward(Q=current_Q.squeeze(-1),
                                                 action_log_prob=current_action_log_prob.squeeze(-1))

            actor_loss.backward()

            self.param_server.receive_gradients(actor=self.actor, critic=self.critic)

    def evaluate(self):
        self.actor.copy_params(self.param_server.shared_actor)
        self.actor.eval()  # Eval mode: Sets the action variance to zero, disables batch-norm and dropout etc.

        obs = torch.tensor(self.env.reset(), dtype=torch.float)
        obs = obs.to(self.device)
        with torch.no_grad():
                rewards = []
                done = False
                while not done:
                    mean, std = self.actor.forward(observation=obs)
                    mean = mean.to(self.device)
                    std = std.to(self.device)
                    action, action_log_prob = self.actor.action_sample(mean, std)
                    action = action.to(self.device)
                    next_obs, reward, done, _ = self.env.step(action.detach().cpu().numpy())
                    rewards.append(reward)
                    obs = torch.tensor(next_obs, dtype=torch.float).to(self.device)

                    if done:
                        obs = torch.tensor(self.env.reset(), dtype=torch.float).to(self.device)
                        print("Mean reward: ", np.mean(rewards))

        self.actor.train()  # Back to train mode

    def update_targnets(self):
        """
        Update the target actor and the target critic by copying the parameter from the updated networks.

        Returns:
            No return value
        """
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
