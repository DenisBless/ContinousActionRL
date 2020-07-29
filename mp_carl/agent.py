import copy

from torch.utils.tensorboard import SummaryWriter

from torch.multiprocessing import current_process, Lock, Condition
from mp_carl.loss_fn import Retrace, ActorLoss
from mp_carl.actor_critic_models import Actor, Critic
import torch
import gym
import numpy as np


class Agent:
    """
    The agent represents one worker which repeatedly samples trajectories from the environment, sends gradients to the
    shared parameter server and evaluates its performance.
    """

    def __init__(self,
                 param_server,
                 shared_replay_buffer,
                 parser_args,
                 condition):

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.param_server = param_server
        self.param_server.init_grad()

        if parser_args.num_workers > 1:
            self.pid = current_process()._identity[0]
        else:
            self.pid = 1

        self.logging = parser_args.logging
        self.logger = None
        if self.pid == 1 and self.logging:  # only create one logger
            self.logger = SummaryWriter()

        self.shared_replay_buffer = shared_replay_buffer
        self.env = gym.make("Swimmer-v2")
        # self.env = gym.make("Pendulum-v0")
        # self.env = gym.make("HalfCheetah-v2")
        self.num_actions = self.env.action_space.shape[0]
        self.num_obs = self.env.observation_space.shape[0]
        self.actor = Actor(num_actions=self.num_actions,
                           num_obs=self.num_obs,
                           log_std_init=np.log(parser_args.init_std)).to(self.device)
        self.critic = Critic(num_actions=self.num_actions, num_obs=self.num_obs).to(self.device)

        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_actor.freeze_net()
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.target_critic.freeze_net()

        self.actor_loss = ActorLoss(alpha=parser_args.entropy_reg)
        self.critic_loss = Retrace(num_actions=self.num_actions)

        self.num_trajectories = parser_args.num_trajectories
        self.update_targnets_every = parser_args.update_targnets_every
        self.learning_steps = parser_args.learning_steps
        self.num_runs = parser_args.num_runs
        self.render = parser_args.render
        self.log_every = parser_args.log_interval

        self.num_grads = parser_args.num_grads
        self.grad_ctr = 0
        self.cv = condition  # TODO: this is just a dummy object

    def run(self) -> None:
        """
        A worker repeatedly samples trajectories from the environment (sample), sends gradients to the parameter server
        (learn) and evaluates its performance (evaluate).

        Returns:
            No return value
        """
        for i in range(self.num_runs):
            import time
            t = time.time()
            self.sample()
            print(time.time() - t)
            self.learn()
            self.evaluate()

    def sample(self) -> None:
        """
        Samples trajectories from the environment and puts them in the shared buffer.

        Returns:
            No return value
        """
        self.actor = self.actor.to(self.device)
        self.critc = self.critic.to(self.device)
        self.target_actor = self.target_actor.to(self.device)
        self.target_critic = self.target_critic.to(self.device)
        self.actor.copy_params(self.param_server.shared_actor)

        for i in range(self.num_trajectories):
            states, actions, rewards, action_log_probs = [], [], [], []

            obs = torch.tensor(self.env.reset(), dtype=torch.float).to(self.device)
            done = False
            while not done:
                mean, log_std = self.actor.forward(obs)
                mean = mean.to(self.device)
                log_std = log_std.to(self.device)
                action, action_log_prob = self.actor.action_sample(mean, log_std)
                action = action.to(self.device)
                next_obs, reward, done, _ = self.env.step(action.detach().cpu())
                next_obs = torch.tensor(next_obs, dtype=torch.float).to(self.device)
                reward = torch.tensor(reward, dtype=torch.float).to(self.device)
                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                action_log_probs.append(action_log_prob)
                obs = next_obs

                if self.render:
                    self.env.render()

            # turn lists into tensors
            states = torch.stack(states).to(self.device)
            actions = torch.stack(actions).to(self.device)
            rewards = torch.stack(rewards).to(self.device)
            action_log_probs = torch.stack(action_log_probs).to(self.device)

            if self.pid == 1 and self.logger is not None and i % self.log_every == 0:
                self.logger.add_scalar(scalar_value=rewards.mean(), tag="Reward/train")

            self.shared_replay_buffer.push(states, actions.detach(), rewards, action_log_probs.detach())

    def learn(self) -> None:
        """
        Calculates gradients w.r.t. the actor and the critic and sends them to a shared parameter server. Whenever
        the server has accumulated G gradients, the parameter of the shared critic and actor are updated and sent
        to the worker. However, the parameters of the shared actor and critic are copied to the worker after each
        iteration since it is unknown to the worker when the gradient updates were happening.

        Returns:
            No return value
        """

        self.actor.copy_params(self.param_server.shared_actor)
        self.critic.copy_params(self.param_server.shared_critic)

        self.actor.train()
        self.critic.train()

        for i in range(self.learning_steps):

            # Update the target networks
            if i % self.update_targnets_every == 0:
                self.update_targnets()

            states, actions, rewards, behaviour_log_pr = self.shared_replay_buffer.sample()

            # Q(a_t, s_t)
            Q = self.critic.forward(actions, states)

            # Q_target(a_t, s_t)
            target_Q = self.target_critic.forward(actions, states)

            # Compute 𝔼_π_target [Q(s_t,•)] with a ~ π_target(•|s_t), log(π_target(a|s)) with 1 sample
            mean, log_std = self.target_actor.forward(states)
            mean, log_std = mean.to(self.device), log_std.to(self.device)

            action_sample, _ = self.target_actor.action_sample(mean, log_std)
            expected_target_Q = self.target_critic.forward(action_sample, states)

            # log(π_target(a_t | s_t))
            target_action_log_prob = self.target_actor.get_log_prob(actions, mean, log_std)

            # a ~ π(•|s_t), log(π(a|s_t))
            current_mean, current_log_std = self.actor.forward(states)
            current_actions, current_action_log_prob = self.actor.action_sample(current_mean, current_log_std)
            current_actions.to(self.device)

            # Q(a, s_t)
            current_Q = self.critic.forward(current_actions, states)

            # Critic update

            critic_loss = self.critic_loss.forward(Q=Q,
                                                   expected_target_Q=expected_target_Q,
                                                   target_Q=target_Q,
                                                   rewards=rewards,
                                                   target_policy_probs=target_action_log_prob,
                                                   behaviour_policy_probs=behaviour_log_pr,
                                                   logger=self.logger)
            self.critic.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.param_server.receive_critic_gradients(self.critic)  # Send critic gradients to the parameter server

            actor_loss = self.actor_loss.forward(Q=current_Q,
                                                 action_log_prob=current_action_log_prob.unsqueeze(-1))
            self.actor.zero_grad()
            actor_loss.backward()
            self.param_server.receive_actor_gradients(self.actor)  # Send actor gradients to the parameter server

            self.grad_ctr += 1

            print(self.param_server.N)
            print(self.grad_ctr)
            #
            if self.grad_ctr == self.num_grads:
                with self.cv:
                    self.cv.wait_for(lambda: self.param_server.N == self.param_server.G)

                    self.actor.copy_params(self.param_server.shared_actor)
                    self.critic.copy_params(self.param_server.shared_critic)

                    self.grad_ctr = 0

            # Keep track of different values
            if self.pid == 1 and self.logging and i % self.log_every == 0:
                self.logger.add_scalar(scalar_value=actor_loss.item(), tag="Loss/Actor_loss")
                self.logger.add_scalar(scalar_value=critic_loss.item(), tag="Loss/Critic_loss")
                self.logger.add_scalar(scalar_value=current_log_std.exp().mean(), tag="Statistics/Action_std")

                self.logger.add_scalar(scalar_value=list(self.critic.parameters())[-1].item(), tag="Critic/param")
                self.logger.add_scalar(scalar_value=list(self.critic.parameters())[-1].grad, tag="Critic/grad")

                # self.logger.add_scalar(scalar_value=Q[0], tag="Q_/Q0")
                # self.logger.add_scalar(scalar_value=Q[-1], tag="Q_/QT")
                # self.logger.add_scalar(scalar_value=target_Q[0], tag="targetQ/targetQ0")
                # self.logger.add_scalar(scalar_value=target_Q[-1], tag="targetQ/targetQT")
                # self.logger.add_scalar(scalar_value=expected_target_Q[0], tag="V/V0")
                # self.logger.add_scalar(scalar_value=expected_target_Q[-1], tag="V/VT")

                self.logger.add_histogram(values=current_mean, tag="Statistics/Action_mean")
                self.logger.add_histogram(values=rewards.sum(dim=-1), tag="Cumm Reward/Action_mean")
                # print(current_mean[:10])
                self.logger.add_histogram(values=current_actions, tag="Statistics/Action")

        self.actor.copy_params(self.param_server.shared_actor)
        self.critic.copy_params(self.param_server.shared_critic)

    def evaluate(self) -> None:
        """
        Evaluates the performance of the learned policy for the agent.

        Returns:
            No return value

        """
        self.actor.copy_params(self.param_server.shared_actor)
        self.actor.eval()  # Eval mode: Sets the action variance to zero, disables batch-norm and dropout etc.

        obs = torch.tensor(self.env.reset(), dtype=torch.float)
        obs = obs.to(self.device)
        with torch.no_grad():
            rewards = []
            done = False
            while not done:
                mean, log_std = self.actor.forward(obs)
                mean = mean.to(self.device)
                log_std = log_std.to(self.device)
                action, action_log_prob = self.actor.action_sample(mean, log_std)
                action = action.to(self.device)
                next_obs, reward, done, _ = self.env.step(action.detach().cpu().numpy())
                rewards.append(reward)
                obs = torch.tensor(next_obs, dtype=torch.float).to(self.device)

                if done:
                    obs = torch.tensor(self.env.reset(), dtype=torch.float).to(self.device)
                    print("Mean reward: ", np.mean(rewards))
                    if self.pid == 1 and self.logger is not None:
                        self.logger.add_scalar(scalar_value=np.mean(rewards), tag="Reward/test")

        self.actor.train()  # Back to train mode

    def update_targnets(self, smoothing_coefficient=1) -> None:
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
                    torch.add(a_target_param.data, a_param.data, alpha=smoothing_coefficient, out=a_target_param.data)

                for c_param, c_target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                    c_target_param.data.mul_(1 - smoothing_coefficient)
                    torch.add(c_target_param.data, c_param.data, alpha=smoothing_coefficient, out=c_target_param.data)
