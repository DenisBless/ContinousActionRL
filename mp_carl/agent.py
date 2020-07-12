from mp_carl.actor_critic_networks import Actor, Critic
import torch


class Agent:
    def __init__(self,
                 manager,
                 shared_replay_buffer,
                 arg_parser,
                 logger=None):

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.manager = manager
        self.shared_replay_buffer = shared_replay_buffer
        self.logger = logger
        self.num_trajectories = arg_parser.num_trajectories
        self.env = ...
        self.num_actions = self.env.action_space.shape[0]
        self.num_obs = self.env.observation_space.shape[0]

    def sample(self):
        actor = Actor(num_actions=self.num_actions, num_obs=self.num_obs)
        actor.copy_params(self.manager.shared_actor)

        for i in range(self.num_trajectories):
            states, actions, rewards, action_log_probs = [], [], [], []

            obs = torch.tensor(self.env.reset(), dtype=torch.float).to(self.device)
            done = False
            while not done:
                mean, std = actor.forward(observation=obs)
                mean = mean.to(self.device)
                std = std.to(self.device)
                action, action_log_prob = actor.action_sample(mean, std)
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

