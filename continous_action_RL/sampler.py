import torch


class Sampler:
    def __init__(self,
                 env,
                 num_trajectories,
                 actor_network,
                 replay_buffer,
                 render=False,
                 logger=None):

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.env = env
        self.logger = logger
        self.num_trajectories = num_trajectories
        self.actor_network = actor_network.to(self.device)
        self.render = render
        self.replay_buffer = replay_buffer

    def collect_trajectories(self):
        for i in range(self.num_trajectories):
            states, actions, rewards, action_log_probs = [], [], [], []

            obs = torch.tensor(self.env.reset(), dtype=torch.float).to(self.device)
            done = False
            while not done:
                mean, std = self.actor_network.forward(observation=obs)
                mean = mean.to(self.device)
                std = std.to(self.device)
                action, action_log_prob = self.actor_network.action_sample(mean, std)
                action = action.to(self.device)
                next_obs, reward, done, _ = self.env.step([action.item()])
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

            if self.logger is not None and i % self.logger.log_every == 0:
                print(rewards.mean().item())
                self.logger.add_scalar(scalar_value=rewards.mean(), tag="sampler_reward")
                # self.logger.add_histogram(values=actions, tag="actions")

            self.replay_buffer.push(states, actions.detach(), rewards, action_log_probs.detach())
