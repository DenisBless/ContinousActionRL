import torch


class Sampler:
    def __init__(self,
                 env,
                 num_trajectories,
                 actor_network,
                 replay_buffer,
                 render=False,
                 logger=None,
                 use_gpu=False
                 ):


        self.env = env
        self.logger = logger
        self.num_trajectories = num_trajectories
        self.actor_network = actor_network
        self.render = render
        self.replay_buffer = replay_buffer

        self.use_gpu = use_gpu

    def collect_trajectories(self):
        for i in range(self.num_trajectories):
            states, actions, rewards, action_log_probs = [], [], [], []

            obs = torch.tensor(self.env.reset(), dtype=torch.float)
            done = False
            while not done:
                obs = obs.cuda() if self.use_gpu else obs
                mean, std = self.actor_network.forward(observation=obs)
                action, action_log_prob = self.actor_network.action_sample(mean, std)
                next_obs, reward, done, _ = self.env.step(action.detach().cpu().numpy())
                next_obs = torch.tensor(next_obs, dtype=torch.float)
                next_obs = next_obs.cuda() if self.use_gpu else next_obs
                reward = torch.tensor(reward, dtype=torch.float)
                reward = reward.cuda() if self.use_gpu else reward

                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                action_log_probs.append(action_log_prob)
                obs = next_obs
                if self.render:
                    self.env.render()

            # turn lists into tensors
            states = torch.stack(states)
            actions = torch.stack(actions)
            rewards = torch.stack(rewards)
            action_log_probs = torch.stack(action_log_probs)

            if self.logger is not None and i % self.logger.log_every == 0:
                self.logger.add_scalar(scalar_value=rewards.mean(), tag="Reward/train")

            self.replay_buffer.push(states, actions.detach(), rewards, action_log_probs.detach())
