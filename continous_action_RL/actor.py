import torch


class Actor:
    def __init__(self,
                 env,
                 num_trajectories,
                 actor_network,
                 replay_buffer,
                 render=False):

        self.env = env
        self.num_trajectories = num_trajectories
        self.actor_network = actor_network
        self.replay_buffer = replay_buffer
        self.render = render

    def collect_trajectories(self):
        for i in range(self.num_trajectories):
            states, actions, rewards, action_probs = [], [], [], []

            obs = torch.tensor(self.env.reset(), dtype=torch.float)
            done = False
            while not done:
                action = self.actor_network.sample(obs)
                action_prob = self.actor_network.action_prob(action=action, observation=obs)
                action = torch.tensor(3)
                next_obs, reward, done, _ = self.env.step([action.item()])
                next_obs, reward = torch.tensor(next_obs, dtype=torch.float), torch.tensor(reward, dtype=torch.float)
                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                action_probs.append(action_prob)
                obs = next_obs
                if self.render:
                    self.env.render()

            # turn lists into tensors
            states = torch.stack(states)
            actions = torch.stack(actions)
            rewards = torch.stack(rewards)
            action_probs = torch.stack(action_probs)

            self.replay_buffer.push(states, actions, rewards, action_probs)
