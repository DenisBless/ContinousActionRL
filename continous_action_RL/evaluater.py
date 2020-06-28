import torch
import numpy as np


class Evaluator:
    def __init__(self,
                 env,
                 actor,
                 critic,
                 save_path,
                 num_trajectories=10,
                 save_model_every=10,
                 logger=None,
                 render=False):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.save_path = save_path
        self.num_samples = num_trajectories
        self.save_model_every = save_model_every
        self.logger = logger
        self.render = render

        self.num_evals = 0

    def evaluate(self):
        self.num_evals += 1
        self.actor.eval()  # Eval mode: Sets the action variance to zero, disables batch-norm and dropout etc.

        obs = torch.tensor(self.env.reset(), dtype=torch.float)
        rewards = []
        with torch.no_grad():
            for i in range(self.num_samples):
                mean, std = self.actor.forward(observation=obs)

                action, action_log_prob = self.actor.action_sample(mean, torch.zeros_like(mean))
                next_obs, reward, done, _ = self.env.step([action.item()])
                rewards.append(reward)
                obs = torch.tensor(next_obs, dtype=torch.float)

                if self.render:
                    self.env.render()

                if done:
                    obs = torch.tensor(self.env.reset(), dtype=torch.float)
                    if self.logger is not None:
                        self.logger.add_scalar("Mean reward", np.mean(reward))

        self.actor.train()  # Back to train mode

        # Saving the model parameters
        if self.num_evals % self.save_model_every == 0 and self.num_evals > 0:
            torch.save(self.actor.state_dict(), self.save_path + "actor_" + str(self.num_evals))
            torch.save(self.critic.state_dict(), self.save_path + "critic_" + str(self.num_evals))
