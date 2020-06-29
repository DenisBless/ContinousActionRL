import torch
import numpy as np


class Evaluator:
    def __init__(self,
                 env,
                 actor,
                 critic,
                 save_path=None,
                 num_trajectories=10,
                 save_model_every=10,
                 logger=None,
                 render=False):

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
        obs = obs.to(self.device)
        with torch.no_grad():
            for i in range(self.num_samples):
                rewards = []
                done = False
                while not done:
                    mean, std = self.actor.forward(observation=obs)
                    mean = mean.to(self.device)
                    std = std.to(self.device)
                    action, action_log_prob = self.actor.action_sample(mean, std)
                    action = action.to(self.device)
                    next_obs, reward, done, _ = self.env.step([action.item()])
                    rewards.append(reward)
                    obs = torch.tensor(next_obs, dtype=torch.float).to(self.device)

                    if self.render:
                        self.env.render()

                    if done:
                        obs = torch.tensor(self.env.reset(), dtype=torch.float).to(self.device)
                        if self.logger is None:
                            print("Mean reward: ", np.mean(rewards))
                        else:
                            self.logger.add_scalar("Mean reward", np.mean(rewards))

        self.actor.train()  # Back to train mode

        # Saving the model parameters
        if self.save_path is not None and self.num_evals % self.save_model_every == 0 and self.num_evals > 0:
            torch.save(self.actor.state_dict(), self.save_path + "actor_" + str(self.num_evals))
            torch.save(self.critic.state_dict(), self.save_path + "critic_" + str(self.num_evals))
