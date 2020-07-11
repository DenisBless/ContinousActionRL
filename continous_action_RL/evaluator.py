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
                 render=False,
                 use_gpu=False):


        self.env = env
        self.actor = actor
        self.critic = critic
        self.save_path = save_path
        self.num_samples = num_trajectories
        self.save_model_every = save_model_every
        self.logger = logger
        self.render = render
        self.use_gpu = use_gpu

        self.num_evals = 0

    def evaluate(self):
        self.num_evals += 1
        self.actor.eval()  # Eval mode: Sets the action variance to zero, disables batch-norm and dropout etc.


        with torch.no_grad():
            for i in range(self.num_samples):
                obs = torch.tensor(self.env.reset(), dtype=torch.float)
                rewards = []
                done = False
                while not done:
                    obs = obs.cuda() if self.use_gpu else obs
                    action, action_log_prob = self.actor.predict(obs, task=0)
                    next_obs, reward, done, _ = self.env.step(action.detach().cpu().numpy())
                    rewards.append(reward)
                    obs = torch.tensor(next_obs, dtype=torch.float)

                    if self.render:
                        self.env.render()

                    if done:
                        if self.logger is None:
                            print("Mean reward: ", np.mean(rewards))
                        else:
                            self.logger.add_scalar("Reward/test", np.mean(rewards))

        self.actor.train()  # Back to train mode

        # Saving the model parameters
        if self.save_path is not None and self.num_evals % self.save_model_every == 0 and self.num_evals > 0:
            torch.save(self.actor.state_dict(), self.save_path + "actor_" + str(self.num_evals))
            torch.save(self.critic.state_dict(), self.save_path + "critic_" + str(self.num_evals))
