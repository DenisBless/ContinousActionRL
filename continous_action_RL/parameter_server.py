import torch
from continous_action_RL.optimizer import SharedAdam


class ParameterServer:
    def __init__(self, G, actor, critic, lock):
        self.G = G  # number of gradients before updating networks
        self.N = 0  # current number of gradients
        self.lock = lock  # enter monitor to prevent race condition
        self.actor = ...
        self.critic = ...
        self.actor_grad_storage = ...
        self.critic_grad_storage = ...
        self.actor_optimizer = ...
        self.critic_optimizer = ...

    def receive_gradients(self, actor_grad, critic_grad):
        self.actor_grad_storage.put(actor_grad)
        self.critic_grad_storage.put(critic_grad)
        with self.lock:
            self.N += 1
            if self.N == self.G:
                self.update_gradients()  # todo: this should reset gradients -> check
                self.N = 0

    def add_gradients(self, source_actor, source_critic):
        for a_param, a_source_param in zip(self.actor.parameters(), source_actor.parameters()):
            a_param.grad_ += a_source_param / self.G

        for c_param, c_source_param in zip(self.critic.parameters(), source_critic.parameters()):
            c_param.grad_ += c_source_param / self.G

    def update_gradients(self):
        self.actor_optimizer.step()
        self.critic_optimizer.step()
