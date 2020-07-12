from mp_carl.actor_critic_networks import Actor, Critic
from mp_carl.optimizer import SharedAdam


class ParameterServer:
    def __init__(self, G, actor_lr, critic_lr, num_actions, num_obs, lock):
        self.G = G  # number of gradients before updating networks
        self.N = 0  # current number of gradients
        self.lock = lock  # enter monitor to prevent race condition
        self.shared_actor = Actor(num_actions=num_actions, num_obs=num_obs)
        self.shared_actor.share_memory()
        self.shared_critic = Critic(num_actions=num_actions, num_obs=num_obs)
        self.shared_critic.share_memory()
        self.actor_optimizer = SharedAdam(self.shared_actor.parameters(), actor_lr)
        self.critic_optimizer = SharedAdam(self.shared_critic.parameters(), critic_lr)

    def receive_gradients(self, actor, critic):
        self.add_gradients(source_actor=actor, source_critic=critic)
        with self.lock:
            self.N += 1
            if self.N == self.G:
                self.update_gradients()  # todo: this should reset gradients -> check
                self.N = 0

    def add_gradients(self, source_actor, source_critic):
        for a_param, a_source_param in zip(self.shared_actor.parameters(), source_actor.parameters()):
            a_param.grad += a_source_param.grad / self.G
            # a_param.grad_ += a_source_param.grad / self.G # todo grad_ ???

        for c_param, c_source_param in zip(self.shared_critic.parameters(), source_critic.parameters()):
            c_param.grad += c_source_param.grad / self.G

    def update_gradients(self):
        self.actor_optimizer.step()
        self.critic_optimizer.step()
