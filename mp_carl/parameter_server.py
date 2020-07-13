from mp_carl.actor_critic_networks import Actor, Critic
from mp_carl.optimizer import SharedAdam
import torch


class ParameterServer:
    def __init__(self, G, actor_lr, critic_lr, num_actions, num_obs, lock, arg_parser):

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.G = G  # number of gradients before updating networks
        self.N = torch.tensor(0)  # current number of gradients
        self.N.share_memory_()
        self.lock = lock  # enter monitor to prevent race condition
        self.shared_actor = Actor(num_actions=num_actions,
                                  num_obs=num_obs,
                                  mean_scale=arg_parser.action_mean_scale,
                                  std_low=arg_parser.action_std_low,
                                  std_high=arg_parser.action_std_high).to(device)
        self.shared_actor.share_memory()
        self.shared_critic = Critic(num_actions=num_actions, num_obs=num_obs).to(device)
        self.shared_critic.share_memory()
        self.actor_optimizer = SharedAdam(self.shared_actor.parameters(), actor_lr)
        self.critic_optimizer = SharedAdam(self.shared_critic.parameters(), critic_lr)

        self.init_grad()

    def receive_gradients(self, actor, critic):
        with self.lock:
            self.add_gradients(source_actor=actor, source_critic=critic)
            self.N += 1
            if self.N >= self.G:
                self.update_gradients()  # todo: this should reset gradients -> check
                # Reset to 0. Ugly but otherwise not working because position will not be in shared mem if new assigned.
                self.N -= self.N

    def add_gradients(self, source_actor, source_critic):
        # print("optim", self.actor_optimizer.param_groups[0]['params'][-1].grad)
        # a = list(self.shared_actor.parameters())[-1].grad
        # b = list(source_actor.parameters())[-1].grad/self.G
        # print(a)
        # print(b)
        # print(a+b)
        # print("-"*30)

        for a_param, a_source_param in zip(self.shared_actor.parameters(), source_actor.parameters()):
            a_param.grad += a_source_param.grad / self.G
            # a_param.grad_ += a_source_param.grad / self.G # todo grad_ ???

        for c_param, c_source_param in zip(self.shared_critic.parameters(), source_critic.parameters()):
            c_param.grad += c_source_param.grad / self.G

    def update_gradients(self):
        # print("update step")
        # print("optim", self.actor_optimizer.param_groups[0]['params'][-1].grad)
        # print(list(self.shared_actor.parameters())[-1])
        self.actor_optimizer.step()
        # print(list(self.shared_actor.parameters())[-1])
        # print("-"*30)
        self.critic_optimizer.step()
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

    def init_grad(self):
        for a_param in self.shared_actor.parameters():
            a_param.grad = torch.zeros_like(a_param.data, dtype=torch.float32)
            a_param.grad.share_memory_()

        for c_param in self.shared_critic.parameters():
            c_param.grad = torch.zeros_like(c_param.data, dtype=torch.float32)
            c_param.grad.share_memory_()
