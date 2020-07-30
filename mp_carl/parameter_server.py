from torch import Tensor

from common.actor_critic_models import Actor, Critic
from mp_carl.optimizer import SharedAdam
import torch
import numpy as np
from typing import Union, List
from torch.multiprocessing import Condition
from functools import reduce


class ParameterServer:
    """
    Shared parameter server. Let g be the gradient of the shared network, g' the incoming gradient of a worker and G
    the fixed number of gradients until a update to the shared network parameters p is performed. The procedure is
    as follows:

    repeat until convergence:

        while i < G do:
            g += g' / G
            i++

        p -= η * g

    """

    def __init__(self, actor_lr: float, critic_lr: float, num_actions: int, num_obs: int,
                 worker_cv: Condition, server_cv: Condition, arg_parser):

        self.G = arg_parser.num_workers * arg_parser.num_grads  # number of gradients before updating networks
        self.N = torch.tensor(0)  # current number of gradients
        self.N.share_memory_()
        self.worker_cv = worker_cv
        self.server_cv = server_cv

        self.shared_actor = Actor(num_actions=num_actions, num_obs=num_obs, log_std_init=np.log(arg_parser.init_std))
        self.shared_actor.share_memory()

        self.shared_critic = Critic(num_actions=num_actions, num_obs=num_obs)
        self.shared_critic.share_memory()

        self.actor_grads, self.critic_grads = self.init_grad()

        self.actor_optimizer = SharedAdam(self.shared_actor.parameters(), actor_lr)
        self.actor_optimizer.share_memory()
        self.critic_optimizer = SharedAdam(self.shared_critic.parameters(), critic_lr)
        self.critic_optimizer.share_memory()

        self.global_gradient_norm = arg_parser.global_gradient_norm

    def run(self) -> None:
        print("Parameter server started.")
        while True:
            with self.server_cv:
                self.server_cv.wait_for(lambda: self.N == self.G)
                self.N.zero_()
                self.update_params()
                self.worker_cv.notify_all()

    def receive_gradients(self, actor_grads, critic_grads) -> None:
        """
        Receive gradients by the workers. Update the parameters of the shared networks after G gradients were
        accumulated.

        Args:
            actor: The actor network

        Returns:
            No return value
        """
        with self.worker_cv:
            self.add_gradients(actor_grads=actor_grads, critic_grads=critic_grads)
            self.N += 1

    def add_gradients(self, actor_grads, critic_grads) -> None:
        for shared_ag, ag in zip(self.actor_grads, actor_grads):
            shared_ag += ag / self.G
        for shared_cg, cg in zip(self.critic_grads, critic_grads):
            shared_cg += cg / self.G

    def update_params(self) -> None:
        """
        Update the parameter of the shared actor and critic networks.

        Returns:
            No return value
        """
        # print(self.critic_grads[-1])

        for a_param, a_grad in zip(self.shared_actor.parameters(), self.actor_grads):
            a_param.grad = a_grad
        for c_param, c_grad in zip(self.shared_critic.parameters(), self.critic_grads):
            c_param.grad = c_grad
        # print("before")
        # print(self.shared_critic.grad_norm)
        # self.print_grad_norm(self.shared_actor)
        if self.global_gradient_norm != -1:
            torch.nn.utils.clip_grad_norm_(self.shared_actor.parameters(), self.global_gradient_norm)
            torch.nn.utils.clip_grad_norm_(self.shared_critic.parameters(), self.global_gradient_norm)
        # print("after")
        # print(self.shared_critic.grad_norm)
        # assert (self.shared_critic.grad_norm <= self.global_gradient_norm).item()
        # assert (self.shared_actor.grad_norm <= self.global_gradient_norm).item()
        # self.print_grad_norm(self.shared_actor)
        # print("bef", list(self.shared_actor.parameters())[0])
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        # print("aft", list(self.shared_actor.parameters())[0])

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.zero_grads()

        assert not self.shared_critic.grad_norm
        assert not self.shared_actor.grad_norm

    def init_grad(self) -> Union[List, List]:
        actor_grads = [torch.zeros_like(x, requires_grad=False) for x in list(self.shared_actor.parameters())]
        critic_grads = [torch.zeros_like(x, requires_grad=False) for x in list(self.shared_critic.parameters())]
        for a, c in zip(actor_grads, critic_grads):
            a.share_memory_()
            c.share_memory_()
        return [actor_grads, critic_grads]

    def zero_grads(self) -> None:
        for a, c in zip(self.actor_grads, self.critic_grads):
            a.zero_()
            c.zero_()

    def get_grad_norm(self) -> Union[float, float]:
        ag_norm = reduce(lambda x, y: torch.norm(x) + torch.norm(y), self.actor_grads).item()
        cg_norm = reduce(lambda x, y: torch.norm(x) + torch.norm(y), self.critic_grads).item()
        return [ag_norm, cg_norm]

    def get_param_norm(self):
        ap_norm = reduce(lambda x, y: torch.norm(x) + torch.norm(y), list(self.shared_actor.parameters())).item()
        cp_norm = reduce(lambda x, y: torch.norm(x) + torch.norm(y), list(self.shared_critic.parameters())).item()
        return [ap_norm, cp_norm]
