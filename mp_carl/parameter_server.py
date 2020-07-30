from torch import Tensor

from mp_carl.actor_critic_models import Actor, Critic
from mp_carl.optimizer import SharedAdam
import torch
from typing import Union, List
from torch.multiprocessing import Condition


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

    def __init__(self, actor_lr: float, critic_lr: float, num_actions: int, num_obs: int, cv: Condition,
                 arg_parser):

        self.G = arg_parser.num_workers * arg_parser.num_grads  # number of gradients before updating networks
        self.N = torch.tensor(0)  # current number of gradients
        self.N.share_memory_()
        self.cv = cv  # enter monitor to prevent race condition

        self.shared_actor = Actor(num_actions=num_actions, num_obs=num_obs)
        self.shared_actor.share_memory()

        self.shared_critic = Critic(num_actions=num_actions, num_obs=num_obs)
        self.shared_critic.share_memory()

        self.actor_grads, self.critic_grads = self.init_grad()

        self.actor_optimizer = SharedAdam(self.shared_actor.parameters(), actor_lr)
        self.actor_optimizer.share_memory()
        self.critic_optimizer = SharedAdam(self.shared_critic.parameters(), critic_lr)
        self.critic_optimizer.share_memory()

        self.global_gradient_norm = arg_parser.global_gradient_norm

        # self.init_grad()

    # def receive_critic_gradients(self, critic: torch.nn.Module):
    #     self.add_gradients(net=self.shared_critic, source_net=critic)

    # def receive_actor_gradients(self, actor: torch.nn.Module) -> None:
    #     """
    #     Receive gradients by the workers. Update the parameters of the shared networks after G gradients were
    #     accumulated.
    #
    #     Args:
    #         actor: The actor network
    #
    #     Returns:
    #         No return value
    #     """
    #     with self.cv:
    #         if self.N == self.G:
    #             self.N.zero_()
    #         self.add_gradients(net=self.shared_actor, source_net=actor)
    #         self.N += 1
    #         print(self.shared_critic.grad_norm)
    #         assert self.N.is_shared()
    #         if self.N >= self.G:
    #             self.update_params()
    #             self.cv.notify_all()

    def receive_gradients(self, actor_grads, critic_grads) -> None:
        """
        Receive gradients by the workers. Update the parameters of the shared networks after G gradients were
        accumulated.

        Args:
            actor: The actor network

        Returns:
            No return value
        """
        with self.cv:
            if self.N == self.G:
                self.N.zero_()
            self.add_gradients(actor_grads=actor_grads, critic_grads=critic_grads)
            print(self.actor_grads[0])
            self.N += 1
            assert self.N.is_shared()
            if self.N >= self.G:
                self.update_params()
                self.cv.notify_all()

    # def add_gradients(self, net: torch.nn.Module, source_net: torch.nn.Module) -> None:
    #     """
    #     Add the gradients from the workers to
    #     Args:
    #         source_net: Worker which delivers gradients
    #         net: Receives gradients
    #
    #     Returns:
    #         No return value
    #     """
    #     # for param, source_param in zip(net.parameters(), source_net.parameters()):
    #     for param, source_param in zip(self.shared_critic.parameters(), source_net.parameters()):
    #         param.grad.add_(source_param.grad)# / self.G todo!!!
    #         assert param.grad.is_shared()

    def add_gradients(self, actor_grads, critic_grads):
        for shared_ag, ag in zip(self.actor_grads, actor_grads):
            shared_ag += ag #/ self.G
        for shared_cg, cg in zip(self.critic_grads, critic_grads):
            shared_cg += cg #/ self.G


    def update_params(self) -> None:
        """
        Update the parameter of the shared actor and critic networks.

        Returns:
            No return value
        """

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
        assert (self.shared_critic.grad_norm <= self.global_gradient_norm).item(), print(self.shared_critic.grad_norm, self.global_gradient_norm)
        assert (self.shared_actor.grad_norm <= self.global_gradient_norm).item(), print(self.shared_actor.grad_norm, self.global_gradient_norm)
        # self.print_grad_norm(self.shared_actor)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        assert not self.shared_critic.grad_norm
        assert not self.shared_actor.grad_norm

    # def init_grad(self) -> None:
    #     """
    #     Load the gradients of the shared networks in shared memory.
    #
    #     Returns:
    #         No return value
    #     """
    #     for a_param in self.shared_actor.parameters():
    #         a_param.grad = torch.zeros_like(a_param.data, dtype=torch.float32)
    #         a_param.grad.share_memory_()
    #
    #     for c_param in self.shared_critic.parameters():
    #         c_param.grad = torch.zeros_like(c_param.data, dtype=torch.float32)
    #         c_param.grad.share_memory_()

    def init_grad(self) -> Union[List, List]:
        actor_grads = [torch.zeros_like(x, requires_grad=False)for x in list(self.shared_actor.parameters())]
        critic_grads: List[Tensor] = [torch.zeros_like(x, requires_grad=False)for x in list(self.shared_critic.parameters())]
        for a, c in zip(actor_grads, critic_grads):
            a.share_memory_()
            c.share_memory_()
        return [actor_grads, critic_grads]

    # Only for test purposes
    @staticmethod
    def print_grad_norm(net):
        n = 0
        for i in range(len(list(net.parameters()))):
            n += torch.norm(list(net.parameters())[i].grad)
        print(n)
