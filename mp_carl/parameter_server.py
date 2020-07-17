from mp_carl.actor_critic_models import Actor, Critic
from mp_carl.optimizer import SharedAdam
import torch
from torch.multiprocessing import current_process


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
    def __init__(self, G: int, actor_lr: float, critic_lr: float, num_actions: int, num_obs: int, lock, arg_parser):

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.delivery = torch.zeros(arg_parser.num_worker)
        self.delivery.share_memory_()

        self.G = G  # number of gradients before updating networks
        self.N = torch.tensor(0)  # current number of gradients
        self.N.share_memory_()
        self.lock = lock  # enter monitor to prevent race condition

        self.shared_actor = Actor(num_actions=num_actions, num_obs=num_obs).to(device)
        self.shared_actor.share_memory()

        self.shared_critic = Critic(num_actions=num_actions, num_obs=num_obs).to(device)
        self.shared_critic.share_memory()

        self.actor_optimizer = SharedAdam(self.shared_actor.parameters(), actor_lr)
        self.actor_optimizer.share_memory()
        self.critic_optimizer = SharedAdam(self.shared_critic.parameters(), critic_lr)
        self.critic_optimizer.share_memory()

        self.global_gradient_norm = arg_parser.global_gradient_norm

        self.init_grad()

    def receive_gradients(self, actor: torch.nn.Module, critic: torch.nn.Module) -> None:
        """
        Receive gradients by the workers. Update the parameters of the shared networks after G gradients were
        accumulated.

        Args:
            actor: The actor network
            critic: The critic network

        Returns:
            No return value
        """
        with self.lock:
            print(self.delivery)
            self.add_gradients(source_actor=actor, source_critic=critic)
            self.delivery[current_process()._identity[0] - 1] += 1
            self.N += 1
            assert self.N.is_shared()  # todo remove when everything is working
            assert self.delivery.is_shared()
            if self.N >= self.G:
                self.update_params()
                # Reset to 0. Ugly but otherwise not working because position will not be in shared mem if new assigned.
                self.N -= self.N

    def add_gradients(self, source_actor: torch.nn.Module, source_critic: torch.nn.Module) -> None:
        """
        Add the gradients from the workers to
        Args:
            source_actor: Worker (actor) which delivers gradients
            source_critic: Worker (critic) which delivers gradients

        Returns:
            No return value
        """
        for a_param, a_source_param in zip(self.shared_actor.parameters(), source_actor.parameters()):
            a_param.grad += a_source_param.grad / self.G
            assert a_param.grad.is_shared()

        for c_param, c_source_param in zip(self.shared_critic.parameters(), source_critic.parameters()):
            c_param.grad += c_source_param.grad / self.G
            assert c_param.grad.is_shared()

    def update_params(self) -> None:
        """
        Update the parameter of the shared actor and critic networks.

        Returns:
            No return value
        """
        # print("before")
        # self.print_grad_norm(self.shared_critic)
        # self.print_grad_norm(self.shared_actor)
        if self.global_gradient_norm != -1:
            torch.nn.utils.clip_grad_norm_(self.shared_actor.parameters(), self.global_gradient_norm)
            torch.nn.utils.clip_grad_norm_(self.shared_critic.parameters(), self.global_gradient_norm)
        # print("after")
        # self.print_grad_norm(self.shared_critic)
        # self.print_grad_norm(self.shared_actor)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

    def init_grad(self) -> None:
        """
        Load the gradients of the shared networks in shared memory.

        Returns:
            No return value
        """
        for a_param in self.shared_actor.parameters():
            a_param.grad = torch.zeros_like(a_param.data, dtype=torch.float32)
            a_param.grad.share_memory_()

        for c_param in self.shared_critic.parameters():
            c_param.grad = torch.zeros_like(c_param.data, dtype=torch.float32)
            c_param.grad.share_memory_()

    # Only for test purposes
    @staticmethod
    def print_grad_norm(net):
        n = 0
        for i in range(len(list(net.parameters()))):
            n += torch.norm(list(net.parameters())[i].grad)
        print(n)


