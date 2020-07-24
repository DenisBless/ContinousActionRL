import torch
import torch.nn.functional as F
from torch.multiprocessing import current_process


class Retrace:
    def __init__(self, num_actions, reward_scale):
        super(Retrace, self).__init__()
        self.num_actions = num_actions
        self.reward_scale = reward_scale
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __call__(self,
                 Q,
                 expected_target_Q,
                 target_Q,
                 rewards,
                 target_policy_probs,
                 behaviour_policy_probs,
                 dones,
                 gamma=0.99,
                 logger=None,
                 n_iter=None):
        """
        Implementation of Retrace loss ((http://arxiv.org/abs/1606.02647)) in PyTorch.

        Args:
            Q: State-Action values.
            Torch tensor with shape `[B, (T+1)]`

            expected_target_Q: 𝔼_π Q(s_t,.) (from the fixed critic network)
            Torch tensor with shape `[B, (T+1)]`

            target_Q: State-Action values from target network.
            Torch tensor with shape `[B, (T+1)]`

            rewards: Holds rewards for taking an action in the environment.
            Torch tensor with shape `[B, (T+1)]`

            target_policy_probs: Probability of target policy π(a|s)
            Torch tensor with shape `[B, (T+1)]`

            behaviour_policy_probs: Probability of behaviour policy b(a|s)
            Torch tensor with shape `[B, (T+1)]`

            gamma: Discount factor

        Returns:

            Computes the retrace loss recursively according to
            L = 𝔼_τ[(Q_t - Q_ret_t)^2]
            Q_ret_t = r_t + γ * (𝔼_π_target [Q(s_t+1,•)] + c_t+1 * Q_π_target(s_t+1,a_t+1)) + γ * c_t+1 * Q_ret_t+1

            with trajectory τ = {(s_0, a_0, r_0),..,(s_k, a_k, r_k)}
        """

        # adjust and check dimensions
        Q.squeeze_(dim=-1)
        target_Q.squeeze_(dim=-1)
        expected_target_Q.squeeze_(dim=-1)

        assert Q.shape == target_Q.shape == expected_target_Q.shape == rewards.shape == target_policy_probs.shape == \
               behaviour_policy_probs.shape

        rewards = rewards * self.reward_scale

        Q = Q * (1 - dones)

        with torch.no_grad():
            # We don't want gradients from computing Q_ret, since:
            # ∇φ (Q - Q_ret)^2 ∝ (Q - Q_ret) * ∇φ Q

            c_ret = self.calc_retrace_weights(target_policy_probs, behaviour_policy_probs)

            # Q_ret = torch.zeros_like(Q, device=self.device, dtype=torch.float)  # (B,T)
            # Q_ret = torch.zeros(T)
            # Q_done = torch.zeros_like(Q, device=self.device, dtype=torch.float)  # (B,T)

            if Q.dim() > 1:  # for batch learning
                Q_ret = torch.zeros_like(Q, device=self.device, dtype=torch.float)  # (B,T)
                for i in range(Q.shape[0]):
                    # T = Q.shape[1]  # total number of time steps in the trajectory
                    T = dones[i].cpu().numpy().argmax() + 1
                    # T = 10
                    # Q[i, T-1:] = 0
                    # Q_ret = torch.zeros(T, device=self.device, dtype=torch.float)
                    Q_ret[i, T - 1] = target_Q[i, T - 1]
                    for t in reversed(range(1, T)):
                        Q_ret[i, t - 1] = rewards[i, t - 1] + gamma * c_ret[i, t] * (Q_ret[i, t] - target_Q[i, t]) + \
                                          gamma * expected_target_Q[i, t]

                # # iterative version
                Q_ret_it = torch.zeros_like(Q_ret)
                for i in range(Q.shape[0]):
                    T = dones[i].cpu().numpy().argmax() + 1
                    # Q_ret_it = torch.zeros(T, device=self.device, dtype=torch.float)
                    for t in range(T):
                        Q_ret_it[i, t] = target_Q[i, t]
                        for j in range(t, T - 1):
                            Q_ret_it[i, t] += (gamma ** (j - t)) * c_ret[i, t + 1:j + 1].prod() * \
                                        (rewards[i, j] + gamma * expected_target_Q[i, j + 1] - target_Q[i, j])
            else:
                # T = Q.shape[0]  # total number of time steps in the trajectory
                T = dones.cpu().numpy().argmax() + 1
                Q_ret = torch.zeros(T, device=self.device, dtype=torch.float)
                Q_ret[-1] = target_Q[-1]
                for t in reversed(range(1, T)):
                    Q_ret[t - 1] = rewards[t - 1] + gamma * c_ret[t] * (Q_ret[t] - target_Q[t]) + \
                                   gamma * expected_target_Q[t]

                # # iterative version
                # # Q_ret_it = torch.zeros_like(Q_ret)
                # T = dones.cpu().numpy().argmax() + 1
                # Q_ret_it = torch.zeros(T, device=self.device, dtype=torch.float)
                # for t in range(T):
                #     Q_ret_it[t] = target_Q[t]
                #     for j in range(t, T - 1):
                #         Q_ret_it[t] += (gamma ** (j - t)) * c_ret[t + 1:j + 1].prod() * \
                #                     (rewards[j] + gamma * expected_target_Q[j + 1] - target_Q[j])

            if logger is not None:
                logger.add_histogram(tag="retrace/ratio", values=c_ret, global_step=n_iter)
                logger.add_histogram(tag="retrace/behaviour", values=behaviour_policy_probs, global_step=n_iter)
                logger.add_histogram(tag="retrace/target", values=target_policy_probs, global_step=n_iter)
                logger.add_histogram(tag="retrace/Qret mean", values=Q_ret, global_step=n_iter)

                logger.add_scalar(tag="retrace/Qret-targetQ mean", scalar_value=(Q_ret - target_Q).mean(), global_step=n_iter)
                logger.add_scalar(tag="retrace/Qret-targetQ std", scalar_value=(Q_ret - target_Q).std(), global_step=n_iter)
                logger.add_scalar(tag="retrace/E[targetQ] mean", scalar_value=expected_target_Q.mean(), global_step=n_iter)
                logger.add_scalar(tag="retrace/E[targetQ] std", scalar_value=expected_target_Q.std(), global_step=n_iter)

        return (Q - Q_ret).square().sum().div((1 - dones).sum())
        # return F.mse_loss(Q, Q_ret)

    def calc_retrace_weights(self, target_policy_logprob, behaviour_policy_logprob):
        """
        Calculates the retrace weights (truncated importance weights) c according to:
        c_t = min(1, π_target(a_t|s_t) / b(a_t|s_t)) where:
        π_target: target policy probabilities
        b: behaviour policy probabilities

        Args:
            target_policy_logprob: log π_target(a_t|s_t)
            behaviour_policy_logprob: log b(a_t|s_t)

        Returns:
            retrace weights c
        """
        assert target_policy_logprob.shape == behaviour_policy_logprob.shape, \
            "Error, shape mismatch. Shapes: target_policy_logprob: " \
            + str(target_policy_logprob.shape) + " mean: " + str(behaviour_policy_logprob.shape)

        log_retrace_weights = (target_policy_logprob - behaviour_policy_logprob).clamp(max=0)

        assert not torch.isnan(log_retrace_weights).any(), "Error, a least one NaN value found in retrace weights."
        # return log_retrace_weights.exp()
        return log_retrace_weights.exp()


class ActorLoss:
    def __init__(self, alpha=0):
        """
        Loss function for the actor.
        Args:
            alpha: entropy regularization parameter.
        """
        self.alpha = alpha

    def __call__(self, Q, action_log_prob, dones):
        """
        Computes the loss of the actor according to
        L = 𝔼_π [Q(a,s) - α log(π(a|s)]
        Args:
            Q: Q(a,s)
            action_log_prob: log(π(a|s)

        Returns:
            Scalar actor loss value
        """
        assert Q.dim() == action_log_prob.dim()
        valid_Q = Q * (1 - dones)
        valid_log_prob = action_log_prob * (1 - dones)
        return (self.alpha * valid_log_prob - valid_Q).sum().div((1 - dones).sum())
        # return - Q.mean()

    @staticmethod
    def kl_divergence(old_mean, old_std, mean, std):
        """
        Computes:
        D_KL(π_old(a|s)||π(a|s)) = ∑_i D_KL(π_old(a_i|s)||π(a_i|s))
        where π(a_i|s) = N(a|μ, σ^2)
        and D_KL(π_old(a_i|s)||π(a_i|s)) = 1/2 * (log(σ^2/σ_old^2) + [σ_old^2 + (μ_old - μ)^2]/σ^2 -1)
        """
        t1 = torch.log(torch.pow(std, 2)) - torch.log(torch.pow(old_std, 2))
        t2 = (torch.pow(old_std, 2) + torch.pow(old_mean - mean, 2)) / torch.pow(std, 2)
        return torch.sum(t1 + t2 - 1)
