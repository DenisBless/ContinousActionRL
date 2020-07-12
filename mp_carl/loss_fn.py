import torch
import torch.nn.functional as F


class Retrace(torch.nn.Module):
    def __init__(self):
        super(Retrace, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def forward(self,
                Q,
                expected_target_Q,
                target_Q,
                rewards,
                target_policy_probs,
                behaviour_policy_probs,
                gamma=0.99,
                logger=None):
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

        T = Q.shape[0]  # total number of time steps in the trajectory

        Q_t = Q[:-1]

        with torch.no_grad():
            # We don't want gradients from computing Q_ret, since:
            # ∇φ (Q - Q_ret)^2 ∝ (Q - Q_ret) * ∇φ Q
            r_t = rewards[:-1]
            target_Q_next_t = target_Q[1:]
            expected_Q_next_t = expected_target_Q[1:]

            c_next_t = self.calc_retrace_weights(target_policy_probs, behaviour_policy_probs)[1:]
            # print(c_next_t.mean(), c_next_t.std())
            if logger is not None:
                logger.add_histogram(tag="retrace/ratio", values=c_next_t)
                logger.add_histogram(tag="retrace/behaviour", values=torch.exp(behaviour_policy_probs))
                logger.add_histogram(tag="retrace/target", values=torch.exp(target_policy_probs))

            Q_ret = torch.zeros_like(Q_t, device=self.device, dtype=torch.float)  # (B,T)
            Q_ret[-1] = target_Q_next_t[-1]

            for t in reversed(range(1, T - 1)):
                Q_ret[t - 1] = r_t[t] + gamma * (expected_Q_next_t[t] - c_next_t[t] * target_Q_next_t[t]) \
                                  + gamma * c_next_t[t] * Q_ret[t]

        return F.mse_loss(Q_t, Q_ret)

    @staticmethod
    def calc_retrace_weights(target_policy_logprob, behaviour_policy_logprob):
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

        num_actions = target_policy_logprob.shape[-1]

        if target_policy_logprob.dim() > 2:
            retrace_weights = (
                    torch.sum(target_policy_logprob, dim=-1) - torch.sum(behaviour_policy_logprob, dim=-1)).clamp(max=0)
        else:
            retrace_weights = (target_policy_logprob - behaviour_policy_logprob).clamp(max=0)

        assert not torch.isnan(retrace_weights).any(), "Error, a least one NaN value found in retrace weights."
        return torch.pow(torch.exp(retrace_weights), 1 / num_actions)


class ActorLoss(torch.nn.Module):
    def __init__(self,
                 alpha=0):
        """
        Loss function for the actor.
        Args:
            alpha: entropy regularization parameter.
        """
        super(ActorLoss, self).__init__()
        self.alpha = alpha

    def forward(self, Q, action_log_prob):
        """
        Computes the loss of the actor according to
        L = 𝔼_π [Q(a,s) - α log(π(a|s)]
        Args:
            Q: Q(a,s)
            action_log_prob: log(π(a|s)

        Returns:
            Scalar actor loss value
        """
        return - (Q.unsqueeze(-1) - self.alpha * action_log_prob).mean()

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
