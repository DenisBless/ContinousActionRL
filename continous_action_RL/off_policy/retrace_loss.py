import torch
import torch.nn.functional as F


class Retrace(torch.nn.Module):
    def __init__(self, gamma=0.99):
        super(Retrace, self).__init__()
        self.gamma = gamma

    def forward(self,
                Q,
                expected_target_Q,
                target_Q,
                rewards,
                target_policy_probs,
                behaviour_policy_probs):
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

            recursive: If true, uses the recursive equation for retrace otherwise iterative. For more information see
            the docstrings.

        Returns:
            Retrace loss
        """



        """
        For information on the parameters see class docs.

        Computes the retrace loss recursively according to
        L = 𝔼_τ[(Q_t - Q_ret_t)^2]
        Q_ret_t = r_t + γ * (𝔼_π_target [Q(s_t+1,•)] + c_t+1 * Q_π_target(s_t+1,a_t+1)) + γ * c_t+1 * Q_ret_t+1

        with trajectory τ = {(s_0, a_0, r_0),..,(s_k, a_k, r_k)}

        Returns:
            Scalar critic loss value.

        """

        T = Q.shape[1]  # total number of time steps in the trajectory

        Q_t = Q[:, :-1]

        with torch.no_grad():
            # We don't want gradients from computing Q_ret, since:
            # ∇φ (Q - Q_ret)^2 ∝ (Q - Q_ret) * ∇φ Q
            log_importance_weights = (target_policy_probs - behaviour_policy_probs.unsqueeze(-1))
            importance_weights = (torch.exp(torch.clamp(log_importance_weights, max=0)))

            Q_ret = torch.zeros_like(Q, dtype=torch.float)  # (B,T)
            Q_ret[:, -1, :] = target_Q[:, -1, :]

            for j in reversed(range(0, T - 1)):
                Q_ret[:, j, :] = (
                        rewards[:, j, :]
                        + self.gamma * (
                                expected_target_Q[:, j + 1, :]
                                + importance_weights[:, j + 1, :] * (Q_ret[:, j + 1, :] - target_Q[:, j + 1, :])
                        )
                )

        return F.mse_loss(Q_t, Q_ret[:, :-1, :])

