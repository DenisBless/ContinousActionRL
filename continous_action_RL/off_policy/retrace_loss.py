import torch
import torch.nn.functional as F


class Retrace(torch.nn.Module):
    def __init__(self):
        super(Retrace, self).__init__()

    def forward(self,
                Q,
                expected_target_Q,
                target_Q,
                rewards,
                target_policy_probs,
                behaviour_policy_probs,
                gamma=0.99,
                recursive=True):
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
            r_t = rewards[:, :-1]
            target_Q_next_t = target_Q[:, 1:]
            expected_Q_next_t = expected_target_Q[:, 1:]
            log_importance_weights = (target_policy_probs - behaviour_policy_probs)
            importance_weights = (torch.exp(torch.clamp(log_importance_weights, max=0)))[:, 1:]
            c_next_t = importance_weights

            Q_ret = torch.zeros_like(Q_t, dtype=torch.float)  # (B,T)
            Q_ret[:, -1] = Q[:, -1]

            for t in reversed(range(1, T - 1)):
                Q_ret[:, t - 1] = r_t[:, t] + gamma * (expected_Q_next_t[:, t] - c_next_t[:, t] * target_Q_next_t[:, t]) \
                                  + gamma * c_next_t[:, t] * Q_ret[:, t]

        return F.mse_loss(Q_t, Q_ret)



    @staticmethod
    def calc_retrace_weights(target_policy_probs, behaviour_policy_probs):
        """
        Calculates the retrace weights (truncated importance weights) c according to:
        c_t = min(1, π_target(a_t|s_t) / b(a_t|s_t)) where:
        π_target: target policy probabilities
        b: behaviour policy probabilities

        Args:
            target_policy_probs: π_target(a_t|s_t)
            behaviour_policy_probs: b(a_t|s_t)

        Returns:
            retrace weights c
        """
        assert target_policy_probs.shape == behaviour_policy_probs.shape, \
            "Error, shape mismatch. Shapes: target_policy_probs: " \
            + str(target_policy_probs.shape) + " mean: " + str(behaviour_policy_probs.shape)

        if target_policy_probs.dim() > 2:
            retrace_weights = (torch.prod(target_policy_probs, dim=-1) / torch.prod(behaviour_policy_probs, dim=-1)).clamp(min=1e-10, max=1)
        else:
            retrace_weights = (target_policy_probs / behaviour_policy_probs).clamp(min=1e-10, max=1)

        assert not torch.isnan(retrace_weights).any(), "Error, a least one NaN value found in retrace weights."
        return retrace_weights

    @staticmethod
    def remove_last_timestep(x):
        return x.narrow(-2, 0, x.shape[-2] - 1)

    @staticmethod
    def remove_first_timestep(x):
        return x.narrow(-2, 1, x.shape[-2] - 1)

