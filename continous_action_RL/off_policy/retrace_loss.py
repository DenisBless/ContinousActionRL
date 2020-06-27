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

        if recursive:
            return self.retrace_recursive(Q=Q,
                                          expected_target_Q=expected_target_Q,
                                          target_Q=target_Q,
                                          rewards=rewards,
                                          target_policy_probs=target_policy_probs,
                                          behaviour_policy_probs=behaviour_policy_probs,
                                          gamma=gamma)

        else:
            return self.retrace_iterative(Q=Q,
                                          expected_target_Q=expected_target_Q,
                                          target_Q=target_Q,
                                          rewards=rewards,
                                          target_policy_probs=target_policy_probs,
                                          behaviour_policy_probs=behaviour_policy_probs,
                                          gamma=gamma)

    def retrace_iterative(self,
                          Q,
                          expected_target_Q,
                          target_Q,
                          rewards,
                          target_policy_probs,
                          behaviour_policy_probs,
                          gamma=0.99):
        """
        For information on the parameters see class docs.

        Computes the retrace loss according to
        L = 𝔼_τ[(Q - Q_ret)^2]
        Q_ret = ∑_j=i γ^(j-i) * (Π_k=i^j c_k) * (r(s_j, a_j) + δ(s_i, s_j))
        δ(s_i, s_j) = 𝔼_π_target [Q(s_i,•)] - Q_π_target(s_j,a_j)
        c_k = min(1, π_target(a_k|s_k) / b(a_k|s_k))

        with trajectory τ = {(s_0, a_0, r_0),..,(s_k, a_k, r_k)}

        Returns:
            Scalar critic loss value.
        """

        B = Q.shape[0]  # batch size
        trajectory_length = Q.shape[1]
        Q_ret = torch.zeros(B, trajectory_length).to(self.device)
        for i in range(trajectory_length - 1):
            for j in range(i, trajectory_length - 1):
                c_k = self.calc_retrace_weights(target_policy_probs, behaviour_policy_probs)
                # delta = gamma * expected_target_Q[:, i] - target_Q[:, j]
                delta = gamma * expected_target_Q[:, j + 1] - target_Q[:, j]
                Q_ret[:, i] += (gamma ** (j - i) * torch.prod(c_k[:, i:j])) * (rewards[:, j] + delta)

        return F.mse_loss(Q, Q_ret)

    def retrace_recursive(self,
                          Q,
                          expected_target_Q,
                          target_Q,
                          rewards,
                          target_policy_probs,
                          behaviour_policy_probs,
                          gamma=0.99):
        """
        For information on the parameters see class docs.

        Computes the retrace loss recursively according to
        L = 𝔼_τ[(Q_t - Q_ret_t)^2]
        Q_ret_t = r_t + γ * (𝔼_π_target [Q(s_t+1,•)] + c_t+1 * Q_π_target(s_t+1,a_t+1)) + γ * c_t+1 * Q_ret_t+1

        with trajectory τ = {(s_0, a_0, r_0),..,(s_k, a_k, r_k)}

        Returns:
            Scalar critic loss value.

        """

        Q_t = Q[:, :-1]
        r_t = rewards[:, :-1]

        with torch.no_grad():
            # We don't want gradients from computing Q_ret, since:
            # ∇φ (Q - Q_ret)^2 ∝ (Q - Q_ret) * ∇φ Q
            target_Q_next_t = target_Q[:, 1:]
            expected_Q_next_t = expected_target_Q[:, 1:]
            c_next_t = self.calc_retrace_weights(target_policy_probs, behaviour_policy_probs)[:, 1:]

            delta = r_t + gamma * expected_Q_next_t - target_Q_next_t
            decay = torch.cumprod(gamma * c_next_t, dim=1)
            target = self.cumsum_reversed(delta * decay) / decay.clamp(min=1e-8)

        return F.mse_loss(Q_t, target)

    @staticmethod
    def cumsum_reversed(sequence):
        """
        Calculates the reversed cumulative sum. I.e. rcs_t =  sum(sequence[:, t:l]) where l is the sequence length.
        Reversion is performed along the axis 1.

        Args:
            sequence: Sequence to operate on

        Returns:
            Reversed cumulative sum
        """
        return torch.flip(torch.cumsum(torch.flip(sequence, [1]), 1), [1])

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

        return (target_policy_probs / behaviour_policy_probs.clamp(min=1e-10)).clamp(max=1)

    # def retrace_recursiveOLD(self,
    #                          Q,
    #                          expected_target_Q,
    #                          target_Q,
    #                          rewards,
    #                          target_policy_probs,
    #                          behaviour_policy_probs,
    #                          gamma=0.99):
    #
    #     B = Q.shape[0]
    #     # We have Q, target_Q, rewards
    #     r_t = rewards[:, :-1]
    #     Q_t = Q[:, :-1]
    #
    #     target_Q_next_t = target_Q[:, 1:]
    #     expected_Q_next_t = expected_target_Q[:, 1:]
    #     c_next_t = self.calc_retrace_weights(target_policy_probs, behaviour_policy_probs)[:, 1:]
    #
    #     delta = r_t + gamma * expected_Q_next_t - target_Q_next_t
    #     delta_rev = self.reverse_sequence(delta, B)
    #     decay = gamma * c_next_t
    #
    #     decay_prod_rev = self.reverse_sequence(torch.cumprod(decay, dim=1), B)
    #     target_rev = torch.cumsum(delta_rev * decay_prod_rev, dim=1) / decay_prod_rev.clamp(min=1e-8)
    #     target = self.reverse_sequence(target_rev, B)
    #
    #     return F.mse_loss(target, Q_t)
    #
    # @staticmethod
    # def calc_retrace_weights(target_policy_probs, behaviour_policy_probs):
    #     assert target_policy_probs.shape == behaviour_policy_probs.shape, \
    #         "Error, shape mismatch. Shapes: target_policy_probs: " \
    #         + str(target_policy_probs.shape) + " mean: " + str(behaviour_policy_probs.shape)
    #
    #     return (target_policy_probs / behaviour_policy_probs.clamp(min=1e-10)).clamp(max=1)
    #
    # @staticmethod
    # def reverse_sequence(sequence, num_sequences, dim=0):
    #     sequence = sequence.unsqueeze(2)
    #     for i in range(num_sequences):
    #         sequence[i, :] = sequence[i, :].flip(dims=[dim])
    #     return sequence.squeeze(-1)
    #
    # @staticmethod
    # def cumsum_reversed(sequence):
    #     return torch.flip(torch.cumsum(torch.flip(sequence, [1]), 1), [1])
    #
    # @staticmethod
    # def cumprod_reversed(sequence):
    #     return torch.flip(torch.cumprod(sequence, 1), [1])
