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
                recursiv=False):
        """
        Reimplementation of Retrace ((http://arxiv.org/abs/1606.02647)) loss from
        Deepmind (https://github.com/deepmind/trfl/blob/master/trfl/retrace_ops.py?l=45) in PyTorch.

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
            Loss

        """
        if recursiv:
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

        B = Q.shape[0]  # batch size
        trajectory_length = Q.shape[1]
        Q_ret = torch.zeros(B, trajectory_length)
        for i in range(trajectory_length):
            for j in range(i, trajectory_length):
                c_k = self.calc_retrace_weights(target_policy_probs, behaviour_policy_probs)
                delta = expected_target_Q[i] - target_Q[:, j]
                Q_ret[:, i] = (gamma ** (j - i) * torch.prod(c_k[:, i:j])) * (rewards[:, j] + delta)

        return F.mse_loss(Q, Q_ret)

    def retrace_recursive(self,
                          Q,
                          expected_target_Q,
                          target_Q,
                          rewards,
                          target_policy_probs,
                          behaviour_policy_probs,
                          gamma=0.99):
        B = Q.shape[0]
        # We have Q, target_Q, rewards
        r_t = rewards[:, :-1]
        Q_t = Q[:, :-1]
        target_Q_t = target_Q[:, :-1]
        expected_Q_next_t = expected_target_Q[:, 1:]

        c_next_t = self.calc_retrace_weights(target_policy_probs, behaviour_policy_probs)[:, 1:]

        delta = r_t + gamma * expected_Q_next_t - target_Q_t
        delta_rev = self.reverse_sequence(delta, B)
        decay = gamma * c_next_t

        decay_prod_rev = self.reverse_sequence(torch.cumprod(decay, dim=1), B)
        target_rev = torch.cumsum(delta_rev * decay_prod_rev, dim=1) / decay_prod_rev
        target = self.reverse_sequence(target_rev, B)

        return F.mse_loss(target, Q_t)

    @staticmethod
    def calc_retrace_weights(target_policy_probs, behaviour_policy_probs):
        return (target_policy_probs / behaviour_policy_probs.clamp(min=1e-10)).clamp(max=1)
        # return torch.ones_like(target_policy_probs)

    @staticmethod
    def reverse_sequence(sequence, num_sequences, dim=0):
        sequence = sequence.unsqueeze(2)
        for i in range(num_sequences):
            sequence[i, :] = sequence[i, :].flip(dims=[dim])
        return sequence.squeeze(-1)
