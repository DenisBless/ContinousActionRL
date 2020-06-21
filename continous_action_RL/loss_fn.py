import torch


class ActorLoss(torch.nn.Module):
    def __init__(self):
        super(ActorLoss, self).__init__()

    def forward(self, Q):
        return - torch.mean(Q)


class Retrace(torch.nn.Module):
    def __init__(self):
        super(Retrace, self).__init__()

    def _retrace_weights(self, pi_probs, b_probs):
        # return (pi_probs / b_probs).clamp(max=1)
        return 1

    def forward(self,
                discount_factor,
                Q,
                targnet_Q,
                rewards,
                target_policy_probs,
                behaviour_policy_probs,
                minibatch_size):
        num_timesteps = Q.shape[1]
        t = range(1, num_timesteps)
        t_prev = range(0, num_timesteps - 1)

        Q_t_prev = Q[:, :-1]
        r_t = rewards[:, :-1]
        gamma = torch.ones_like(r_t) * discount_factor

        targnet_Q_t = targnet_Q[:, 1:]
        target_policy_probs_t = target_policy_probs[:, 1:]

        c_t = self._retrace_weights(target_policy_probs, behaviour_policy_probs)

        expected_Q_t = target_policy_probs_t * targnet_Q_t
        current = r_t + gamma * (expected_Q_t - c_t * targnet_Q_t)

        # calculate the sequence
        # Q'_t_prev   = r_t + γ * (expected_Q_t - c_t * targnet_Q_t) + γ * c_t * Q'_t
        # with current = r_t + γ * (expected_Q_t - c_t * targnet_Q_t)

        initial = targnet_Q_t[:, -1]

        reversed = self.reverse_sequence(sequence=current, num_sequences=minibatch_size, dim=1)

    def reverse_sequence(self, sequence, num_sequences, dim=1):
        for i in range(num_sequences):
            sequence[i, :] = sequence[i, :].flip(dims=[dim])
        return sequence


"""
sequence_ops.scan_discounted_sum(
        current,
        pcont_t * c_t,
        initial_value,
        reverse=True,
        back_prop=back_prop)

        def scan_discounted_sum(sequence, decay, initial_value, reverse=False,
                        sequence_lengths=None, back_prop=True,
                        name="scan_discounted_sum"):
"""

"""    target = _general_off_policy_corrected_multistep_target(
        r_t, pcont_t, target_policy_t, c_t, targnet_q_t, a_t,
        not stop_targnet_gradients)

    _general_off_policy_corrected_multistep_target(r_t,
                                                   pcont_t,
                                                   target_policy_t,
                                                   c_t,
                                                   q_t,
                                                   a_t,
                                                   back_prop=False,
                                                   name=None):
"""

# todo< include discount factor later, for now assume it is 1

