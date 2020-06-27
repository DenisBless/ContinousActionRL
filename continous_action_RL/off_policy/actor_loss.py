import torch


class ActorLoss(torch.nn.Module):
    def __init__(self, alpha=1.):
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
        return - Q.mean()
        # return - (Q + self.alpha * action_log_prob).mean()
