import torch


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
        return - (Q - self.alpha * action_log_prob).mean()

    def kl_divergence(self, old_mean, old_std, mean, std):
        """
        Computes:
        D_KL(π_old(a|s)||π(a|s)) = ∑_i D_KL(π_old(a_i|s)||π(a_i|s))
        where π(a_i|s) = N(a|μ, σ^2)
        and D_KL(π_old(a_i|s)||π(a_i|s)) = 1/2 * (log(σ^2/σ_old^2) + [σ_old^2 + (μ_old - μ)^2]/σ^2 -1)
        """
        t1 = torch.log(torch.pow(std, 2)) - torch.log(torch.pow(old_std, 2))
        t2 = (torch.pow(old_std, 2) + torch.pow(old_mean - mean, 2)) / torch.pow(std, 2)
        return torch.sum(t1 + t2 - 1)




