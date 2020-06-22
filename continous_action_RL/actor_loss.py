import torch


class ActorLoss(torch.nn.Module):
    def __init__(self, alpha=1.):
        super(ActorLoss, self).__init__()
        self.alpha = alpha

    def forward(self, Q, action_log_prob):
        return - (Q + self.alpha * action_log_prob).mean()
