import torch
from torch.utils.tensorboard import SummaryWriter


class Logger(SummaryWriter):
    def __init__(self, log_every=10, logdir=None):
        super(Logger, self).__init__(log_dir=logdir)
        self.log_every = log_every

    def log_DNN_gradients(self, net: [torch.nn.Module], name=""):
        try:
            for layer, params in net.named_parameters():
                self.add_histogram(tag=name + "_grad_" + layer, values=params.grad.data)

        except ValueError:
            print("Error, network gradients could not be logged.")

    def log_DNN_params(self, net: [torch.nn.Module], name=""):
        try:
            for layer, params in net.named_parameters():
                self.add_histogram(tag=name + "_" + layer, values=params)

        except ValueError:
            print("Error, network parameters could not be logged.")







