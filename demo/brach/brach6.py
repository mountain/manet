import torch as th

from torchvision.ops import MLP
from demo.brach.model import TraceNet


class BRModel6(TraceNet):
    def __init__(self):
        super().__init__()
        self.mlp = MLP(2, [1])
        self.model_name = 'v6'

    def init(self, width, x, y):
        pass

    def trace(self, width, x, y):
        return self.mlp(th.cat((x, y), dim=1).view(-1, 2, 1)).view(-1, 1, 1)


def _model_():
    return BRModel6()
