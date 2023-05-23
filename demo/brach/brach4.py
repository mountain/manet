import torch as th
import torch.nn as nn

from manet.mac import MLP
from demo.brach.model import TraceNet


class BRModel4(TraceNet):
    def __init__(self):
        super().__init__()
        self.mlp = MLP(2, [1])
        self.model_name = 'v3'

    def init(self, width, x, y):
        pass

    def trace(self, width, x, y):
        return self.mlp(th.cat((x, y), dim=1).view(-1, 2, 1)).view(-1, 1, 1)


def _model_():
    return BRModel4()
