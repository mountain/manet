import torch as th

from demo.brach.model import TraceNet
from manet.mac import MLP, MacSplineUnit


class BRModel3(TraceNet):
    def __init__(self):
        super().__init__()
        self.mlp = MLP(3, [1], mac_unit=MacSplineUnit)

    def init(self, width, x, y):
        pass

    def trace(self, width, x, y):
        return self.mlp(th.cat((x, y, width), dim=1).view(-1, 3, 1)).view(-1, 1, 1)


def _model_():
    return BRModel3()
