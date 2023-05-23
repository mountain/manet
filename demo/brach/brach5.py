import torch as th
import torch.nn as nn

from manet.mac import MLP, MacMatrixUnit
from demo.brach.model import TraceNet


class BRModel5(TraceNet):
    def __init__(self):
        super().__init__()
        self.mlp = MLP(2, [1], mac_unit=MacMatrixUnit)
        self.model_name = 'v5'

    def init(self, width, x, y):
        pass

    def trace(self, width, x, y):
        return self.mlp(th.cat((x, y), dim=1).view(-1, 2, 1)).view(-1, 1, 1)


def _model_():
    return BRModel5()
