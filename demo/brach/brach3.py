import torch as th
import torch.nn as nn

from manet.aeg.flow import LearnableFunction
from demo.brach.model import TraceNet


class BRModel2(TraceNet):
    def __init__(self):
        super().__init__()
        self.lni = nn.Linear(in_features=2, out_features=2)
        self.lno = nn.Linear(in_features=2, out_features=2)
        self.lf = LearnableFunction()
        self.model_name = 'v2'

    def init(self, width, x, y):
        pass

    def trace(self, width, x, y):
        w = self.lni(th.cat((x, y), dim=1))
        w1, w2 = w[:, 0:1], w[:, 1:2]
        w1 = w1.view(-1, 1, 1, 1)
        z1 = self.lf(w1).view(-1, 1, 1)
        return self.lno(th.cat((z1, w2), dim=1))[:, 0].view(-1, 1, 1)


def _model_():
    return BRModel2()
