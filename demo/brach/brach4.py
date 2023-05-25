import torch as th
from torch import nn

from demo.brach.model import TraceNet
from manet.aeg.flow import LearnableFunction


class BRModel4(TraceNet):
    def __init__(self):
        super().__init__()
        self.a = LearnableFunction(in_channel=1, out_channel=1)
        self.b = LearnableFunction(in_channel=1, out_channel=1)
        self.t = LearnableFunction(in_channel=3, out_channel=1)

    def init(self, width, x, y):
        pass

    def trace(self, width, x, y):
        w = width.view(-1, 1, 1)
        a = self.a(w).view(-1, 1, 1)
        b = self.b(w).view(-1, 1, 1)
        x_0 = x.view(-1, 1, 1)
        y_0 = y.view(-1, 1, 1)
        t_1 = self.t(th.cat((w, x_0, y_0), dim=1))
        return 2 - a * (t_1 - th.sin(b * t_1))


def _model_():
    return BRModel4()
