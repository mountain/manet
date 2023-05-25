import torch as th

from demo.brach.model import TraceNet
from manet.aeg.flow import LearnableFunction


class BRModel3(TraceNet):
    def __init__(self):
        super().__init__()
        self.a = LearnableFunction(in_channel=1, out_channel=1)
        self.t = LearnableFunction(in_channel=1, out_channel=1)

    def init(self, width, x, y):
        pass

    def trace(self, width, x, y):
        a = self.a(width.view(-1, 1, 1)).view(-1, 1, 1)
        t = self.t(x.view(-1, 1, 1)).view(-1, 1, 1)
        y = 2 - a * (t - th.sin(t))
        return y


def _model_():
    return BRModel3()
