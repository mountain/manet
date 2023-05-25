import torch as th

from manet.aeg.flow import LearnableFunction
from demo.brach.model import TraceNet


class BRModel3(TraceNet):
    def __init__(self):
        super().__init__()
        self.lf = LearnableFunction(in_channel=2, out_channel=1)
        self.model_name = 'v3'

    def init(self, width, x, y):
        pass

    def trace(self, width, x, y):
        return self.lf(th.cat((x, y), dim=1).view(-1, 2, 1))


def _model_():
    return BRModel3()
