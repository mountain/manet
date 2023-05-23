import torch as th

from manet.aeg.flow import LearnableFunction
from demo.brach.model import TraceNet


class BRModel2(TraceNet):
    def __init__(self):
        super().__init__()
        self.lf = LearnableFunction()
        self.model_name = 'v2'

    def init(self, width, x, y):
        pass

    def trace(self, width, x, y):
        y = y.view(-1, 1, 1, 1)
        return self.lf(y).view(-1, 1, 1)


def _model_():
    return BRModel2()
