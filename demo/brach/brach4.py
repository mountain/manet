import torch as th

from demo.brach.model import TraceNet
from manet.mac import MLP


class BRModel4(TraceNet):
    def __init__(self):
        super().__init__()
        self.r = MLP(1, [1])
        self.t = MLP(3, [1])

    def init(self, width, x, y):
        pass

    def trace(self, width, x, y):
        w = width.view(-1, 1, 1)
        x = x.view(-1, 1, 1)
        y = y.view(-1, 1, 1)
        r = self.r(w).view(-1, 1, 1)
        t = self.t(th.cat((w, x, y), dim=1).view(-1, 3, 1)).view(-1, 1, 1)
        y_hat = 2 - r * (1 - th.cos(t))
        return y_hat


def _model_():
    return BRModel4()
