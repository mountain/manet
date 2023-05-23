import torch as th

from torch import nn
from demo.brach.model import TraceNet


class BRModel0(TraceNet):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(th.linspace(0.99, 0, 1001).reshape(1, 1001))

    def forward(self, xs):
        return self.param * th.ones_like(xs)

    def init(self, width, x, y):
        self.hdn = hidden
        self.cur = current

    def trace(self, width, x, y):
        b = width.size()[0]
        output, (self.hdn, self.cur) = self.lstm(th.cat((x, y), dim=2), (self.hdn, self.cur))
        return self.ln(output.reshape(-1, 2 * h)).reshape(b, s, 1)


def _model_():
    return BRModel0()
