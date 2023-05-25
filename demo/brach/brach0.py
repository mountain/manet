import torch as th
import torch.nn as nn

from torch.nn import LSTM
from demo.brach.model import TraceNet


s, c, h, l = 1, 3, 1, 1
hidden, current = th.zeros(l, s, h), th.zeros(l, s, h)


class BRModel1(TraceNet):
    def __init__(self):
        super().__init__()
        self.lstm = LSTM(input_size=c, hidden_size=h, num_layers=l)
        self.ln = nn.Linear(in_features=2 * h, out_features=c)
        self.model_name = 'v1'

        self.hdn = None
        self.cur = None

    def init(self, width, x, y):
        self.hdn = hidden
        self.cur = current

    def trace(self, width, x, y):
        b = width.size()[0]
        output, (self.hdn, self.cur) = self.lstm(th.cat((x, y, width), dim=2), (self.hdn, self.cur))
        return self.ln(output.reshape(-1, 2 * h)).reshape(b, s, 1)


def _model_():
    return BRModel1()
