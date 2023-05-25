import torch as th
from torch import nn

from demo.brach.model import TraceNet
from manet.aeg.flow import LearnableFunction


class BRModel3(TraceNet):
    def __init__(self):
        super().__init__()
        self.g = nn.Parameter(th.ones(1, 1))
        self.a = LearnableFunction(in_channel=1, out_channel=1)
        self.t = LearnableFunction(in_channel=3, out_channel=1)

    def init(self, width, x, y):
        pass

    def trace(self, width, x, y):
        w = width.view(-1, 1, 1)
        a = self.a(w).view(-1, 1, 1)
        x_0 = x.view(-1, 1, 1)
        y_0 = y.view(-1, 1, 1)
        t_1 = self.t(th.cat((w, x_0, y_0), dim=1))
        x_1 = a * (1 - th.cos(t_1))
        y_1 = 2 - a * (t_1 - th.sin(t_1))
        t_2 = self.t(th.cat((w, x_1, y_1), dim=1))
        x_2 = a * (1 - th.cos(t_2))
        y_2 = 2 - a * (t_2 - th.sin(t_2))

        dt = t_2 - t_1
        vx = (x_2 - x_1) / dt
        vy = (y_2 - y_1) / dt
        k_1 = vx * vx + vy * vy
        u_1 = self.g * y_1
        e = k_1 + u_1

        dy = th.sqrt((e - self.g * y_0) * dt * dt - (x_1 - x_0) * (x_1 - x_0))
        return (y_0 + dy) * (y_2 > y_1) + (y_0 - dy) * (y_1 > y_2)


def _model_():
    return BRModel3()
