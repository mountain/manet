import torch as th

from demo.brach.model import TraceNet
from manet.aeg.flow import LearnableFunction


class BRModel3(TraceNet):
    def __init__(self):
        super().__init__()
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
        y_2 = 2 - a * (t_1 - th.sin(t_2))
        vx = (x_2 - x_1) / (t_2 - t_1)
        vy = (y_2 - y_1) / (t_2 - t_1)
        t_0_x = t_1 - (x_1 - x_0) / vx
        t_0_y = t_1 - (y_1 - y_0) / vy
        t_0 = (t_0_x + t_0_y) / 2

        return y_0 + (t_1 - t_0) * vy


def _model_():
    return BRModel3()
