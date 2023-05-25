import torch as th
from torch import nn

from demo.brach.model import TraceNet
from manet.aeg.flow import LearnableFunction
from manet.tools.profiler import reset_profiling_stage


class BRModel4(TraceNet):
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

        return y_1, (x_1 - x) ** 2

    def training_step(self, train_batch, batch_idx):
        reset_profiling_stage('train')
        xs, yt = train_batch
        ys, x_err = self.forward(xs)
        err, t, lss = self.benchmark(xs, ys, yt)
        self.log('x_err', x_err, prog_bar=True)
        self.log('pos_err', err, prog_bar=True)
        self.log('t_cost', t, prog_bar=True)
        self.log('train_loss', lss, prog_bar=True)
        return lss + x_err


def _model_():
    return BRModel4()
