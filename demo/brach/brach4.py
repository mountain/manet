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

    def forward(self, inputs):
        b = inputs.size()[0]
        w = inputs[:, 0:1].reshape(b, 1, 1)
        y0 = inputs[:, 1:2].reshape(b, 1, 1)
        self.init(w, th.zeros_like(y0), y0)
        y = y0
        result, error = [], th.zeros_like(y0)
        for ix in range(1001):
            if ix == 0:
                result.append(y0)
            elif ix == 1000:
                result.append(th.zeros_like(y0))
            else:
                ratio = ix / 1000
                x = w * ratio
                y, xerr = self.trace(w, x, y)
                y = (1 - ratio) * (y0 - ratio * y)
                error = error + xerr
                result.append(y)
        return th.cat(result, dim=1), error

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

    def validation_step(self, val_batch, batch_idx):
        reset_profiling_stage('valid')
        xs, yt = val_batch
        ys, x_err = self.forward(xs)
        err, t, lss = self.benchmark(xs, ys, yt)
        self.log('x_err', x_err, prog_bar=True)
        self.log('val_loss', lss, prog_bar=True)
        self.make_plot(xs, ys, self.current_epoch)

    def test_step(self, test_batch, batch_idx):
        reset_profiling_stage('test')
        xs, yt = test_batch
        ys, x_err = self.forward(xs)
        err, t, lss = self.benchmark(xs, ys, yt)
        self.log('x_err', x_err, prog_bar=True)
        self.log('test_loss', lss)


def _model_():
    return BRModel4()
