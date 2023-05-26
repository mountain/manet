import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import lightning as pl

from manet.tools.profiler import reset_profiling_stage
from manet.tools.profiler import ctx


class BrachNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.lr = 0.01

    def make_plot(self, xs, ys, ix):
        inputs = xs[0].detach().cpu().numpy().reshape([2])
        ylist = ys[0].detach().cpu().numpy().reshape([1001])
        xlist = np.linspace(0, inputs[0], 1001)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xlist, ylist, color='b')

        r = 1.0590274735513345948
        ts = np.linspace(-np.pi, 4 * np.pi, 1001)
        xs = r * (ts + np.sin(ts))
        ys = 2 - r * (np.cos(ts) - 1)
        ax.plot(xs, ys, color='g')

        r = 1.3301938088969672306
        ts = np.linspace(-np.pi, 4 * np.pi, 1001)
        xs = r * (ts + np.sin(ts))
        ys = 2 - r * (np.cos(ts) - 1)
        ax.plot(xs, ys, color='r')

        r = 2.5335670497927349199
        ts = np.linspace(-np.pi, 4 * np.pi, 1001)
        xs = r * (ts + np.sin(ts))
        ys = 2 - r * (np.cos(ts) - 1)
        ax.plot(xs, ys, color='c')

        ctx['tb_logger'].add_figure('curve', fig, ix)

    def backward(self, loss, *args, **kwargs):
        loss.backward(*args, **kwargs, retain_graph=True)

    def forward(self, inputs):
        raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def benchmark(self, inputs, predict, targets):
        err0 = (predict[:, 0] - inputs[:, 1]) * (predict[:, 0] - inputs[:, 1])
        err1 = (predict[:, -1] - targets) * (predict[:, -1] - targets)
        err = err0 + err1
        t = 0
        for ix in range(0, 1000, 1):
            y0 = predict[:, ix, 0]
            x0 = inputs[:, 0] * ix / 1000
            v0 = th.sqrt(2 * 9.8 * th.relu(inputs[:, 1] - y0))
            y1 = predict[:, ix + 1, 0]
            x1 = inputs[:, 0] * (ix + 1) / 1000
            v1 = th.sqrt(2 * 9.8 * th.relu(inputs[:, 1] - y1))
            ds = th.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))
            dt = 2 * ds / (v0 + v1)
            t = t + dt

        mse = th.mean(err)
        mt = th.mean(t)

        return mse, mt, mse + mt

    def training_step(self, train_batch, batch_idx):
        reset_profiling_stage('train')
        xs, yt = train_batch
        ys = self.forward(xs)
        err, t, lss = self.benchmark(xs, ys, yt)
        self.log('pos_err', err, prog_bar=True)
        self.log('t_cost', t, prog_bar=True)
        self.log('train_loss', lss, prog_bar=True)
        return lss

    def validation_step(self, val_batch, batch_idx):
        reset_profiling_stage('valid')
        xs, yt = val_batch
        ys = self.forward(xs)
        err, t, lss = self.benchmark(xs, ys, yt)
        self.log('val_loss', lss, prog_bar=True)
        self.make_plot(xs, ys, self.current_epoch)

    def test_step(self, test_batch, batch_idx):
        reset_profiling_stage('test')
        xs, yt = test_batch
        ys = self.forward(xs)
        err, t, lss = self.benchmark(xs, ys, yt)
        self.log('test_loss', lss)


class TraceNet(BrachNet):
    def init(self, width, x0, y0):
        raise NotImplementedError()

    def trace(self, width, x, y):
        raise NotImplementedError()

    def forward(self, inputs):
        b = inputs.size()[0]
        result = []
        w = inputs[:, 0:1].reshape(b, 1, 1)
        y0 = inputs[:, 1:2].reshape(b, 1, 1)
        self.init(w, th.zeros_like(y0), y0)
        y = y0
        for ix in range(1001):
            if ix == 0:
                result.append(y0)
            elif ix == 1000:
                result.append(th.zeros_like(y0))
            else:
                ratio = ix / 1000
                x = w * ratio
                y = (1 - ratio) * (y0 - ratio * self.trace(w, x, y))
                result.append(y)
        return th.cat(result, dim=1)
