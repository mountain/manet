from typing import Union

import torch as th
import matplotlib.pyplot as plt

from aspectlib import Aspect, Proceed, Return

from manet.iter import IterativeMap
from manet.tools.profiler import Profiler, global_step

Model = Union[IterativeMap, Profiler]


def plot_iterative_function(dkey: str = None):
    @Aspect
    def iterative_function_plotter(*args, **kwargs):
        from manet.tools.profiler import global_step, tb_logger

        model: Model = args[0]
        if model.counter is None:
            model.initialize()

        result = yield Proceed

        if model.debug and global_step % model.sampling_steps == 0:
            xs = th.linspace(0, 1, 1000).view(1, 1000)
            xs.requires_grad = False
            ys = model.mapping(xs)
            xs, ys = xs[0].detach().cpu().numpy(), ys[0].detach().cpu().numpy()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(xs, ys)
            key = '%d:%s:%s' % (model.order, dkey, model.dkey)
            tb_logger.add_figure(key, fig, model.global_step)

        yield Return(result)

    return iterative_function_plotter


def plot_image(dkey: str = None):
    @Aspect
    def image_plotter(*args, **kwargs):
        from manet.tools.profiler import global_step, tb_logger

        model: Model = args[0]
        if model.counter is None:
            model.initialize()

        result = yield Proceed

        if model.debug and global_step % model.sampling_steps == 0:
            sz = model.size
            data = result.view(-1, sz[1], sz[2] * sz[3])
            key = '%d:%s:%s' % (model.order, dkey, model.dkey)
            tb_logger.add_image(key, data[0], model.global_step)

        yield Return(result)

    return image_plotter


def plot_histogram(dkey: str = None):
    @Aspect
    def histogram_plotter(*args, **kwargs):
        from manet.tools.profiler import global_step, tb_logger

        model: Model = args[0]
        if model.counter is None:
            model.initialize()

        result = yield Proceed

        if model.debug and global_step % model.sampling_steps == 0:
            sz = model.size
            data = result.view(-1, sz[1], sz[2] * sz[3])
            key = '%d:%s:%s' % (model.order, dkey, model.dkey)
            tb_logger.add_histogram(key, data[0], model.global_step)

        yield Return(result)

    return histogram_plotter


def plot_invoke(dkey: str = None):
    @Aspect
    def invoke_plotter(*args, **kwargs):
        from manet.tools.profiler import global_step, tb_logger

        velo, theta = velocity.detach().cpu().numpy(), angle.detach().cpu().numpy()
        x = velo * np.cos(theta)
        y = velo * np.sin(theta)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, y)
        return fig

    return invoke_plotter