import torch as th
import matplotlib.pyplot as plt

from aspectlib import Aspect, Proceed, Return

from manet.tools.profiler import find_tb_logger


@Aspect
def plot_iterative_function(*args, **kwargs):
    result = yield Proceed

    model = args[0]

    xs = th.linspace(0, 1, 1000).view(1, 1000)
    xs.requires_grad = False
    ys = model.mapping(xs)

    xs, ys = xs[0].detach().cpu().numpy(), ys[0].detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(model.title())
    ax.plot(xs, ys)

    find_tb_logger(model).add_figure(model.title(), fig, model.global_step)

    yield Return(result)


def plot_image():
    return None


def plot_histogram():
    return None