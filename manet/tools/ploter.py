import aspectlib
import torch as th
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from aspectlib import Aspect, Proceed, Return


@Aspect
def strip_return_value(*args, **kwargs):
    result = yield Proceed
    yield Return(result.strip())


def plot_iterative_function() -> Figure:
    line = th.linspace(0, 1, 1000).view(1, 1000)
    curve = self.p * line * (1 - line)

    x, y = line[0].detach().cpu().numpy(), curve[0].detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('p=%f' % self.p.item())
    ax.plot(x, y)
    return fig
