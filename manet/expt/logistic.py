import torch as th

from torch import nn
from manet.iter import IterativeMap
from manet.tools.profiler import Profiler

from torch import Tensor
from typing import TypeVar, Type

from manet.tools.ploter import plot_iterative_function, plot_image, plot_histogram

Lg: Type = TypeVar('Lg', bound='LogisticFunction')


class LogisticFunction(IterativeMap, Profiler):
    dkey: str = 'lg'

    def __init__(self: Lg, num_steps: int = 3, p: float = 3.8, debug: bool = False, dkey: str = None) -> None:
        IterativeMap.__init__(self, num_steps=num_steps)
        Profiler.__init__(self, debug=debug, dkey=dkey)

        self.size = None
        self.p = nn.Parameter(th.ones(1).view(1, 1)) * p
        self.channel_transform = nn.Parameter(th.normal(0, 1, (1, 1)))
        self.spatio_transform = nn.Parameter(th.normal(0, 1, (1, 1)))

    @plot_iterative_function(dkey)
    def before_forward(self: Lg, data: Tensor) -> Tensor:
        self.size = data.size()
        return data

    @plot_image(dkey)
    @plot_histogram(dkey)
    def pre_transform(self: Lg, data: Tensor) -> Tensor:
        sz = self.size
        data = data.view(-1, sz[1], sz[2] * sz[3])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.channel_transform)
        data = data.view(-1, sz[2] * sz[3], sz[1])
        data = th.sigmoid(data)
        return data

    def mapping(self: Lg, data: th.Tensor) -> th.Tensor:
        return self.p * data * (1 - data)

    @plot_image(dkey)
    @plot_histogram(dkey)
    def post_transform(self: Lg, data: Tensor) -> Tensor:
        sz = self.size
        data = data.view(-1, sz[2] * sz[3], sz[1])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.spatio_transform)
        data = data.view(*sz)
        return data
