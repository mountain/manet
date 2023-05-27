import numpy as np
import torch as th
import torch.nn as nn

from torch import Tensor
from typing import TypeVar, Callable, Union, List, Tuple

from manet.aeg.params import CubicHermiteParam
from manet.nn.iter import IterativeMap
from manet.tools.ploter import plot_iterative_function, plot_image, plot_histogram
from manet.tools.profiler import Profiler

Lf = TypeVar('Lf', bound='LearnableFunction')


class LearnableFunction(IterativeMap, Profiler):
    dkey: str = 'lf'

    def __init__(self: Lf, in_channel: int = 1, out_channel: int = 1,
                 num_steps: int = 3, num_points: int = 5, length: float = 1.0, dkey: str = None) -> None:
        IterativeMap.__init__(self, num_steps=num_steps)
        Profiler.__init__(self, dkey=dkey)

        self.in_channel = in_channel
        self.out_channel = out_channel
        # self.ch_transform = nn.Parameter(th.normal(0, 1, (out_channel, in_channel)))
        self.ch_transform = nn.Parameter(th.normal(0, 1, (in_channel, out_channel)))
        self.sp_transform = nn.Parameter(th.normal(0, 1, (1, 1)))

        self.size = None
        self.num_points = num_points
        self.length = length
        self.params = CubicHermiteParam(self, num_points=num_points, initializers={
            'velocity': th.cat((
                th.linspace(0, 1, num_points).view(1, num_points),
                th.ones(num_points).view(1, num_points) * 2 * th.pi / num_points
            ), dim=0),
            'angles': th.cat((
                th.linspace(0, 2 * th.pi, num_points).view(1, num_points),
                th.ones(num_points).view(1, num_points) * 2 * th.pi / num_points
            ), dim=0),
        })
        self.maxval = np.sinh(self.length)

    @plot_iterative_function(dkey)
    def before_forward(self: Lf, data: Tensor) -> Tensor:
        # data = th.matmul(self.ch_transform, data)

        data = data.view(-1, self.in_channel, 1)
        data = th.permute(data, [0, 2, 1]).reshape(-1, self.in_channel)
        data = th.matmul(data, self.ch_transform)
        data = data.view(-1, 1, self.out_channel)

        self.size = data.size()
        return data

    @plot_image(dkey)
    @plot_histogram(dkey)
    def pre_transform(self: Lf, data: Tensor) -> Tensor:
        data = data.view(-1, 1, 1) * self.maxval
        return data

    def mapping(self: Lf, data: th.Tensor) -> th.Tensor:
        handle = self.params.handler(data)
        velocity, angle = self.params('velocity', handle), self.params('angles', handle)
        return data + (velocity * th.cos(angle) + data * velocity * th.sin(angle)) * self.length / self.num_steps

    @plot_image(dkey)
    @plot_histogram(dkey)
    def post_transform(self: Lf, data: Tensor) -> Tensor:
        return data.view(*self.size) / self.maxval

    def after_forward(self: Lf, data: Tensor) -> Tensor:
        # data = th.matmul(data, self.sp_transform)

        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.sp_transform)
        data = data.view(-1, self.out_channel, 1)

        return data
