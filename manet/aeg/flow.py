import numpy as np
import torch as th
import torch.nn as nn

from torch import Tensor
from typing import TypeVar, Callable, Union

from manet.aeg.params import CubicHermiteParam
from manet.nn.iter import IterativeMap
from manet.tools.ploter import plot_iterative_function, plot_image, plot_histogram
from manet.tools.profiler import Profiler

Lf = TypeVar('Lf', bound='LearnableFunction')


class LearnableFunction(IterativeMap, Profiler):
    dkey: str = 'lf'

    def __init__(self: Lf, in_channel: int = 1, out_channel: int = 1,  in_spatio: int = 1, out_spatio: int = 1,
                 num_steps: int = 3, num_points: int = 5, length: float = 1.0, dkey: str = None) -> None:
        IterativeMap.__init__(self, num_steps=num_steps)
        Profiler.__init__(self, dkey=dkey)

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.in_spatio = in_spatio
        self.out_spatio = out_spatio

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
        self.channel_transform = nn.Parameter(th.normal(0, 1, (out_channel, in_channel)))
        self.spatio_transform = nn.Parameter(th.normal(0, 1, (in_spatio, out_spatio)))
        self.maxval = np.sinh(self.length)

    @plot_iterative_function(dkey)
    def before_forward(self: Lf, data: Tensor) -> Tensor:
        self.size = data.size()
        return data * self.maxval

    @plot_image(dkey)
    @plot_histogram(dkey)
    def pre_transform(self: Lf, data: Tensor) -> Tensor:
        data = th.matmul(self.channel_transform, data)
        return data

    def mapping(self: Lf, data: th.Tensor) -> th.Tensor:
        handle = self.params.handler(data)
        velocity, angle = self.params('velocity', handle), self.params('angles', handle)
        return data + (velocity * th.cos(angle) + data * velocity * th.sin(angle)) * self.length / self.num_steps

    @plot_image(dkey)
    @plot_histogram(dkey)
    def post_transform(self: Lf, data: Tensor) -> Tensor:
        sz = self.size
        data = th.matmul(data, self.spatio_transform)
        return data

    def after_forward(self: Lf, data: Tensor) -> Tensor:
        return data / self.maxval
