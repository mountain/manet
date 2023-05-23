import numpy as np
import torch as th
import torch.nn as nn
from lightning.pytorch.loggers import TensorBoardLogger

from torch import Tensor
from typing import TypeVar, Tuple, Callable, Union

from manet.aeg.params import CubicHermiteParam
from manet.nn.iter import IterativeMap
from manet.tools.ploter import plot_iterative_function, plot_image, plot_histogram
from manet.tools.profiler import Profiler

F = TypeVar('F', bound='ExprFlow')
Lf = TypeVar('Lf', bound='LearnableFunction')


Accessor = Callable[[Tensor], Tensor]
Mapper = Callable[[Tensor], Tensor]
Evaluator = Callable[[Tensor], Tensor]
Reducer = Callable[[Evaluator, Tensor, Tensor], Tensor]
Initializer = Union[None, Callable, Tensor]


class LearnableFunction(IterativeMap, Profiler):
    dkey: str = 'lf'

    def __init__(self: Lf, num_steps: int = 3, num_points: int = 5, length: float = 1.0, dkey: str = None) -> None:
        IterativeMap.__init__(self, num_steps=num_steps)
        Profiler.__init__(self, dkey=dkey)

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
        self.channel_transform = nn.Parameter(th.normal(0, 1, (1, 1)))
        self.spatio_transform = nn.Parameter(th.normal(0, 1, (1, 1)))
        self.maxval = np.sinh(self.length)

    @plot_iterative_function(dkey)
    def before_forward(self: Lf, data: Tensor) -> Tensor:
        self.size = data.size()
        return data * self.maxval

    @plot_image(dkey)
    @plot_histogram(dkey)
    def pre_transform(self: Lf, data: Tensor) -> Tensor:
        sz = self.size
        data = data.view(-1, sz[1], sz[2] * sz[3])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.channel_transform)
        data = data.view(-1, sz[2] * sz[3], sz[1])
        return data

    def mapping(self: Lf, data: th.Tensor) -> th.Tensor:
        handle = self.params.handler(data)
        velocity, angle = self.params('velocity', handle), self.params('angles', handle)
        return data + (velocity * th.cos(angle) + data * velocity * th.sin(angle)) * self.length / self.num_steps

    @plot_image(dkey)
    @plot_histogram(dkey)
    def post_transform(self: Lf, data: Tensor) -> Tensor:
        sz = self.size
        data = data.view(-1, sz[2] * sz[3], sz[1])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.spatio_transform)
        data = data.view(*sz)
        return data

    def after_forward(self: Lf, data: Tensor) -> Tensor:
        sz = self.size
        data = data.view(*sz)
        return data / self.maxval
