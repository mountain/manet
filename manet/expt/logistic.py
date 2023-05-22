import torch as th

from torch import nn
from manet.iter import IterativeMap

from torch import Tensor
from typing import TypeVar, Type

from manet.tools.ploter import plot_iterative_function, plot_image, plot_histogram

Lg: Type = TypeVar('Lg', bound='LogisticFunction')


class LogisticFunction(IterativeMap):

    def __init__(self: Lg, num_steps: int = 3, p: float = 3.8, debug: bool = False, debug_key: str = None, logger: TensorBoardLogger = None) -> None:
        super().__init__(num_steps=num_steps)
        self.size = None
        self.p = nn.Parameter(th.ones(1).view(1, 1)) * p
        self.channel_transform = nn.Parameter(th.normal(0, 1, (1, 1)))
        self.spatio_transform = nn.Parameter(th.normal(0, 1, (1, 1)))

        self.debug = debug
        self.debug_key = debug_key
        self.logger = logger
        self.global_step = 0
        self.labels = None
        self.num_samples = 20

    @plot_iterative_function
    def before_forward(self: Lg, data: Tensor) -> Tensor:
        self.size = data.size()
        return data

    @plot_image
    @plot_histogram
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

    @plot_image
    @plot_histogram
    def post_transform(self: Lg, data: Tensor) -> Tensor:
        sz = self.size
        data = data.view(-1, sz[2] * sz[3], sz[1])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.spatio_transform)
        data = data.view(*sz)
        return data
