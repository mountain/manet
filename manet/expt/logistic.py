import torch as th

from torch import nn
from manet.iter import IterativeMap

from torch import Tensor
from typing import List, TypeVar, Tuple, Type, Callable, Union, Any, Dict
from torch.nn import Module

Lg: Type = TypeVar('Lg', bound='LogisticFunction')


class LogisticFunction(IterativeMap):

    def __init__(self: Lg, num_steps: int = 3, p: float = 3.8, debug: bool = False, debug_key: str = None, logger: TensorBoardLogger = None) -> None:
        super().__init__(num_steps=num_steps)
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
    def pre_forward(self: Lg, data: Tensor) -> Tensor:
        sz = data.size()
        data = data.view(-1, sz[1], sz[2] * sz[3])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.channel_transform)
        data = data.view(-1, sz[2] * sz[3], sz[1])
        return data

    def forward(self: F, data: Tensor) -> Tensor:
        sz = data.size()

        if self.debug and self.logger is not None:
            if sz[0] > self.num_samples:
                self.logger.add_figure('%s:0:function:total' % self.debug_key, self.plot_total_function(), self.global_step)

        data = data.view(-1, sz[1], sz[2] * sz[3])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.channel_transform)
        data = data.view(-1, sz[2] * sz[3], sz[1])

        if self.debug and self.logger is not None:
            if sz[0] > self.num_samples:
                for ix in range(self.num_samples):
                    self.logger.add_histogram('%s:1:before:%d:histo' % (self.debug_key, self.labels[ix]), data[ix], self.global_step)
                    image = data[ix].view(sz[2], sz[3] * sz[1])
                    image = (image - image.min()) / (image.max() - image.min())
                    self.logger.add_image('%s:1:before:%d:image' % (self.debug_key, self.labels[ix]), image, self.global_step, dataformats='HW')

        data = th.sigmoid(data)
        for ix in range(self.num_steps):
            data = self.p * data * (1 - data)

        if self.debug and self.logger is not None:
            if sz[0] > self.num_samples:
                for ix in range(self.num_samples):
                    self.logger.add_histogram('%s:2:after:%d:histo' % (self.debug_key, self.labels[ix]), data[ix], self.global_step)
                    image = data[ix].view(sz[2], sz[3] * sz[1])
                    image = (image - image.min()) / (image.max() - image.min())
                    self.logger.add_image('%s:2:after:%d:image' % (self.debug_key, self.labels[ix]), image, self.global_step, dataformats='HW')

        data = data.view(-1, sz[2] * sz[3], sz[1])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.spatio_transform)
        data = data.view(*sz)

        return data
