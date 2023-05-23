import numpy as np
import torch as th
import torch.nn as nn
from lightning.pytorch.loggers import TensorBoardLogger

from torch import Tensor
from typing import TypeVar, Tuple, Callable, Union

F = TypeVar('F', bound='ExprFlow')


Accessor = Callable[[Tensor], Tensor]
Mapper = Callable[[Tensor], Tensor]
Evaluator = Callable[[Tensor], Tensor]
Reducer = Callable[[Evaluator, Tensor, Tensor], Tensor]
Initializer = Union[None, Callable, Tensor]


class ExprFlow(nn.Module):
    def __init__(self: F) -> None:
        super().__init__()

    def accessor(self: F, data: Tensor) -> Tensor:
        raise NotImplemented()

    def access(self: F, handler: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplemented()

    def forward(self: F, data: Tensor) -> Tensor:
        return self.reduce(data)


class LearnableFunction(ExprFlow):
    def __init__(self: U, num_steps: int = 3, num_points: int = 5, length: float = 1.0,
                 debug: bool = False, debug_key: str = None, logger: TensorBoardLogger = None) -> None:
        super().__init__()
        self.num_steps = num_steps
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

        self.debug = debug
        self.debug_key = debug_key
        self.logger = logger
        self.global_step = 0
        self.labels = None
        self.num_samples = 20

    def plot_total_function(self: F) -> Tensor:
        line = th.linspace(0, self.num_points, 1000)
        handler = self.params.handler(line.view(1, 1000, 1))
        velocity = self.params('velocity', handler).view(1000)
        angle = self.params('angles', handler).view(1000)
        curve = line + (velocity * th.cos(angle) + line * velocity * th.sin(angle)) * self.length / self.num_steps

        import matplotlib.pyplot as plt
        x, y = line.detach().cpu().numpy(), curve.detach().cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        return fig

    def forward(self: F, data: Tensor) -> Tensor:
        sz = data.size()
        data = data * self.maxval

        if self.debug and self.logger is not None:
            if sz[0] > self.num_samples:
                self.logger.add_figure('%s:function:velocity' % self.debug_key, self.params.plot_function('velocity'), self.global_step)
                self.logger.add_figure('%s:function:angles' % self.debug_key, self.params.plot_function('angles'), self.global_step)
                self.logger.add_figure('%s:function:total' % self.debug_key, self.plot_total_function(), self.global_step)
                # for ix in range(self.num_samples):
                #     pass
                #     self.logger.add_histogram('%s:input:%d:histo' % (self.debug_key, self.labels[ix]), data[ix], self.global_step)

        data = data.view(-1, sz[1], sz[2] * sz[3])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.channel_transform)
        data = data.view(-1, sz[2] * sz[3], sz[1])

        if self.debug and self.logger is not None:
            if sz[0] > self.num_samples:
                for ix in range(self.num_samples):
                    self.logger.add_histogram('%s:before:%d:histo' % (self.debug_key, self.labels[ix]), data[ix], self.global_step)
                    image = data[ix].view(sz[2], sz[3] * sz[1])
                    image = (image - image.min()) / (image.max() - image.min())
                    self.logger.add_image('%s:before:%d:image' % (self.debug_key, self.labels[ix]), image, self.global_step, dataformats='HW')

        for ix in range(self.num_steps):
            handler = self.params.handler(data)
            velocity, angle = self.params('velocity', handler), self.params('angles', handler)
            data = data + (velocity * th.cos(angle) + data * velocity * th.sin(angle)) * self.length / self.num_steps

            # if self.debug and self.logger is not None:
            #     if sz[0] > self.num_samples:
            #         for jx in range(self.num_samples):
            #             pass
            #             self.logger.add_figure(
            #                 '%s:invoke:%d:%d' % (self.debug_key, self.labels[jx], ix),
            #                 self.plot_invoke(velocity[jx], angle[jx]), self.global_step
            #             )
            #             self.logger.add_histogram('%s:distr:%d:%d' % (self.debug_key, self.labels[jx], ix), data[ix], self.global_step)

        if self.debug and self.logger is not None:
            if sz[0] > self.num_samples:
                for ix in range(self.num_samples):
                    self.logger.add_histogram('%s:after:%d:histo' % (self.debug_key, self.labels[ix]), data[ix], self.global_step)
                    image = data[ix].view(sz[2], sz[3] * sz[1])
                    image = (image - image.min()) / (image.max() - image.min())
                    self.logger.add_image('%s:after:%d:image' % (self.debug_key, self.labels[ix]), image, self.global_step, dataformats='HW')

        data = data.view(-1, sz[2] * sz[3], sz[1])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.spatio_transform)
        data = data.view(*sz)

        # if self.debug and self.logger is not None:
        #     if sz[0] > self.num_samples:
        #         for ix in range(self.num_samples):
        #             pass
        #             self.logger.add_histogram('%s:output:%d' % (self.debug_key, self.labels[ix]), data[ix], self.global_step)

        return data / self.maxval
