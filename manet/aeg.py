import numpy as np
import torch as th
import torch.nn as nn
from lightning.pytorch.loggers import TensorBoardLogger

from torch import Tensor
from typing import List, TypeVar, Tuple, Type, Callable, Union, Any, Dict
from torch.nn import Module

P = TypeVar('P', bound='Parametor')
U = TypeVar('U', bound='UnaryFunction')
F = TypeVar('F', bound='Flow')
M = TypeVar('M', bound='MacMatrixUnit')

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

    def plot_invoke(self: F, velocity: Tensor, angle: Tensor) -> Tensor:
        import numpy as np
        import matplotlib.pyplot as plt
        velo, theta = velocity.detach().cpu().numpy(), angle.detach().cpu().numpy()
        x = velo * np.cos(theta)
        y = velo * np.sin(theta)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, y)
        return fig


class Param:
    def __init__(self: P, module: Module, num_points: int = 5, initializers: Dict[str, Initializer] = None) -> None:
        super().__init__()
        self.num_points = num_points
        self.module = module
        self.alpha = nn.Parameter(th.ones(1, 1, 1))
        self.beta = nn.Parameter(th.zeros(1, 1, 1))
        if initializers is not None:
            for k, v in initializers.items():
                self._construct(k, v)

    def __call__(self: P, key: str, handler: Tensor) -> Tensor:
        return self.construct(key)[handler.long()]

    def handler(self: P, data: Tensor) -> Tensor:
        return th.sigmoid(data * self.alpha + self.beta) * self.num_points

    def begin_end_of(self: P, handler: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        bgn = handler.floor().long()
        end = bgn + 1
        bgn = (bgn * (bgn + 1 < self.num_points) + (bgn - 1) * (bgn + 1 >= self.num_points)) * (bgn >= 0)
        end = (end * (end < self.num_points) + (end - 1) * (end == self.num_points)) * (end >= 0)
        t = handler - bgn
        return bgn, end, t

    def _construct(self: P, key: str, initializer: Initializer = None):
        if key not in self.module._parameters:
            if initializer is None:
                tensor = th.normal(0, 1, [self.num_params])
            elif type(initializer) is Callable:
                tensor = initializer()
            else:
                tensor = initializer
            params = nn.Parameter(tensor)
            self.module.register_parameter(key, params)
        return self.module.get_parameter(key)

    def plot_function(self: F, key: str) -> Tensor:
        line = th.linspace(0, self.num_points, 1000)
        handler = self.handler(line.view(1, 1000, 1))
        curve = self(key, handler).view(1000)

        import matplotlib.pyplot as plt
        x, y = line.detach().cpu().numpy(), curve.detach().cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        return fig


class DiscreteParam(Param):
    def __init__(self: P, module: Module, num_params: int = 5, initializers: Dict[str, Initializer] = None) -> None:
        super().__init__(module, num_params, initializers)


class PiecewiseLinearParam(Param):
    def __init__(self: P, module: Module, num_points: int = 5, initializers: Dict[str, Initializer] = None) -> None:
        super().__init__(module, num_points, initializers)

    def __call__(self: P, key: str, handler: Tensor) -> Tensor:
        param = self._construct(key)
        bgn, end, t = self.begin_end_of(handler)
        p0, p1 = param[bgn], param[end]
        return (1 - t) * p0 + t * p1


class CubicHermiteParam(Param):
    def __init__(self: P, module: Module, num_points: int = 5, initializers: Dict[str, Initializer] = None) -> None:
        super().__init__(module, num_points, initializers)

    def __call__(self: P, key: str, handler: Tensor) -> Tensor:
        param = self._construct(key)
        value, derivative = param[0], param[1]

        bgn, end, t = self.begin_end_of(handler)
        p0, p1 = value[bgn], value[end]
        m0, m1 = derivative[bgn], derivative[end]

        # Cubic Hermite spline
        q1 = (2 * t ** 3 - 3 * t ** 2 + 1) * p0
        q2 = (t ** 3 - 2 * t ** 2 + t) * m0
        q3 = (-2 * t ** 3 + 3 * t ** 2) * p1
        q4 = (t ** 3 - t ** 2) * m1

        return q1 + q2 + q3 + q4


class CubicHermiteParam(Param):
    def __init__(self: P, module: Module, num_points: int = 5, initializers: Dict[str, Initializer] = None) -> None:
        super().__init__(module, num_points, initializers)

    def __call__(self: P, key: str, handler: Tensor) -> Tensor:
        param = self._construct(key)
        value, derivative = param[0], param[1]

        bgn, end, t = self.begin_end_of(handler)
        p0, p1 = value[bgn], value[end]
        m0, m1 = derivative[bgn], derivative[end]

        # Cubic Hermite spline
        q1 = (2 * t ** 3 - 3 * t ** 2 + 1) * p0
        q2 = (t ** 3 - 2 * t ** 2 + t) * m0
        q3 = (-2 * t ** 3 + 3 * t ** 2) * p1
        q4 = (t ** 3 - t ** 2) * m1

        return q1 + q2 + q3 + q4


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


class LogisticFunction(ExprFlow):

    def __init__(self: U, num_steps: int = 3, debug: bool = False, debug_key: str = None, logger: TensorBoardLogger = None) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.p = nn.Parameter(th.ones(1).view(1, 1)) * 3.8
        self.channel_transform = nn.Parameter(th.normal(0, 1, (1, 1)))
        self.spatio_transform = nn.Parameter(th.normal(0, 1, (1, 1)))

        self.debug = debug
        self.debug_key = debug_key
        self.logger = logger
        self.global_step = 0
        self.labels = None
        self.num_samples = 20

    def plot_total_function(self: F) -> Tensor:
        line = th.linspace(0, 1, 1000).view(1, 1000)
        curve = self.p * line * (1 - line)

        import matplotlib.pyplot as plt
        x, y = line.detach().cpu().numpy(), curve.detach().cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        return fig

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
