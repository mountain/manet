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
P = TypeVar('P', bound='MLP')

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

    def reduce(self: F, data: Tensor) -> Tensor:
        raise NotImplemented()

    def evaluate(self: F, data: Tensor) -> Tensor:
        handler = self.accessor(data)
        velocity, angle = self.access(handler)
        return data + velocity * th.cos(angle) + data * velocity * th.sin(angle)

    def forward(self: F, data: Tensor) -> Tensor:
        return self.reduce(data)

    def plot_invoke(self: F, velocity: Tensor, angle: Tensor) -> Tensor:
        import numpy as np
        import matplotlib.pyplot as plt
        velo, theta = velocity.detach().cpu().numpy(), angle.detach().cpu().numpy()
        x = velo * np.cos(theta)
        y = velo * np.sin(theta)
        plt.scatter(x, y)
        return plt.figure()


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
        plt.plot(x, y)
        return plt.figure()


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

        self.debug = debug
        self.debug_key = debug_key
        self.logger = logger
        self.current_epoch = 0

    def forward(self: F, data: Tensor) -> Tensor:
        sz = data.size()

        if self.debug and self.logger is not None:
            self.logger.add_figure('%s:function:velocity:%d' % (self.debug_key, self.current_epoch), self.params.plot_function('velocity'), self.current_epoch)
            self.logger.add_figure('%s:function:angles:%d' % (self.debug_key, self.current_epoch), self.params.plot_function('angles'), self.current_epoch)
            self.logger.add_histogram('%s:input:%d:0' % (self.debug_key, self.current_step), data[0], self.current_epoch)
            if sz[0] > 1:
                self.logger.add_histogram('%s:input:%d:1' % (self.debug_key, self.current_step), data[1], self.current_epoch)
                self.logger.add_histogram('%s:input:%d:2' % (self.debug_key, self.current_step), data[2], self.current_epoch)
                self.logger.add_histogram('%s:input:%d:3' % (self.debug_key, self.current_step), data[3], self.current_epoch)
                self.logger.add_histogram('%s:input:%d:4' % (self.debug_key, self.current_step), data[4], self.current_epoch)

        data = data.view(-1, sz[1], sz[2] * sz[3])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.channel_transform)

        for ix in range(self.num_steps):
            handler = self.params.handler(data)
            velocity, angle = self.params('velocity', handler), self.params('angles', handler)
            data = data + (velocity * th.cos(angle) + data * velocity * th.sin(angle)) * self.length / self.num_steps

            if self.debug and self.logger is not None:
                self.logger.add_figure(
                    '%s:invoke:%d:%d:%d' % (self.debug_key, self.current_step, 0, ix),
                    self.plot_invoke(velocity[0], angle[0]), self.current_epoch
                )
                if sz[0] > 1:
                    self.logger.add_figure(
                        '%s:invoke:%d:%d:%d' % (self.debug_key, self.current_step, 0, ix),
                        self.plot_invoke(velocity[1], angle[1]), self.current_epoch
                    )
                    self.logger.add_figure(
                        '%s:invoke:%d:%d:%d' % (self.debug_key, self.current_step, 0, ix),
                        self.plot_invoke(velocity[2], angle[2]), self.current_epoch
                    )
                    self.logger.add_figure(
                        '%s:invoke:%d:%d:%d' % (self.debug_key, self.current_step, 0, ix),
                        self.plot_invoke(velocity[3], angle[3]), self.current_epoch
                    )
                    self.logger.add_figure(
                        '%s:invoke:%d:%d:%d' % (self.debug_key, self.current_step, 0, ix),
                        self.plot_invoke(velocity[4], angle[4]), self.current_epoch
                    )

        data = data.view(-1, sz[2] * sz[3], sz[1])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.spatio_transform)

        if self.debug and self.logger is not None:
            self.logger.add_histogram('%s:output:%d:0' % (self.debug_key, self.current_step), data[0], self.current_epoch)
            if sz[0] > 1:
                self.logger.add_histogram('%s:output:%d:1' % (self.debug_key, self.current_step), data[1], self.current_epoch)
                self.logger.add_histogram('%s:output:%d:2' % (self.debug_key, self.current_step), data[2], self.current_epoch)
                self.logger.add_histogram('%s:output:%d:3' % (self.debug_key, self.current_step), data[3], self.current_epoch)
                self.logger.add_histogram('%s:output:%d:4' % (self.debug_key, self.current_step), data[4], self.current_epoch)

        data = data.view(*sz)
        return data
