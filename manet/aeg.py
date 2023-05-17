import torch as th
import torch.nn as nn

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

    def plot(self: P) -> Tensor:
        th.linspace(0, self.nun_points - 1, 1000)


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
    def __init__(self: U, num_steps: int = 3, num_points: int = 5, length: float = 1.0) -> None:
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

    def forward(self: F, data: Tensor) -> Tensor:
        sz = data.size()

        data = data.view(-1, sz[1], sz[2] * sz[3])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.channel_transform)

        for _ in range(self.num_steps):
            handler = self.handler(data)
            velocity, angle = self.params('velocity', handler), self.params('angles', handler)
            data = data + (velocity * th.cos(angle) + data * velocity * th.sin(angle)) * self.length / self.num_steps

        data = data.view(-1, sz[2] * sz[3], sz[1])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.spatio_transform)

        data = data.view(*sz)
        return data
