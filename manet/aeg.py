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
Initializor = Union[None, Callable, Tensor]


def flowable(data: Tensor) -> Tuple[Callable[[Tensor, Tensor], Tensor], Callable[[Tensor, Tensor], Tensor]]:
    a, da = data, data.grad
    delta = (1 + a * a - da * da)
    y1, y2 = (a * da + th.sqrt(delta)) / (1 + a * a), (a * da - th.sqrt(delta)) / (1 + a * a)
    x1, x2 = da - a * y1, da - a * y2

    def forward(epsilon, z):
        return (z + x1 * epsilon) * (1 + y1 * epsilon)

    def backward(epsilon, z):
        return z * (1 - y2 * epsilon) - x2 * epsilon

    return forward, backward


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
    def __init__(self: P, module: Module, num_params: int = 5, initializors: Dict[str, Initializor] = None) -> None:
        super().__init__()
        self.num_params = num_params
        self.module = module
        if initializors is not None:
            for k, v in initializors.items():
                self._construct(k, v)

    def __call__(self: P, key: str, handler: Tensor) -> Tensor:
        return self.construct(key)[handler.long()]

    def _construct(self: P, key: str, initializor: Initializor = None):
        if key not in self.module._parameters:
            if initializor is None:
                tensor = th.normal(0, 1, [self.num_params])
            elif type(initializor) is Callable:
                tensor = initializor()
            else:
                tensor = initializor
            params = nn.Parameter(tensor)
            self.module.register_parameter(key, params)
        return self.module.get_parameter(key)


class DiscreteParam(Param):
    def __init__(self: P, module: Module, num_params: int = 5, initializors: Dict[str, Initializor] = None) -> None:
        super().__init__(module, num_params, initializors)


class PiecewiseLinearParam(Param):
    def __init__(self: P, module: Module, num_params: int = 5, initializors: Dict[str, Initializor] = None) -> None:
        super().__init__(module, num_params, initializors)

    def __call__(self: P, key: str, handler: Tensor) -> Tensor:
        param = self._construct(key)
        bgn = handler.floor().long()
        end = bgn + 1
        bgn = (bgn * (bgn + 1 < self.num_params) + (bgn - 1) * (bgn + 1 >= self.num_params)) * (bgn >= 0)
        end = (end * (end < self.num_params) + (end - 1) * (end == self.num_params)) * (end >= 0)
        t = handler - bgn
        p0, p1 = param[bgn], param[end]
        return (1 - t) * p0 + t * p1


class CubicHermiteParam(Param):
    def __init__(self: P, module: Module, num_params: int = 5, initializors: Dict[str, Initializor] = None) -> None:
        super().__init__(module, num_params, initializors)

    def __call__(self: P, key: str, handler: Tensor) -> Tensor:
        param = self._construct(key)
        value, derivative = param[0], param[1]

        bgn = handler.floor().long()
        end = bgn + 1
        bgn = (bgn * (bgn + 1 < self.num_params) + (bgn - 1) * (bgn + 1 >= self.num_params)) * (bgn >= 0)
        end = (end * (end < self.num_params) + (end - 1) * (end == self.num_params)) * (end >= 0)

        t = handler - bgn
        p0, p1 = value[bgn], value[end]
        m0, m1 = derivative[bgn], derivative[end]

        # Cubic Hermite spline
        q1 = (2 * t ** 3 - 3 * t ** 2 + 1) * p0
        q2 = (t ** 3 - 2 * t ** 2 + t) * m0
        q3 = (-2 * t ** 3 + 3 * t ** 2) * p1
        q4 = (t ** 3 - t ** 2) * m1

        return q1 + q2 + q3 + q4


class ZigzagFunction(ExprFlow):
    def __init__(self: U, num_steps: int = 3, num_points: int = 2, channel_dim: int = 1, spatio_dim: int = 1) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.num_points = num_points
        self.params = PiecewiseLinearParam(self, num_params=num_points, initializors={
            'channel:velocity': th.cat((
                th.linspace(0, 1, num_points).view(num_points),
                # th.linspace(0, 1, num_points).view(1, num_points),
                # th.ones(num_points).view(1, num_points) * 2 * th.pi / num_points
            ), dim=0),
            'channel:angles': th.cat((
                th.linspace(0, 1, num_points).view(num_points),
                # th.linspace(0, 2 * th.pi, num_points).view(1, num_points),
                # th.ones(num_points).view(1, num_points) * 2 * th.pi / num_points
            ), dim=0),
            # 'spatio:velocity': th.cat((
                                           #     th.linspace(0, 1, num_points).view(num_points),
                # th.linspace(0, 1, num_points).view(1, num_points),
                # th.ones(num_points).view(1, num_points) * 2 * th.pi / num_points
            # ), dim=0),
            # 'spatio:angles': th.cat((
                                           #     th.linspace(0, 1, num_points).view(num_points),
                # th.linspace(0, 2 * th.pi, num_points).view(1, num_points),
                # th.ones(num_points).view(1, num_points) * 2 * th.pi / num_points
            # ), dim=0)
        })
        self.channel_transform = nn.Parameter(th.normal(0, 1, (1, 1)))
        self.spatio_transform = nn.Parameter(th.normal(0, 1, (1, 1)))
        self.alpha1 = nn.Parameter(th.ones(1, 1, 1))
        self.alpha2 = nn.Parameter(th.ones(1, 1, 1))
        self.beta1 = nn.Parameter(th.zeros(1, 1, 1))
        self.beta2 = nn.Parameter(th.zeros(1, 1, 1))

    def forward(self: F, data: Tensor) -> Tensor:
        sz = data.size()

        data = data.view(-1, sz[1], sz[2] * sz[3])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.channel_transform)

        handler = th.sigmoid(data * self.alpha1 + self.beta1) * self.num_points
        velocity, angle = self.params('channel:velocity', handler), self.params('channel:angles', handler)
        data = data + (velocity * th.cos(angle) + data * velocity * th.sin(angle)) / 3
        data = data + (velocity * th.cos(angle) + data * velocity * th.sin(angle)) / 3
        data = data + (velocity * th.cos(angle) + data * velocity * th.sin(angle)) / 3

        data = data.view(-1, sz[2] * sz[3], sz[1])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.spatio_transform)

        # handler = th.sigmoid(data * self.alpha2 + self.beta2) * self.num_points
        # velocity, angle = self.params('spatio:velocity', handler), self.params('spatio:angles', handler)
        # data = data + (velocity * th.cos(angle) + data * velocity * th.sin(angle)) / 3
        # data = data + (velocity * th.cos(angle) + data * velocity * th.sin(angle)) / 3
        # data = data + (velocity * th.cos(angle) + data * velocity * th.sin(angle)) / 3

        data = data.view(*sz)
        return data
