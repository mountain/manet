import torch as th
import torch.nn as nn

from torch import Tensor
from typing import TypeVar, Tuple, Callable, Union, Dict
from torch.nn import Module

P = TypeVar('P', bound='Param')

Initializer = Union[None, Callable, Tensor]


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
