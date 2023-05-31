import numpy as np
import torch as th
import torch.nn as nn

from torch import Tensor
from typing import List, TypeVar, Tuple, Type

A = TypeVar('A', bound='AbstractUnit')
T = TypeVar('T', bound='MacTensorUnit')
M = TypeVar('M', bound='MacMatrixUnit')
S = TypeVar('S', bound='MacSplineUnit')
P = TypeVar('P', bound='MLP')


def _exchangeable_multiplier_(factor1: int, factor2: int) -> Tuple[int, int, int]:
    # this is the normal setting
    # return factor2, factor1

    # in this setting, the size of parameters can be reduced dramatically
    # but the performance may be worse, need to be tested more
    lcm = np.lcm(factor1, factor2)
    return lcm, lcm // factor1, lcm // factor2


class AbstractUnit(nn.Module):
    def __init__(self: A,
                 in_channel: int,
                 out_channel: int,
                 in_spatio: int = 1,
                 out_spatio: int = 1,
                 num_steps: int = 3,
                 step_length: float = 0.33333,
                 num_points: int = 5,
                 ) -> None:
        super().__init__()

        # the hyperparameters
        self.num_steps = num_steps
        self.step_length = step_length
        self.num_points = num_points
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.in_spatio = in_spatio
        self.out_spatio = out_spatio

        self.channel_dim, self.spatio_dim = self.calculate()
        # the learnable parameters which govern the MAC unit
        self.angles = nn.Parameter(
            th.linspace(0, 2 * th.pi, num_points).view(1, 1, num_points)
        )
        self.velocity = nn.Parameter(
            th.linspace(0, 1, num_points).view(1, 1, num_points)
        )

    def calculate(self: T) -> Tuple[int, int]:
        raise NotImplemented()

    def expansion(self: T, data: Tensor) -> Tensor:
        raise NotImplemented()

    def reduction(self: T, data: Tensor) -> Tensor:
        raise NotImplemented()

    def nonlinear(self: T, data: Tensor) -> Tensor:
        for ix in range(self.num_steps):
            data = data + self.step(data) * self.step_length
        return data

    def accessor(self: T,
                 data: Tensor,
                 func: str = 'ngd',
                 ) -> Tuple[Tensor, Tensor, Tensor]:

        # calculate the index of the accessor
        # index = th.sigmoid(data) * self.num_points
        import manet.func.sigmoid as sgmd
        num_points = self.num_points
        index = sgmd.functions[func](data) * num_points

        bgn = index.floor().long()
        bgn = bgn * (bgn >= 0)
        bgn = bgn * (bgn <= num_points - 2) + (bgn - 1) * (bgn > num_points - 2)
        bgn = bgn * (bgn <= num_points - 2) + (bgn - 1) * (bgn > num_points - 2)
        end = bgn + 1

        return index, bgn, end

    def access(self: T,
               memory: Tensor,
               accessor: Tuple[Tensor, Tensor, Tensor]
               ) -> Tensor:

        index, bgn, end = accessor
        pos = index - bgn
        memory = memory.flatten(0)
        return (1 - pos) * memory[bgn] + pos * memory[end]

    def step(self: T,
             data: Tensor
             ) -> Tensor:

        accessor = self.accessor(data, 'ngd')
        angels = self.access(self.angles, accessor)
        accessor = self.accessor(data, 'nerf')
        velo = self.access(self.velocity, accessor)

        # by the flow equation of the arithmetic expression geometry
        return (velo * th.cos(angels) + data * velo * th.sin(angels)) * self.step_length


class SplineUnit(AbstractUnit):
    def __init__(self: A,
                 in_channel: int,
                 out_channel: int,
                 in_spatio: int = 1,
                 out_spatio: int = 1,
                 num_steps: int = 3,
                 step_length: float = 0.33333,
                 num_points: int = 5,
                 ) -> None:
        super().__init__(in_channel, out_channel, in_spatio, out_spatio, num_steps, step_length, num_points)
        self.dangles = nn.Parameter(
            th.ones(num_points).view(1, 1, num_points) * 2 * th.pi / num_points
        )
        self.dvelocity = nn.Parameter(
            th.ones(num_points).view(1, 1, num_points) / num_points
        )

    def calculate(self: T) -> Tuple[int, int]:
        raise NotImplemented()

    def expansion(self: T, data: Tensor) -> Tensor:
        raise NotImplemented()

    def reduction(self: T, data: Tensor) -> Tensor:
        raise NotImplemented()

    def access2nd(self: T, value: Tensor, derivative: Tensor, accessor: Tuple[Tensor, Tensor, Tensor]) -> Tensor:

        index, bgn, end = accessor
        value = value.flatten(0)
        derivative = derivative.flatten(0)

        t = index - bgn
        p0, p1 = value[bgn], value[end]
        m0, m1 = derivative[bgn], derivative[end]

        # Cubic Hermite spline
        term1 = (2 * t ** 3 - 3 * t ** 2 + 1) * p0
        term2 = (t ** 3 - 2 * t ** 2 + t) * m0
        term3 = (-2 * t ** 3 + 3 * t ** 2) * p1
        term4 = (t ** 3 - t ** 2) * m1
        return term1 + term2 + term3 + term4

    def step(self: T,
             data: Tensor
             ) -> Tensor:

        accessor = self.accessor(data, 'ngd')
        angels = self.access2nd(self.angles, self.dangles, accessor)
        accessor = self.accessor(data, 'nerf')
        velo = self.access2nd(self.velocity, self.dvelocity, accessor)

        # by the flow equation of the arithmetic expression geometry
        return velo * th.cos(angels) + data * velo * th.sin(angels)


class MacTensorUnit(AbstractUnit):
    def __init__(self: T,
                 in_channel: int,
                 out_channel: int,
                 in_spatio: int = 1,
                 out_spatio: int = 1,
                 num_steps: int = 3,
                 step_length: float = 0.33333,
                 num_points: int = 5,
                 ) -> None:
        super().__init__(in_channel, out_channel, in_spatio, out_spatio, num_steps, step_length, num_points)
        self.in_channel_factor, self.out_channel_factor = None, None
        self.in_spatio_factor, self.out_spatio_factor = None, None
        self.channel_dim, self.spatio_dim = self.calculate()

        self.in_weight = nn.Parameter(
            th.normal(0, 1, (1, self.channel_dim, self.spatio_dim))
        )
        self.in_bias = nn.Parameter(
            th.normal(0, 1, (1, self.channel_dim, self.spatio_dim))
        )
        self.out_weight = nn.Parameter(
            th.normal(0, 1, (1, self.channel_dim, self.spatio_dim))
        )
        self.out_bias = nn.Parameter(
            th.normal(0, 1, (1, self.channel_dim, self.spatio_dim))
        )

    def calculate(self: T) -> Tuple[int, int]:
        channel_dim, self.in_channel_factor, self.out_channel_factor = _exchangeable_multiplier_(
            self.in_channel, self.out_channel
        )
        spatio_dim, self.in_spatio_factor, self.out_spatio_factor = _exchangeable_multiplier_(
            self.in_spatio, self.out_spatio
        )
        return channel_dim, spatio_dim

    def expansion(self: T, data: Tensor) -> Tensor:
        data = data.view(-1, self.in_channel, 1, self.in_spatio, 1)
        data = data * self.in_weight.view(1, self.in_channel, self.in_channel_factor, self.in_spatio, self.in_spatio_factor)
        data = data + self.in_bias.view(1, self.in_channel, self.in_channel_factor, self.in_spatio, self.in_spatio_factor)
        return data.view(-1, self.channel_dim, self.spatio_dim)

    def attention(self: T, data: Tensor) -> Tensor:
        data = data.view(-1, self.channel_dim, self.spatio_dim)
        data = data * self.out_weight + self.out_bias
        import manet.func.sigmoid as sgmd
        return sgmd.alg1(data)

    def reduction(self: T, data: Tensor) -> Tensor:
        data = data.view(-1, self.out_channel_factor, self.out_channel, self.out_spatio_factor, self.out_spatio)
        return th.sum(data, dim=(1, 3))

    def forward(self: T,
                data: Tensor
                ) -> Tensor:

        data = self.expansion(data)
        data = self.nonlinear(data)
        data = data * self.attention(data)

        return self.reduction(data)


class MacMatrixUnit(AbstractUnit):
    def __init__(self: M,
                 in_channel: int,
                 out_channel: int,
                 in_spatio: int = 1,
                 out_spatio: int = 1,
                 num_steps: int = 3,
                 step_length: float = 0.33333,
                 num_points: int = 5,
                 ) -> None:

        super().__init__(in_channel, out_channel, in_spatio, out_spatio, num_steps, step_length, num_points)
        self.flag = False
        self.channel_dim, self.spatio_dim = self.calculate()

        self.channel_transform = nn.Parameter(
            th.normal(0, 1, (1, self.in_channel, self.out_channel))
        )
        self.spatio_transform = nn.Parameter(
            th.normal(0, 1, (1, self.in_spatio, self.out_spatio))
        )

    def calculate(self: T) -> Tuple[int, int]:
        channel_dim = self.in_channel * self.out_channel
        spatio_dim = self.in_spatio * self.out_spatio
        self.flag = self.in_channel * self.in_spatio > self.out_channel * self.out_spatio
        return channel_dim, spatio_dim

    def forward(self: T,
                data: Tensor
                ) -> Tensor:

        data = data.contiguous()

        data = data.view(-1, self.in_channel, self.in_spatio)
        data = th.permute(data, [0, 2, 1]).reshape(-1, self.in_channel)
        data = th.matmul(data, self.channel_transform)
        data = data.view(-1, self.in_spatio, self.out_channel)

        data = self.nonlinear(data)

        data = th.permute(data, [0, 2, 1]).reshape(-1, self.in_spatio)
        data = th.matmul(data, self.spatio_transform)
        data = data.view(-1, self.out_channel, self.out_spatio)

        return data


class MacSplineUnit(SplineUnit):
    def __init__(self: S,
                 in_channel: int,
                 out_channel: int,
                 in_spatio: int = 1,
                 out_spatio: int = 1,
                 num_steps: int = 3,
                 step_length: float = 0.33333,
                 num_points: int = 5,
                 ) -> None:

        super().__init__(in_channel, out_channel, in_spatio, out_spatio, num_steps, step_length, num_points)
        self.channel_dim, self.spatio_dim = self.calculate()
        self.length = num_steps * step_length

        self.in_weight = nn.Parameter(
            th.normal(0, 1, (1, self.channel_dim, self.spatio_dim))
        )
        self.in_bias = nn.Parameter(
            th.normal(0, 1, (1, self.channel_dim, self.spatio_dim))
        )
        self.out_weight = nn.Parameter(
            th.normal(0, 1, (1, self.channel_dim, self.spatio_dim))
        )
        self.out_bias = nn.Parameter(
            th.normal(0, 1, (1, self.channel_dim, self.spatio_dim))
        )

    def calculate(self: S) -> Tuple[int, int]:
        channel_dim, self.in_channel_factor, self.out_channel_factor = _exchangeable_multiplier_(
            self.in_channel, self.out_channel
        )
        spatio_dim, self.in_spatio_factor, self.out_spatio_factor = _exchangeable_multiplier_(
            self.in_spatio, self.out_spatio
        )
        return channel_dim, spatio_dim

    def expansion(self: S, data: Tensor) -> Tensor:
        data = data.view(-1, self.in_channel, 1, self.in_spatio, 1)
        data = data * self.in_weight.view(1, self.in_channel, self.in_channel_factor, self.in_spatio, self.in_spatio_factor)
        data = data + self.in_bias.view(1, self.in_channel, self.in_channel_factor, self.in_spatio, self.in_spatio_factor)
        return data.view(-1, self.channel_dim, self.spatio_dim)

    def attention(self: S, data: Tensor) -> Tensor:
        data = data.view(-1, self.channel_dim, self.spatio_dim)
        data = data * self.out_weight + self.out_bias
        return th.sigmoid(data)

    def reduction(self: S, data: Tensor) -> Tensor:
        data = data.view(-1, self.out_channel_factor, self.out_channel, self.out_spatio_factor, self.out_spatio)
        return th.sum(data, dim=(1, 3))

    def forward(self: S,
                data: Tensor
                ) -> Tensor:

        data = self.expansion(data)
        data = self.nonlinear(data)
        data = data * self.attention(data)

        return self.reduction(data)


class MLP(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        spatio_dim: int = 1,
        mac_steps: int = 3,
        mac_length: float = 1.0,
        mac_points: int = 5,
        mac_unit: Type[AbstractUnit] = MacTensorUnit
    ) -> None:
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels:
            layers.append(mac_unit(
                in_dim, hidden_dim, spatio_dim, spatio_dim, mac_steps, mac_length / mac_steps, mac_points
            ))
            in_dim = hidden_dim
        layers.append(nn.Flatten())
        super().__init__(*layers)


class Classification(nn.Module):
    def __init__(
        self,
        num_class: int,
        length: float,
    ) -> None:
        super().__init__()
        self.num_class = num_class
        self.length = length
        theta = th.linspace(0, 2 * th.pi, 2 * num_class + 1)[1::2]
        values = (th.exp(length * th.sin(theta)) - 1) / th.tan(theta)
        self.values = values.view(1, num_class, 1, 1)
        self.matrix = nn.Parameter(
            th.normal(0, 1, (1, num_class, num_class))
        )

    def forward(self, x):
        err = ((x - self.values) ** 2).view(-1, 10)
        return th.matmul(err, self.matrix)


class Reshape(nn.Module):
    def __init__(self, *shape) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)
