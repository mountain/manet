import numpy as np
import torch
import torch as th

from torch import nn
from torch import Tensor, device, dtype
from typing import Optional, List, Union, TypeVar, Tuple

from torch.nn import Module

T = TypeVar('T', bound='MacUnit')


def _exchangeable_multiplier_(factor1: int, factor2: int) -> Tuple[int, int]:
    # this is the normal setting
    # return factor2, factor1

    # in this setting, the size of parameters can be reduced dramatically
    # but the performance may be worse, need to be tested more
    lcm = np.lcm(factor1, factor2)
    return lcm // factor1, lcm // factor2


class MacUnit(nn.Module):
    def __init__(self: T,
                 in_channels: int,
                 out_channels: int,
                 in_spatio_dims: int = 1,
                 out_spatio_dims: int = 1,
                 num_steps: int = 3,
                 num_points: int = 17,
                 ) -> None:

        super().__init__()

        # the hyper-parameters
        self.num_steps = num_steps
        self.num_points = num_points
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_spatio_dims = in_spatio_dims
        self.out_spatio_dims = out_spatio_dims

        # the constant tensors
        self.in_channels_factor, self.out_channels_factor = _exchangeable_multiplier_(in_channels, out_channels)
        self.in_spatio_factor, self.out_spatio_factor = _exchangeable_multiplier_(in_spatio_dims, out_spatio_dims)
        self.ones = th.ones(1, 1, self.in_channels_factor, 1, self.in_spatio_factor)

        # the learnable parameters which govern the MAC unit
        self.angles = nn.Parameter(
            2 * th.rand(num_points) * th.pi
        )
        self.velocity = nn.Parameter(
            th.rand(num_points)
        )
        self.attention = nn.Parameter(
            th.rand(1, self.out_channels_factor, self.out_channels, self.out_spatio_factor, self.out_spatio_dims)
        )

        # the learnable parameters which govern the accessor
        # it is not works so far but will be improved in the future
        self.coeff = nn.Parameter(
            2 * th.rand(1) - 1
        )
        self.bias = nn.Parameter(
            2 * th.rand(1) - 1
        )

    def to(self: T,
           device: Optional[Union[int, device]] = ...,
           dtype: Optional[Union[dtype, str]] = ...,
           non_blocking: bool = ...
           ) -> Module:
        return super().to(device, dtype, non_blocking)

    def step(self: T,
             data: Tensor
             ) -> Tensor:

        # calculate the index of the accessor
        index = 1 + th.tanh(data * self.coeff + self.bias) / 2 * self.num_points
        bgn = index.floor().long()
        end = (index + 1).floor().long()
        bgn = bgn * (end < self.num_points) + (bgn - 1) * (end == self.num_points)
        end = end * (end < self.num_points) + (end - 1) * (end == self.num_points)

        # the velocity and angels of the accessor
        pos = index - bgn
        velo = ((1 - pos) * self.velocity[bgn] + pos * self.velocity[end])
        angels = ((1 - pos) * self.angles[bgn] + pos * self.angles[end])

        # by the flow equation of the arithmetic expression geometry
        return velo * th.cos(angels) + data * velo * th.sin(angels)

    def forward(self: T,
                data: Tensor
                ) -> Tensor:

        ones = torch.ones_like(data).view(
            -1, self.in_channels, 1,  self.in_spatio_dims, 1
        ).expand(
            -1, self.in_channels, self.in_channels_factor, self.in_spatio_dims, self.in_spatio_factor
        )
        data = data.view(-1, self.in_channels, 1, self.in_spatio_dims, 1) * ones

        for ix in range(self.num_steps):
            data = data + self.step(data) / self.num_steps

        return th.sum(
            self.attention * data.view(
                -1, self.out_channels_factor, self.out_channels,
                self.out_spatio_factor, self.out_spatio_dims
            ),
            dim=(1, 3)
        )


class MLP(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int]
    ):
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels:
            layers.append(MacUnit(in_dim, hidden_dim))
            in_dim = hidden_dim
        layers.append(nn.Flatten())
        super().__init__(*layers)


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)
