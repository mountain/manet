import torch as th
import torchvision as tv

from torch import nn
from manet.nn.model import MNISTModel
from torch import Tensor
from typing import List, TypeVar, Tuple, Type

import torch._dynamo

torch._dynamo.config.suppress_errors = True


U = TypeVar('U', bound='Unit')


class Unit(nn.Module):
    def __init__(self: U,
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
        # the learnable parameters which govern the unit
        self.angles = nn.Parameter(
            th.linspace(0, 2 * th.pi, num_points).view(1, 1, num_points)
        )
        self.velocity = nn.Parameter(
            th.linspace(0, 1, num_points).view(1, 1, num_points)
        )

        self.channel_dim, self.spatio_dim = self.calculate()

        self.channel_transform = nn.Parameter(
            th.normal(0, 1, (1, self.in_channel, self.out_channel))
        )
        self.spatio_transform = nn.Parameter(
            th.normal(0, 1, (1, self.in_spatio, self.out_spatio))
        )

    def calculate(self: U) -> Tuple[int, int]:
        raise NotImplemented()

    def expansion(self: U, data: Tensor) -> Tensor:
        raise NotImplemented()

    def reduction(self: U, data: Tensor) -> Tensor:
        raise NotImplemented()

    @th.compile
    def nonlinear(self: U, data: Tensor) -> Tensor:
        for ix in range(self.num_steps):
            data = data + self.step(data) * self.step_length
        return data

    @th.compile
    def accessor(self: U,
                 data: Tensor,
                 ) -> Tuple[Tensor, Tensor, Tensor]:
        # calculate the index of the accessor
        num_points = self.num_points
        index = th.sigmoid(data) * num_points
        bgn = index.floor().long()
        bgn = bgn * (bgn >= 0)
        bgn = bgn * (bgn <= num_points - 2) + (bgn - 1) * (bgn > num_points - 2)
        bgn = bgn * (bgn <= num_points - 2) + (bgn - 1) * (bgn > num_points - 2)
        end = bgn + 1

        return index, bgn, end

    @th.compile
    def access(self: U,
               memory: Tensor,
               accessor: Tuple[Tensor, Tensor, Tensor]
               ) -> Tensor:

        index, bgn, end = accessor
        pos = index - bgn
        memory = memory.flatten(0)
        return (1 - pos) * memory[bgn] + pos * memory[end]

    @th.compile
    def step(self: U,
             data: Tensor
             ) -> Tensor:

        accessor = self.accessor(data)
        angels = self.access(self.angles, accessor)
        velo = self.access(self.velocity, accessor)

        # by the flow equation of the arithmetic expression geometry
        return (velo * th.cos(angels) + data * velo * th.sin(angels)) * self.step_length

    def calculate(self: U) -> Tuple[int, int]:
        channel_dim = self.in_channel * self.out_channel
        spatio_dim = self.in_spatio * self.out_spatio
        self.flag = self.in_channel * self.in_spatio > self.out_channel * self.out_spatio
        return channel_dim, spatio_dim

    @th.compile
    def forward(self: U,
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


class MLP(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        spatio_dim: int = 1,
        steps: int = 3,
        length: float = 1.0,
        points: int = 5,
        unit: Type[Unit] = Unit
    ) -> None:
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels:
            layers.append(unit(
                in_dim, hidden_dim, spatio_dim, spatio_dim, steps, length / steps, points
            ))
            in_dim = hidden_dim
        layers.append(nn.Flatten())
        super().__init__(*layers)


class Reshape(nn.Module):
    def __init__(self, *shape) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)


class Fashion0(MNISTModel):
    def __init__(self):
        super().__init__()
        self.recognizer = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, padding=2),
            MLP(1, [1], steps=12, length=1, points=2320),
            Reshape(5, 28, 28),
            nn.Conv2d(5, 5, kernel_size=3, padding=1),
            MLP(1, [1], steps=12, length=1, points=2320),
            Reshape(5, 28, 28),
            nn.MaxPool2d(2),
            nn.Conv2d(5, 15, kernel_size=3, padding=1),
            MLP(1, [1], steps=12, length=1, points=2320),
            Reshape(15, 14, 14),
            nn.Conv2d(15, 15, kernel_size=3, padding=1),
            MLP(1, [1], steps=12, length=1, points=2320),
            Reshape(15, 14, 14),
            nn.MaxPool2d(2),
            nn.Conv2d(15, 45, kernel_size=3, padding=1),
            MLP(1, [1], steps=12, length=1, points=2320),
            Reshape(45, 7, 7),
            nn.Conv2d(45, 45, kernel_size=3, padding=1),
            MLP(1, [1], steps=12, length=1, points=2320),
            Reshape(45, 7, 7),
            nn.MaxPool2d(2),
            nn.Conv2d(45, 135, kernel_size=3, padding=1),
            MLP(1, [1], steps=12, length=1, points=2320),
            Reshape(135 * 3 * 3),
            tv.ops.MLP(135 * 9, [10]),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.recognizer(x)

    def configure_optimizers(self):
        return th.optim.AdamW(self.parameters(), lr=self.learning_rate)


def _model_():
    return Fashion0()
