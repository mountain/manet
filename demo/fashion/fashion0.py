import torch as th
import torch.nn.functional as F
import torch.nn as nn
import manet.func.sigmoid as sgmd

from manet.nn.model import MNISTModel
from torch import Tensor
from typing import TypeVar, Tuple


U = TypeVar('U', bound='Unit')


class LNon(nn.Module):
    def __init__(self: U,
                 steps: int = 3,
                 length: float = 0.33333,
                 points: int = 5,
                 ) -> None:
        super().__init__()

        self.num_steps = steps
        self.step_length = length
        self.num_points = points

        self.theta = nn.Parameter(
            th.linspace(0, 4 * th.pi, points).view(1, 1, points)
        )
        self.velocity = nn.Parameter(
            th.linspace(0, 1, points).view(1, 1, points)
        )
        self.channel_transform = nn.Parameter(
            th.normal(0, 1, (1, 1, 1))
        )
        self.spatio_transform = nn.Parameter(
            th.normal(0, 1, (1, 1, 1))
        )

    def accessor(self: U,
                 data: Tensor,
                 func: str = 'ngd',
                 ) -> Tuple[Tensor, Tensor, Tensor]:

        index = sgmd.functions[func](data) * self.num_points

        bgn = index.floor().long()
        bgn = bgn * (bgn >= 0)
        bgn = bgn * (bgn <= self.num_points - 2) + (bgn - 1) * (bgn > self.num_points - 2)
        bgn = bgn * (bgn <= self.num_points - 2) + (bgn - 1) * (bgn > self.num_points - 2)
        end = bgn + 1

        return index, bgn, end

    def access(self: U,
               memory: Tensor,
               accessor: Tuple[Tensor, Tensor, Tensor]
               ) -> Tensor:

        index, bgn, end = accessor
        pos = index - bgn
        memory = memory.flatten(0)
        return (1 - pos) * memory[bgn] + pos * memory[end]

    def step(self: U,
             data: Tensor
             ) -> Tensor:

        accessor = self.accessor(data, 'ngd')
        theta = self.access(self.theta, accessor)
        accessor = self.accessor(data, 'nerf')
        velo = self.access(self.velocity, accessor)

        # by the flow equation of the arithmetic expression geometry
        ds = velo * self.step_length
        dx = ds * th.cos(theta)
        dy = ds * th.sin(theta)
        val = data * (1 + dy) + dx
        return val

    def forward(self: U,
                data: Tensor
                ) -> Tensor:
        shape = data.size()
        data = data.flatten(1)
        data = data.contiguous()
        data = data.view(-1, 1, 1)

        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.channel_transform)
        # data = data * self.channel_transform
        data = data.view(-1, 1, 1)

        for ix in range(self.num_steps):
            data = self.step(data)

        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.spatio_transform)
        # data = data * self.spatio_transform
        data = data.view(-1, 1, 1)

        return data.view(*shape)


class Fashion0(MNISTModel):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(1, 5, kernel_size=7, padding=3)
        self.lnon0 = LNon(steps=2, length=1, points=120)
        self.conv1 = nn.Conv2d(5, 15, kernel_size=3, padding=1)
        self.lnon1 = LNon(steps=2, length=1, points=120)
        self.conv2 = nn.Conv2d(15, 45, kernel_size=3, padding=1)
        self.lnon2 = LNon(steps=2, length=1, points=120)
        self.conv3 = nn.Conv2d(45, 135, kernel_size=3, padding=1)
        self.lnon3 = LNon(steps=2, length=1, points=120)
        self.fc = nn.Linear(135 * 9, 10)
        self.lsftmx = nn.LogSoftmax(dim=1)

    def forward(self, x):
        y = self.conv0(x)
        z = self.lnon0(y)
        y = F.max_pool2d(y + z, 2)
        y = self.conv1(y)
        z = self.lnon1(y)
        y = F.max_pool2d(y + z, 2)
        y = self.conv2(y)
        z = self.lnon2(y)
        y = F.max_pool2d(y + z, 2)
        y = self.conv3(y)
        z = self.lnon3(y)
        y = y + z
        y = y.view(-1, 135 * 9)
        y = self.fc(y)
        y = self.lsftmx(y)
        return y


def _model_():
    return Fashion0()
