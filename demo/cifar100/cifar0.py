import torch as th
import torch.nn.functional as F
import torch.nn as nn

from manet.nn.model import CIFARModel
from torch import Tensor
from typing import TypeVar, Tuple


U = TypeVar('U', bound='Unit')


class LNon(nn.Module):
    def __init__(self: U,
                 groups: int = 1,
                 points: int = 5,
                 ) -> None:
        super().__init__()

        self.groups = groups
        self.points = points

        theta = th.cat([th.linspace(0, 2 * th.pi, points).view(1, 1, points) for _ in range(groups)], dim=1)
        velocity = th.cat([th.linspace(0, 1, points).view(1, 1, points) for _ in range(groups)], dim=1)
        self.params = nn.Parameter(th.cat([theta, velocity], dim=0))

        self.channel_transform = nn.Parameter(
            th.normal(0, 1, (1, 1, 1, 1))
        )
        self.spatio_transform = nn.Parameter(
            th.normal(0, 1, (1, 1, 1, 1))
        )

    def accessor(self: U,
                 data: Tensor,
                 param: Tensor,
                 ) -> Tuple[Tensor, Tensor]:
        data = data.flatten(0)
        param = param.flatten(0)

        dmax, dmin = data.max().item(), data.min().item()
        prob, grid = th.histogram(data, bins=self.points, range=(dmin, dmax), density=True)
        prob = prob / prob.sum()
        accum = th.cumsum(prob, dim=0) * (self.points - 1)
        grid = (grid[1:] + grid[:-1]) / 2

        import manet.func.interp as interp
        index = interp.interp1d(grid, accum, data)
        frame = interp.interp1d(accum, param, th.arange(self.points))

        return frame, index

    @staticmethod
    def access(accessor: Tuple[Tensor, Tensor]) -> Tensor:

        frame, index = accessor
        frame = frame.view(1, 1, -1)
        index = index.view(-1)

        begin = index.floor().long()
        begin = begin.clamp(0, frame.size(2) - 1)
        pos = index - begin
        end = begin + 1
        end = end.clamp(0, frame.size(2) - 1)

        return (1 - pos) * frame[:, :, begin] + pos * frame[:, :, end]

    def step(self: U, data: Tensor, param: Tensor) -> Tensor:

        accessor = self.accessor(data, param[0:1])
        theta = self.access(accessor).reshape(*data.size())
        accessor = self.accessor(data, param[1:2])
        velo = self.access(accessor).reshape(*data.size())

        ds = velo * 0.01
        dx = ds * th.cos(theta)
        dy = ds * th.sin(theta)
        val = data * (1 + dy) + dx

        return val

    def forward(self: U,
                data: Tensor
                ) -> Tensor:
        shape = data.size()
        data = data.contiguous()

        data = data * self.channel_transform

        trunk = []
        params = self.params * th.ones_like(self.params)
        for ix in range(self.groups):
            data_slice = data[:, ix::self.groups].reshape(-1, 1, 1)
            param_slice = params[:, ix:ix+1]
            trunk.append(self.step(data_slice, param_slice))
        data = th.cat(trunk, dim=1)

        data = data * self.spatio_transform

        return data.view(*shape)


class Cifar0(CIFARModel):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 5, kernel_size=7, padding=3)
        self.lnon0 = LNon(groups=5, points=60)
        self.conv1 = nn.Conv2d(5, 15, kernel_size=3, padding=1)
        self.lnon1 = LNon(groups=5, points=60)
        self.conv2 = nn.Conv2d(15, 45, kernel_size=1, padding=0)
        self.lnon2 = LNon(groups=5, points=60)
        self.conv3 = nn.Conv2d(45, 45, kernel_size=1, padding=0)
        self.lnon3 = LNon(groups=5, points=60)
        self.fc = nn.Linear(45 * 16, 100)

    def forward(self, x):
        x = self.conv0(x)
        x = self.lnon0(x)
        x = F.max_pool2d(x, 2)
        x = self.conv1(x)
        x = self.lnon1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.lnon2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.lnon3(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


def _model_():
    return Cifar0()
