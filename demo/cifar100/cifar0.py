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
            th.normal(0, 1, (1, 1, 1))
        )
        self.spatio_transform = nn.Parameter(
            th.normal(0, 1, (1, 1, 1))
        )

    def accessor(self: U,
                 data: Tensor,
                 ) -> Tuple[Tensor, Tensor]:

        import manet.func.interp as interp
        data = data.flatten(0)
        dmax, dmin = data.max().item() + 0.1, data.min().item() - 0.1
        prob, grid = th.histogram(data, bins=self.points, range=(dmin, dmax), density=True)
        prob = prob / prob.sum()
        accum = th.cumsum(prob, dim=0) * self.points
        grid = (grid[1:] + grid[:-1]) / 2
        index = interp.interp1d(grid, accum, data)
        frame = interp.interp1d(accum, grid, th.arange(self.points))
        return frame, index

    @staticmethod
    def access(param: Tensor,
               accessor: Tuple[Tensor, Tensor]
               ) -> Tensor:

        frame, index = accessor
        frame = frame.view(1, 1, -1)
        index = index.view(1, 1, -1)
        param = th.addcmul(frame, th.ones_like(param), param, value=1e-3)

        begin = index.floor().long()
        pos = index - begin
        end = begin + 1
        begin = begin.clamp(0, param.size(0) - 1)
        end = end.clamp(0, param.size(0) - 1)

        value = (1 - pos) * param[begin] + pos * param[end]
        print('begin', begin.size(), begin)
        print('end', end.size(), end)
        print('pos', pos.size(), pos)
        print('param', param.size(), param)
        print('value', value.size(), value)
        return value

    def step(self: U,
             data: Tensor,
             param: Tensor,
             ) -> Tensor:

        accessor = self.accessor(data)
        theta = self.access(param[0:1], accessor)
        velo = self.access(param[1:2], accessor)

        ds = velo
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
        data = data.view(-1, 1, 1)

        trunk = []
        for ix in range(self.groups):
            data_slice = data[:, ix::self.groups]
            params_slice = self.params[:, ix:ix+1]
            trunk.append(self.step(data_slice, params_slice))
        data = th.cat(trunk, dim=1)

        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.spatio_transform)
        data = data.view(-1, 1, 1)

        return data.view(*shape)


class Cifar0(CIFARModel):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 15, kernel_size=7, padding=3)
        self.lnon0 = LNon(groups=5, points=120)
        self.conv1 = nn.Conv2d(15, 45, kernel_size=3, padding=1)
        self.lnon1 = LNon(groups=5, points=120)
        self.conv2 = nn.Conv2d(45, 135, kernel_size=1, padding=0)
        self.lnon2 = LNon(groups=5, points=120)
        self.conv3 = nn.Conv2d(135, 135, kernel_size=1, padding=0)
        self.lnon3 = LNon(groups=5, points=120)
        self.fc = nn.Linear(135 * 16, 100)

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
