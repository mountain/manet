import torch as th
import torch.nn.functional as F
import torch.nn as nn
import manet.func.interp as interp

from manet.nn.model import CIFARModel
from torch import Tensor
from typing import TypeVar, Tuple


U = TypeVar('U', bound='Unit')


class Foil(nn.Module):
    def __init__(self: U, groups: int = 1, points: int = 120) -> None:
        super().__init__()

        self.groups = groups
        self.points = points

        theta = th.cat([th.linspace(-th.pi, th.pi, points).view(1, 1, points) for _ in range(groups)], dim=1)
        velocity = th.cat([th.linspace(-3, 3, points).view(1, 1, points) for _ in range(groups)], dim=1)
        self.params = th.cat([theta, velocity], dim=0)

    @staticmethod
    def by_minmax(param, data):
        points = param.size(-1)
        shape = data.size()
        data_ = data.flatten(0)
        param_ = param.flatten(0)

        dmax, dmin = data_.max().item(), data_.min().item()
        data_ = (data_ - dmin) / (dmax - dmin + 1e-8) * (points - 1)
        index = data_.floor().long()
        frame = param_

        begin = index.floor().long()
        begin = begin.clamp(0, param.size(1) - 1)
        pos = index - begin
        end = begin + 1
        end = end.clamp(0, param.size(1) - 1)

        result = (1 - pos) * frame[begin] + pos * frame[end]

        return result.view(*shape)

    @staticmethod
    def by_sigmoid(param, data):
        points = param.size(-1)
        shape = data.size()
        data_ = data.flatten(0)
        param_ = param.flatten(0)

        index = th.sigmoid(data_) * (points - 1)
        frame = param_

        begin = index.floor().long()
        begin = begin.clamp(0, param.size(1) - 1)
        pos = index - begin
        end = begin + 1
        end = end.clamp(0, param.size(1) - 1)

        result = (1 - pos) * frame[begin] + pos * frame[end]

        return result.view(*shape)

    @staticmethod
    def by_tanh(param, data):
        points = param.size(-1)
        shape = data.size()
        data_ = data.flatten(0)
        param_ = param.flatten(0)

        index = th.tanh(data_) * (points - 1)
        frame = param_

        begin = index.floor().long()
        begin = begin.clamp(0, param.size(1) - 1)
        pos = index - begin
        end = begin + 1
        end = end.clamp(0, param.size(1) - 1)

        result = (1 - pos) * frame[begin] + pos * frame[end]

        return result.view(*shape)

    @staticmethod
    def by_dynamic(param, data):
        import manet.func.interp as interp

        points = param.size(-1)
        shape = data.size()
        data_ = data.flatten(0)
        param_ = param.flatten(0)

        dmax, dmin = data_.max().item(), data_.min().item()
        prob, grid = th.histogram(data_, bins=points, range=(dmin, dmax), density=True)
        prob = prob / prob.sum()
        accum = th.cumsum(prob, dim=0) * (points - 1)
        grid = (grid[1:] + grid[:-1]) / 2

        index = interp.interp1d(grid, accum, data_)
        frame = interp.interp1d(accum, param_, th.arange(points))

        begin = index.floor().long()
        begin = begin.clamp(0, frame.size(1) - 1)
        pos = index - begin
        end = begin + 1
        end = end.clamp(0, frame.size(1) - 1)

        result = (1 - pos) * frame[:, begin] + pos * frame[:, end]

        return result.view(*shape)

    def foilize(self: U, data: Tensor, param: Tensor) -> Tensor:

        theta = self.by_tanh(param[0:1], data)
        velo = self.by_tanh(param[1:2], data)
        ds = velo * 0.01
        dx = ds * th.cos(theta)
        dy = ds * th.sin(theta)
        val = data * (1 + dy) + dx
        return val

    def forward(self: U, data: Tensor) -> Tensor:
        shape = data.size()
        data = data.contiguous()

        trunk = []
        params = self.params * th.ones_like(self.params)
        for ix in range(self.groups):
            data_slice = data[:, ix::self.groups].reshape(-1, 1, 1)
            param_slice = params[:, ix:ix+1]
            trunk.append(self.foilize(data_slice, param_slice))
        data = th.cat(trunk, dim=1)

        return data.view(*shape)


class Cifar0(CIFARModel):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 135, kernel_size=7, padding=3)
        self.fiol0 = Foil(groups=1, points=3)
        self.conv1 = nn.Conv2d(135, 405, kernel_size=3, padding=1)
        self.fiol1 = Foil(groups=1, points=3)
        self.conv2 = nn.Conv2d(405, 405, kernel_size=1, padding=0)
        self.fiol2 = Foil(groups=1, points=3)
        self.conv3 = nn.Conv2d(405, 405, kernel_size=1, padding=0)
        self.fiol3 = Foil(groups=1, points=3)
        self.fc = nn.Linear(135 * 16, 100)

    def forward(self, x):
        x = self.conv0(x)
        x = self.fiol0(x)
        x = F.max_pool2d(x, 2)
        x = self.conv1(x)
        x = self.fiol1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = self.fiol2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.fiol3(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


def _model_():
    return Cifar0()
