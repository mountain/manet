import torch as th
import torch.nn.functional as F
import torch.nn as nn

from manet.nn.model import CIFARModel
from torch import Tensor
from typing import TypeVar, Tuple


U = TypeVar('U', bound='Unit')


class LNon(nn.Module):
    def __init__(self: U, groups: int = 1, points: int = 120) -> None:
        super().__init__()

        self.groups = groups
        self.points = points

        theta = th.cat([th.linspace(-th.pi, th.pi, points).view(1, 1, points) for _ in range(groups)], dim=1)
        velocity = th.cat([th.linspace(0, 3, points).view(1, 1, points) for _ in range(groups)], dim=1)
        self.params = th.cat([theta, velocity], dim=0)
        self.scalei = nn.Parameter(th.ones(1, groups, 1, 1))
        self.scaleo = nn.Parameter(th.ones(1, groups, 1, 1))

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

        index = th.abs(th.tanh(data_) * (points - 1))
        frame = param_

        begin = index.floor().long()
        begin = begin.clamp(0, param.size(1) - 1)
        pos = index - begin
        end = begin + 1
        end = end.clamp(0, param.size(1) - 1)

        result = (1 - pos) * frame[begin] + pos * frame[end]

        return result.view(*shape)

    def foilize(self: U, data: Tensor, param: Tensor) -> Tensor:
        theta = self.by_sigmoid(param[0:1], data)
        velo = self.by_tanh(param[1:2], data)
        ds = velo
        dx = ds * th.cos(theta)
        dy = ds * th.sin(theta)
        val = data * th.exp(dy) + dx
        return val

    def forward(self: U, data: Tensor) -> Tensor:
        shape = data.size()
        data = data.contiguous()
        data = (data - data.mean()) / data.std() * self.scalei.to(data.device)

        trunk = []
        params = self.params
        for ix in range(self.groups):
            data_slice = data[:, ix::self.groups].reshape(-1, 1, 1)
            param_slice = params[:, ix:ix+1]
            trunk.append(self.foilize(data_slice, param_slice))
        data = th.cat(trunk, dim=1)

        data = (data - data.mean()) / data.std() * self.scaleo.to(data.device)

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
