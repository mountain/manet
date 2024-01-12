import torch as th
import torch.nn.functional as F
import torch.nn as nn
import torchvision as tv

from manet.nn.model import MNISTModel
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


class Fashion1(MNISTModel):
    def __init__(self):
        super().__init__()
        self.resnet = tv.models.resnet18(pretrained=False)
        self.resnet.num_classes = 10
        self.resnet.relu = LNon(groups=1, points=60)
        self.resnet.conv1 = nn.Conv2d(1, self.resnet.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512 * self.resnet.layer4[1].expansion, self.num_classes)
        self.resnet.layer1[0].relu = LNon(groups=1, points=60)
        self.resnet.layer1[1].relu = LNon(groups=1, points=60)
        self.resnet.layer2[0].relu = LNon(groups=1, points=60)
        self.resnet.layer2[1].relu = LNon(groups=1, points=60)
        self.resnet.layer3[0].relu = LNon(groups=1, points=60)
        self.resnet.layer3[1].relu = LNon(groups=1, points=60)
        self.resnet.layer4[0].relu = LNon(groups=1, points=60)
        self.resnet.layer4[1].relu = LNon(groups=1, points=60)

    def forward(self, x):
        x = self.resnet(x)
        x = F.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        return [th.optim.AdamW(self.parameters(), lr=self.learning_rate)]


def _model_():
    return Fashion1()
