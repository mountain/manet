import torch as th
import torch.nn.functional as F
import torch.nn as nn

from manet.nn.model import CIFARModel
from torch import Tensor
from typing import TypeVar, Tuple


U = TypeVar('U', bound='Unit')


class LNon(nn.Module):

    def __init__(self, points=12):
        super().__init__()
        self.points = points
        self.iscale = nn.Parameter(th.normal(0, 1, (1, 1, 1, 1)))
        self.oscale = nn.Parameter(th.normal(0, 1, (1, 1, 1, 1)))
        self.theta = th.linspace(-th.pi, th.pi, points)
        self.velocity = th.linspace(0, 3, points)
        self.weight = nn.Parameter(th.normal(0, 1, (points, points)))

    @th.compile
    def integral(self, param, index):
        return th.sum(param[index].view(-1, 1) * th.softmax(self.weight, dim=1)[index, :], dim=1)

    @th.compile
    def interplot(self, param, index):
        lmt = param.size(0) - 1

        p0 = index.floor().long()
        p1 = p0 + 1
        pos = index - p0
        p0 = p0.clamp(0, lmt)
        p1 = p1.clamp(0, lmt)

        v0 = self.integral(param, p0)
        v1 = self.integral(param, p1)

        return (1 - pos) * v0 + pos * v1

    @th.compile
    def forward(self: U, data: Tensor) -> Tensor:
        if self.theta.device != data.device:
            self.theta = self.theta.to(data.device)
            self.velocity = self.velocity.to(data.device)
            self.weight = self.weight.to(data.device)

        shape = data.size()
        data = (data - data.mean()) / data.std() * self.iscale
        data = data.flatten(0)

        theta = self.interplot(self.theta, th.sigmoid(data) * (self.points - 1))
        ds = self.interplot(self.velocity, th.abs(th.tanh(data) * (self.points - 1)))

        dx = ds * th.cos(theta)
        dy = ds * th.sin(theta)
        data = data * th.exp(dy) + dx

        data = (data - data.mean()) / data.std()
        return data.view(*shape) * self.oscale


class Cifar2(CIFARModel):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 1024, kernel_size=7, padding=3, bias=False)
        self.fiol0 = LNon()
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.fiol1 = LNon()
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.fiol2 = LNon()
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.fiol3 = LNon()
        self.fc = nn.Linear(128 * 16, 100)

    @th.compile
    def forward(self, x):
        x = self.conv0(x)
        x = self.fiol0(x)
        x = F.max_pool2d(x, 2)
        x = self.conv1(x)
        x = F.dropout2d(x, 0.2, training=self.training)
        x = self.fiol1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.dropout2d(x, 0.2, training=self.training)
        x = self.fiol2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.fiol3(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


def _model_():
    return Cifar2()
