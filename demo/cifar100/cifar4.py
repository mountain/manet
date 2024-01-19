import torch as th
import torch.nn.functional as F
import torch.nn as nn

from manet.nn.model import CIFARModel
from torch import Tensor
from typing import TypeVar, Tuple


U = TypeVar('U', bound='Unit')


class LNon(nn.Module):

    def __init__(self, channel, spatio, points=29):
        super().__init__()
        self.points = points
        self.iscale = nn.Parameter(th.normal(0, 1, (1, 1, 1, 1)))
        self.oscale = nn.Parameter(th.normal(0, 1, (1, 1, 1, 1)))
        self.theta = th.linspace(-th.pi, th.pi, points)
        self.velocity = th.linspace(0, th.e, points)
        self.weight_sp = nn.Parameter(th.normal(0, 1, (1, spatio, points)))
        self.weight_ch = nn.Parameter(th.normal(0, 1, (1, channel, points)))
        self.conv2d = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.channel = channel
        self.spatio = spatio
        self.shape = None

    @th.compile
    def calculate_channel_weight(self, data: Tensor) -> Tensor:
        data = data.view(-1, self.channel, self.spatio)
        batch = data.size(0)
        weight = th.bmm(data, self.weight_sp.repeat(batch, 1, 1))
        weight = weight.permute(0, 2, 1)
        return weight

    @th.compile
    def calculate_spatio_weight(self, data: Tensor) -> Tensor:
        data = data.view(-1, self.channel, self.spatio)
        data = data.permute(0, 2, 1)
        batch = data.size(0)
        weight = th.bmm(data, self.weight_ch.repeat(batch, 1, 1))
        return weight

    @th.compile
    def integral(self, data, param, index):
        ch_weight = self.calculate_channel_weight(data)
        sp_weight = self.calculate_spatio_weight(data)
        data = self.conv2d(data.view(*self.shape))
        weight = th.bmm(th.bmm(ch_weight, data.view(-1, self.channel, self.spatio)), sp_weight)
        weight = weight.view(-1, self.points)
        return th.sum(param[index].view(-1, 1) * th.softmax(weight, dim=1)[index, :], dim=1)

    @th.compile
    def interplot(self, data, param, index):
        lmt = param.size(0) - 1

        p0 = index.floor().long()
        p1 = p0 + 1
        pos = index - p0
        p0 = p0.clamp(0, lmt)
        p1 = p1.clamp(0, lmt)

        v0 = self.integral(data, param, p0)
        v1 = self.integral(data, param, p1)

        return (1 - pos) * v0 + pos * v1

    @th.compile
    def forward(self: U, data: Tensor) -> Tensor:
        if self.theta.device != data.device:
            self.theta = self.theta.to(data.device)
            self.velocity = self.velocity.to(data.device)
            self.weight_ch = self.weight_ch.to(data.device)
            self.weight_sp = self.weight_sp.to(data.device)

        shape = data.size()
        self.shape = shape
        data = (data - data.mean()) / data.std() * self.iscale
        data = data.flatten(0)

        theta = self.interplot(data, self.theta, th.sigmoid(data) * (self.points - 1))
        ds = self.interplot(data, self.velocity, th.abs(th.tanh(data) * (self.points - 1)))

        dx = ds * th.cos(theta)
        dy = ds * th.sin(theta)
        data = data * th.exp(dy) + dx

        data = (data - data.mean()) / data.std()
        return data.view(*shape) * self.oscale


class Cifar4(CIFARModel):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 256, kernel_size=7, padding=3, bias=False, stride=2)
        self.fiol0 = LNon(channel=256, spatio=16 * 16)
        self.conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False)
        self.fiol1 = LNon(channel=512, spatio=8 * 8)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2, bias=False)
        self.fiol2 = LNon(channel=512, spatio=4 * 4)
        self.conv3 = nn.Conv2d(512, 128, kernel_size=3, padding=1, stride=2, bias=False)
        self.fiol3 = LNon(channel=128, spatio=2 * 2)
        self.fc = nn.Linear(512, 100)

    @th.compile
    def forward(self, x):
        x = self.conv0(x)
        x = self.fiol0(x)

        x = self.conv1(x)
        x = self.fiol1(x)

        x = self.conv2(x)
        x = F.dropout2d(x, 0.72, training=self.training)
        x = self.fiol2(x)

        x = self.conv3(x)
        x = self.fiol3(x)

        x = x.flatten(1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x


def _model_():
    return Cifar4()
