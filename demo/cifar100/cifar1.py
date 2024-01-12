import torch as th
import torch.nn.functional as F
import torch.nn as nn
import torchvision as tv

from manet.nn.model import CIFARModel
from torch import Tensor
from typing import TypeVar


U = TypeVar('U', bound='Unit')


class LNon(nn.Module):

    def __init__(self, points=30):
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


class Cifar1(CIFARModel):
    def __init__(self):
        super().__init__()
        self.resnet = tv.models.resnet18(pretrained=False)
        self.resnet.num_classes = 100
        self.resnet.inplanes = 64
        self.resnet.conv1 = nn.Conv2d(3, self.resnet.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.bn1 = self.resnet._norm_layer(self.resnet.inplanes)
        self.resnet.fc = nn.Linear(512 * self.resnet.layer4[1].expansion, self.resnet.num_classes)

        self.resnet.relu = LNon()
        self.resnet.layer1[0].relu = LNon()
        self.resnet.layer1[1].relu = LNon()
        self.resnet.layer2[0].relu = LNon()
        self.resnet.layer2[1].relu = LNon()
        self.resnet.layer3[0].relu = LNon()
        self.resnet.layer3[1].relu = LNon()
        self.resnet.layer4[0].relu = LNon()
        self.resnet.layer4[1].relu = LNon()

    def forward(self, x):
        x = self.resnet(x)
        x = F.log_softmax(x, dim=1)
        return x


def _model_():
    return Cifar1()
