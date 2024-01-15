import torch as th
import torch.nn.functional as F
import torch.nn as nn
import torchvision as tv

from manet.nn.model import CIFARModel
from torch import Tensor
from typing import TypeVar

U = TypeVar('U', bound='Unit')


class LNon(nn.Module):

    def __init__(self, points=29):
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

    @th.compile
    def reverse(self: U, data: Tensor) -> Tensor:
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
        data = (data - dx) / th.exp(dy)

        data = (data - data.mean()) / data.std()
        return data.view(*shape) * self.oscale


class Cifar8(CIFARModel):
    def __init__(self):
        super().__init__()
        self.conv000 = nn.Conv2d(3, 128, kernel_size=7, padding=3, stride=2)  # (64, 64), 128
        self.lnon000 = LNon()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (32, 32), 128

        self.conv100 = nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=2)  # (16, 16), 256
        self.conv101 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.lnon101 = LNon()
        self.conv102 = nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=2)  # (16, 16), 128
        self.lnon102 = LNon()

        self.conv200 = nn.Conv2d(256, 512, kernel_size=1, padding=0, stride=2)  # (8, 8), 512
        self.conv201 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.lnon201 = LNon()
        self.conv202 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)  # (8, 8), ?
        self.lnon202 = LNon()

        self.conv300 = nn.Conv2d(512, 1024, kernel_size=1, padding=0, stride=2)  # (4, 4), 1024
        self.conv301 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.lnon301 = LNon()
        self.conv302 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=2)  # (4, 4), ?
        self.lnon302 = LNon()

        self.conv400 = nn.Conv2d(1024, 2048, kernel_size=1, padding=0, stride=2)  # (2, 2), 2048
        self.conv401 = nn.Conv2d(2048, 2048, kernel_size=3, padding=1)
        self.lnon401 = LNon()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # (1, 1), 2048
        self.fc = nn.Linear(2048, 100)  # (1), 2048

        self.feedback1 = None
        self.shape1 = None
        self.feedback2 = None
        self.shape2 = None
        self.feedback3 = None
        self.shape3 = None

    @th.compile
    def forward(self, x):
        b = x.size()[0]
        if b < 1024:
            x = th.cat([x, x[:1024 - b]], dim=0)

        x = self.conv000(x)
        x = self.lnon000(x)
        x = self.maxpool(x)

        if self.feedback1 == None:
            self.shape1 = x.size()
            self.feedback1 = x
        x = th.cat([x, self.feedback1], dim=1)

        x = self.conv100(x)
        x = self.conv101(x)
        x = self.lnon101(x)

        if self.feedback2 == None:
            self.shape2 = x.size()

        feedback1 = th.cat([x, x], dim=1).reshape(*self.shape1)

        x = self.conv200(x)
        x = self.conv201(x)
        x = self.lnon201(x)

        if self.feedback3 == None:
            self.shape3 = x.size()

        feedback2 = th.cat([x, x], dim=1).reshape(*self.shape2)

        x = self.conv300(x)
        x = self.conv301(x)
        x = self.lnon301(x)

        feedback3 = th.cat([x, x], dim=1).reshape(*self.shape3)
        feedback3 = self.lnon302(feedback3)
        feedback2 = th.cat([feedback3, feedback3], dim=1).reshape(*self.shape2) + feedback2
        feedback2 = self.lnon202(feedback2)
        feedback1 = th.cat([feedback2, feedback2], dim=1).reshape(*self.shape1) + feedback1
        self.feedback1 = self.lnon102(feedback1)
        self.feedback2 = feedback2
        self.feedback3 = feedback3

        x = self.conv400(x)
        x = self.conv401(x)
        x = self.lnon401(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        if b < 1024:
            x = x[:b]

        return x


class Cifar7(CIFARModel):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 256, kernel_size=7, padding=3, bias=False, stride=2)
        self.fiol0 = LNon()
        self.conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=False)
        self.fiol1 = LNon()
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2, bias=False)
        self.fiol2 = LNon()
        self.conv3 = nn.Conv2d(512, 128, kernel_size=3, padding=1, stride=2, bias=False)
        self.fiol3 = LNon()
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


class Cifar6(CIFARModel):
    def __init__(self):
        super().__init__()
        expansion = 4
        self.conv000 = nn.Conv2d(3, 64 * expansion, kernel_size=7, padding=3, stride=2)
        self.lnon000 = LNon()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv100 = nn.Conv2d(64 * expansion, 64 * expansion, kernel_size=1, padding=0, stride=2)
        self.conv101 = nn.Conv2d(64 * expansion, 64 * expansion, kernel_size=3, padding=1)
        self.lnon101 = LNon()

        self.conv200 = nn.Conv2d(64 * expansion, 128 * expansion, kernel_size=1, padding=0, stride=2)
        self.conv201 = nn.Conv2d(128 * expansion, 128 * expansion, kernel_size=3, padding=1)
        self.lnon201 = LNon()

        self.conv300 = nn.Conv2d(128 * expansion, 256 * expansion, kernel_size=1, padding=0, stride=2)
        self.conv301 = nn.Conv2d(256 * expansion, 256 * expansion, kernel_size=3, padding=1)
        self.lnon301 = LNon()

        self.conv400 = nn.Conv2d(256 * expansion, 512 * expansion, kernel_size=1, padding=0, stride=2)
        self.conv401 = nn.Conv2d(512 * expansion, 512 * expansion, kernel_size=3, padding=1)
        self.lnon401 = LNon()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * expansion, 100)

    @th.compile
    def forward(self, x):
        x = self.conv000(x)
        x = self.lnon000(x)
        x = self.maxpool(x)

        x = self.conv100(x)
        x = self.conv101(x)
        x = self.lnon101(x)

        x = self.conv200(x)
        x = self.conv201(x)
        x = F.dropout2d(x, 0.25, training=self.training)
        x = self.lnon201(x)

        x = self.conv300(x)
        x = self.conv301(x)
        x = F.dropout2d(x, 0.25, training=self.training)
        x = self.lnon301(x)

        x = self.conv400(x)
        x = self.conv401(x)
        x = self.lnon401(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x


class Cifar5(CIFARModel):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 810, kernel_size=7, padding=3)
        self.fiol0 = LNon()
        self.conv1 = nn.Conv2d(810, 2430, kernel_size=3, padding=1)
        self.fiol1 = LNon()
        self.conv2 = nn.Conv2d(2430, 2430, kernel_size=3, padding=1)
        self.fiol2 = LNon()
        self.conv3 = nn.Conv2d(2430, 2430, kernel_size=3, padding=1)
        self.fiol3 = LNon()
        self.fc = nn.Linear(2430 * 16, 100)

    @th.compile
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


class Cifar4(CIFARModel):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 405, kernel_size=7, padding=3)
        self.fiol0 = LNon()
        self.conv1 = nn.Conv2d(405, 1215, kernel_size=3, padding=1)
        self.fiol1 = LNon()
        self.conv2 = nn.Conv2d(1215, 1215, kernel_size=3, padding=1)
        self.fiol2 = LNon()
        self.conv3 = nn.Conv2d(1215, 1215, kernel_size=3, padding=1)
        self.fiol3 = LNon()
        self.fc = nn.Linear(1215 * 16, 100)

    @th.compile
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


class Cifar3(CIFARModel):
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


class Cifar2(CIFARModel):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 1024, kernel_size=7, padding=3, bias=False)
        self.fiol0 = LNon()
        self.conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False)
        self.fiol1 = LNon()
        self.conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.fiol2 = LNon()
        self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False)
        self.fiol3 = LNon()
        self.fc = nn.Linear(256 * 16, 100)

    @th.compile
    def forward(self, x):
        x = self.conv0(x)
        x = F.dropout2d(x, 0.2, training=self.training)
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


class Cifar1(CIFARModel):
    def __init__(self):
        super().__init__()
        self.resnet = tv.models.resnet50(pretrained=False)
        self.resnet.num_classes = 100
        self.resnet.inplanes = 64
        self.resnet.conv1 = nn.Conv2d(3, self.resnet.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.bn1 = self.resnet._norm_layer(self.resnet.inplanes)
        self.resnet.fc = nn.Linear(512 * self.resnet.layer4[1].expansion, self.resnet.num_classes)

    @th.compile
    def forward(self, x):
        x = self.resnet(x)
        x = F.log_softmax(x, dim=1)
        return x


class Cifar0(CIFARModel):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 135, kernel_size=7, padding=3)
        self.fiol0 = LNon()
        self.conv1 = nn.Conv2d(135, 405, kernel_size=3, padding=1)
        self.fiol1 = LNon()
        self.conv2 = nn.Conv2d(405, 405, kernel_size=3, padding=1)
        self.fiol2 = LNon()
        self.conv3 = nn.Conv2d(405, 405, kernel_size=3, padding=1)
        self.fiol3 = LNon()
        self.fc = nn.Linear(405 * 16, 100)

    @th.compile
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
    return Cifar7()

