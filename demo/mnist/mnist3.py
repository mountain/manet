import torch

from torch import nn
from torch.nn import functional as F

from manet.mac import MLP, Reshape
from manet.conv import Conv2d


class MNModel3(nn.Module):
    def __init__(self):
        super().__init__()
        self.recognizer = nn.Sequential(
            Conv2d(1, 10, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            Reshape(10, 14, 14),
            Conv2d(10, 20, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            Reshape(20, 7, 7),
            Conv2d(20, 40, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            Reshape(40, 3, 3),
            Conv2d(40, 80, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            Reshape(80, 1, 1),
            nn.Flatten(),
            MLP(80, [40, 20, 10]),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.recognizer(x)


def _model_():
    return MNModel3()
