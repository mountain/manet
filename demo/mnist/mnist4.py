from torch import nn
from torchvision.ops import MLP

from manet.mac import Reshape
from manet.expt.spline import SplineFunction
from manet.nn.model import MNISTModel


# Spline functions can be used to approximate any continuous function.
# but in an iterative system, the non-linearity can not be learned for the dispersal of the gradient.


class MNModel4(MNISTModel):
    def __init__(self):
        super().__init__()
        self.recognizer = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            SplineFunction(),
            nn.Conv2d(5, 10, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            SplineFunction(),
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            SplineFunction(),
            nn.Conv2d(20, 40, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            SplineFunction(),
            Reshape(40 * 3 * 3),
            MLP(40 * 9, [10]),
            Reshape(10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.recognizer(x)


def _model_():
    return MNModel4()
