from torch import nn
from torchvision.ops import MLP

from manet.mac import Reshape
from manet.expt.logistic import LogisticFunction
from manet.nn.model import MNISTModel

# Logistic map exhibits chaotic behavior for p > 3.56995
# Here we check the behavior of the logistic map for p = 3.5
# and compare it with the behavior of the ReLU function.

class MNModel1(MNISTModel):
    def __init__(self):
        super().__init__()
        self.recognizer = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            LogisticFunction(p=3),
            nn.Conv2d(5, 10, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            LogisticFunction(p=3),
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            LogisticFunction(p=3),
            nn.Conv2d(20, 40, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            LogisticFunction(p=3),
            Reshape(40 * 3 * 3),
            MLP(40 * 9, [10]),
            Reshape(10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.recognizer(x)


def _model_():
    return MNModel1()
