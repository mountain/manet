from torch import nn
from torchvision.ops import MLP

from manet.mac import Reshape
from manet.nn.model import MNISTModel

# Baseline model with ReLU activation function.
# A normal CNN as backbone model for all our tests.


class MNModel0(MNISTModel):
    def __init__(self):
        super().__init__()
        self.recognizer = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(5, 10, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 40, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            Reshape(40 * 3 * 3),
            MLP(40 * 9, [10]),
            Reshape(10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.recognizer(x)


def _model_():
    return MNModel0()
