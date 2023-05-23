from torch import nn

from manet.mac import Reshape, MLP
from manet.aeg.flow import LearnableFunction
from manet.nn.model import MNISTModel

# A learnable non-linearity functions with the help of gradient formula from arithmetical expression geometry.
# The non-linearity is learned in an iterative system, and the gradient dispersal phenomenon is avoided.
# We change the backbone to UNet.


class MNModel6(MNISTModel):
    def __init__(self):
        super().__init__()
        self.recognizer = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            LearnableFunction(),
            nn.Conv2d(5, 10, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            LearnableFunction(),
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            LearnableFunction(),
            nn.Conv2d(20, 40, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            LearnableFunction(),
            Reshape(40 * 3 * 3),
            MLP(40 * 9, [10]),
            Reshape(10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.recognizer(x)


def _model_():
    return MNModel6()
