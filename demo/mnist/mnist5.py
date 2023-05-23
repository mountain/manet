from torch import nn

from manet.mac import Reshape, MLP
from manet.aeg.flow import LearnableFunction
from manet.nn.model import MNISTModel

# A learnable non-linearity functions with the help of gradient formula from arithmetical expression geometry.
# The non-linearity is learned in an iterative system, and the gradient dispersal phenomenon is avoided.
#
# Result:
# The loss was dropped from 2.30, and the accuracy was better.
# accuracy = ?????

class MNModel5(MNISTModel):
    def __init__(self):
        super().__init__()
        self.recognizer = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, padding=2),
            LearnableFunction(),
            nn.MaxPool2d(2),
            nn.Conv2d(5, 15, kernel_size=5, padding=2),
            LearnableFunction(),
            nn.MaxPool2d(2),
            nn.Conv2d(15, 45, kernel_size=5, padding=2),
            LearnableFunction(),
            nn.MaxPool2d(2),
            nn.Conv2d(45, 135, kernel_size=3, padding=1),
            LearnableFunction(),
            Reshape(135 * 3 * 3),
            MLP(135 * 9, [10]),
            Reshape(10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.recognizer(x)


def _model_():
    return MNModel5()
