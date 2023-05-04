from torch import nn


class MNModel0(nn.Module):
    def __init__(self):
        super().__init__()
        self.recognizer = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 40, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(40, 80, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.recognizer(x)


def _model_():
    return MNModel0()
