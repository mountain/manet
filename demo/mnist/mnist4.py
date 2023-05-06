import torch as th
import lightning as pl

from torch import nn
from torch.nn import functional as F

from manet.mac import Reshape
# from manet.mac import MLP
from torchvision.ops import MLP


class MNModel4(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            MLP(1, [1]),
            Reshape(10, 14, 14),
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            MLP(1, [1]),
            Reshape(20, 7, 7),
            nn.Conv2d(20, 40, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            MLP(1, [1]),
            Reshape(40, 3, 3),
            nn.Conv2d(40, 80, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            MLP(1, [1]),
            Reshape(80, 1, 1),
            nn.Flatten(),
        )
        self.ulearner = MLP(80, [160, 40, 80 * 3])
        self.decoder = nn.Sequential(
            MLP(80, [40, 20, 10]),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        inputs = self.encoder(x)

        lastr = th.ones_like(inputs)
        output = th.zeros_like(inputs)

        do = th.zeros_like(inputs)
        for _ in range(6):
            state = th.sigmoid(self.ulearner(inputs)).view(-1, 3, 80)
            p, r, t = state[:, 0], state[:, 1], state[:, 2]
            p = 4 * p

            r, lastr = r * lastr, r

            do = th.fmod((1 - do) * do * p + inputs, 1) * r + do * (1 - r)
            do = do * t * (1 - r) + do * r
            output = output + do

        return self.decoder(output)

    def configure_optimizers(self):
        return th.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(-1, 1, 28, 28)
        z = self(x)
        loss = F.nll_loss(z, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view(-1, 1, 28, 28)
        z = self(x)
        loss = F.nll_loss(z, y)
        self.log('val_loss', loss, prog_bar=True)
        pred = z.data.max(1, keepdim=True)[1]
        correct = pred.eq(y.data.view_as(pred)).sum() / y.size()[0]
        self.log('correct', correct, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x = x.view(-1, 1, 28, 28)
        z = self(x)
        pred = z.data.max(1, keepdim=True)[1]
        correct = pred.eq(y.data.view_as(pred)).sum() / y.size()[0]
        self.log('correct', correct, prog_bar=True)


def _model_():
    return MNModel2()
