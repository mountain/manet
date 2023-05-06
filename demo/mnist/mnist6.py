import torch as th
import lightning as pl

from torch import nn
from torch.nn import functional as F

from manet.mac import Reshape
from torchvision.ops import MLP


class MNModel4(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.encoder = nn.Sequential(
            Reshape(28 * 28),
            MLP(28 * 28, [240]),
        )
        self.learner = MLP(80 * 2 * 3, [320 * 3, 640 * 3, 1280 * 3, 2560 * 3, 1280 * 3, 80 * 8 * 3])
        self.decoder = nn.Sequential(
            MLP(80 * 6, [10]),
            nn.LogSoftmax(dim=1)
        )

    def ulearn(self, learner, inputs):
        output = th.zeros_like(inputs)
        context = th.zeros_like(inputs)
        dc = th.zeros_like(inputs)
        do = th.zeros_like(inputs)
        for _ in range(3):
            state = th.sigmoid(learner(th.cat((context, inputs), dim=1))).view(-1, 8, 240)
            p, r, t, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
            q, s, u, w = state[:, 4], state[:, 5], state[:, 6], state[:, 7]
            p, q = 4 * p, 4 * q
            self.log('pmax', p.max().item(), prog_bar=True)
            self.log('qmax', q.max().item(), prog_bar=True)
            self.log('pmean', p.mean().item(), prog_bar=True)
            self.log('qmean', q.mean().item(), prog_bar=True)

            do = th.fmod((1 - do) * do * p + inputs * v, 1) * r + do * (1 - r)
            do = do * t * (1 - r) + do * r
            output = output + do

            dc = th.fmod((1 - dc) * dc * q + inputs * w, 1) * s + dc * (1 - s)
            dc = dc * u * (1 - s) + dc * s
            context = context + dc

        return th.cat((output, context), dim=1)

    def forward(self, x):
        inputs = self.encoder(x)
        # output = self.ulearn(self.learner, inputs).flatten(1)
        output = th.cat((inputs, inputs), dim=1)
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
    return MNModel4()
