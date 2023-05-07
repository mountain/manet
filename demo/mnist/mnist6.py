import torch as th
import lightning as pl

from torch import nn
from torch.nn import functional as F

from manet.mac import Reshape
# from torchvision.ops import MLP
from manet.mac import MLP


class MNModel4(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.hidden = 49 * 3
        self.encoder = nn.Sequential(
            Reshape(28 * 28),
            MLP(28 * 28, [self.hidden]),
        )
        # self.learner = MLP(self.hidden * 2, [self.hidden * 8])
        self.decoder = nn.Sequential(
            MLP(self.hidden, [10]),
            nn.LogSoftmax(dim=1)
        )

    def ulearn(self, learner, inputs):
        output = th.zeros_like(inputs)
        context = th.zeros_like(inputs)
        dc = th.zeros_like(inputs)
        do = th.zeros_like(inputs)
        for _ in range(3):
            state = th.sigmoid(learner(th.cat((context, inputs), dim=1))).view(-1, 8, self.hidden)
            p, r, t, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
            q, s, u, w = state[:, 4], state[:, 5], state[:, 6], state[:, 7]
            p, q = 4 * p, 4 * q

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
        # output = th.cat((inputs, inputs), dim=1)
        output = inputs
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
        self.labeled_loss = loss.item()
        self.labeled_correct = correct.item()

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x = x.view(-1, 1, 28, 28)
        z = self(x)
        pred = z.data.max(1, keepdim=True)[1]
        correct = pred.eq(y.data.view_as(pred)).sum() / y.size()[0]
        self.log('correct', correct, prog_bar=True)

    def on_save_checkpoint(self, checkpoint) -> None:
        import pickle, glob, os

        record = '%2.5f-%03d-%1.5f.ckpt' % (self.labeled_loss, checkpoint['epoch'], self.labeled_correct)
        fname = 'best-%s' % record
        with open(fname, 'bw') as f:
            pickle.dump(checkpoint, f)
        for ix, ckpt in enumerate(sorted(glob.glob('best-*.ckpt'))):
            if ix > 5:
                os.unlink(ckpt)


def _model_():
    return MNModel4()
