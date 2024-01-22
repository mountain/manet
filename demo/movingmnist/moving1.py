import torch as th
import torch.nn.functional as F
import torch.nn as nn
import lightning as ltn

from torch import Tensor
from typing import TypeVar

U = TypeVar('U', bound='Unit')


class LNon(nn.Module):

    def __init__(self, points=11):
        super().__init__()
        self.points = points
        self.iscale = nn.Parameter(th.normal(0, 1, (1, 1, 1, 1)))
        self.oscale = nn.Parameter(th.normal(0, 1, (1, 1, 1, 1)))
        self.theta = th.linspace(-th.pi, th.pi, points)
        self.velocity = th.linspace(0, 3, points)
        self.weight = nn.Parameter(th.normal(0, 1, (points, points)))

    @th.compile
    def integral(self, param, index):
        return th.sum(param[index].view(-1, 1) * th.softmax(self.weight, dim=1)[index, :], dim=1)

    @th.compile
    def interplot(self, param, index):
        lmt = param.size(0) - 1

        p0 = index.floor().long()
        p1 = p0 + 1
        pos = index - p0
        p0 = p0.clamp(0, lmt)
        p1 = p1.clamp(0, lmt)

        v0 = self.integral(param, p0)
        v1 = self.integral(param, p1)

        return (1 - pos) * v0 + pos * v1

    @th.compile
    def forward(self: U, data: Tensor) -> Tensor:
        if self.theta.device != data.device:
            self.theta = self.theta.to(data.device)
            self.velocity = self.velocity.to(data.device)
        shape = data.size()
        data = (data - data.mean()) / data.std() * self.iscale
        data = data.flatten(0)

        theta = self.interplot(self.theta, th.sigmoid(data) * (self.points - 1))
        ds = self.interplot(self.velocity, th.abs(th.tanh(data) * (self.points - 1)))

        dx = ds * th.cos(theta)
        dy = ds * th.sin(theta)
        data = data * th.exp(dy) + dx

        data = (data - data.mean()) / data.std()
        return data.view(*shape) * self.oscale


def recover(x):
    return x * 51.070168 + 12.562058


class Moving0(ltn.LightningModule):
    def __init__(self):
        super().__init__()

        self.learning_rate = 1e-3
        self.counter = 0
        self.labeled_loss = 0
        self.dropout = nn.Dropout2d(0.2)

        self.conv0 = nn.Conv3d(1, 20, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv01 = nn.Conv3d(1, 20, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.lnon01 = LNon()
        self.conv02 = nn.Conv3d(1, 20, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.lnon02 = LNon()
        self.lnon0 = LNon()

        self.conv1 = nn.Conv3d(20, 20, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv11 = nn.Conv3d(20, 20, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.lnon11 = LNon()
        self.conv12 = nn.Conv3d(20, 20, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.lnon12 = LNon()
        self.lnon1 = LNon()

        self.conv2 = nn.Conv3d(20, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv21 = nn.Conv3d(20, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.lnon21 = LNon()
        self.conv22 = nn.Conv3d(20, 1, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.lnon22 = LNon()
        self.lnon2 = LNon()


    def training_step(self, train_batch, batch_idx):
        w = train_batch.view(-1, 1, 20, 64, 64)
        x, y = w[:, :, :10], w[:, :, 10:]
        z = self.forward(x)
        loss = F.mse_loss(z, w)
        self.log('train_loss', loss, prog_bar=True)

        mse = F.mse_loss(recover(z[:, :, 10:]), recover(y))
        self.log('train_mse', mse, prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        w = val_batch.view(-1, 1, 20, 64, 64)
        x, y = w[:, :, :10], w[:, :, 10:]
        z = self.forward(x)
        loss = F.mse_loss(z, w)
        self.log('val_loss', loss, prog_bar=True)

        mse = F.mse_loss(recover(z[:, :, 10:]), recover(y))
        self.log('val_mse', mse, prog_bar=True)

    def test_step(self, test_batch, batch_idx):
        w = test_batch.view(-1, 20, 64, 64)
        x, y = w[:, :, :10], w[:, :, 10:]
        x = x.view(-1, 1, 10, 64, 64)
        z = self.forward(x)
        mse = F.mse_loss(recover(z[:, :, 10:]), recover(y))
        self.log('test_mse', mse, prog_bar=True)

    def on_save_checkpoint(self, checkpoint) -> None:
        import glob, os

        loss = self.labeled_loss / self.counter
        record = '%2.5f-%03d.ckpt' % (loss, checkpoint['epoch'])
        fname = 'best-%s' % record
        with open(fname, 'bw') as f:
            th.save(checkpoint, f)
        for ix, ckpt in enumerate(sorted(glob.glob('best-*.ckpt'))):
            if ix > 5:
                os.unlink(ckpt)
        self.counter = 0
        self.labeled_loss = 0
        print()

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, 53)
        return [optimizer], [scheduler]

    @th.compile
    def block_calc(self, conv0, conv1, lnon1, conv2, lnon2, lnon0, x):
        y = conv0(x)
        a = conv1(x)
        a = lnon1(a)
        b = conv2(x)
        b = lnon2(b)
        return lnon0(a * y + b)

    def forward(self, x):
        x = x.view(-1, 1, 10, 64, 64)
        x = self.block_calc(self.conv0, self.conv01, self.lnon01, self.conv02, self.lnon02, self.lnon0, x)
        x = self.block_calc(self.conv1, self.conv11, self.lnon11, self.conv12, self.lnon12, self.lnon1, x)
        x = self.block_calc(self.conv2, self.conv21, self.lnon21, self.conv22, self.lnon22, self.lnon2, x)
        return x


def _model_():
    return Moving0()
