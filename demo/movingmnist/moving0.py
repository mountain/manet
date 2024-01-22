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
        self.dnsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=33 / 16, mode='nearest')

        self.conv0 = nn.Conv2d(10, 40, kernel_size=3, padding=1)
        self.conv01 = nn.Conv2d(10, 40, kernel_size=3, padding=1)
        self.lnon01 = LNon()
        self.conv02 = nn.Conv2d(10, 40, kernel_size=3, padding=1)
        self.lnon02 = LNon()
        self.lnon0 = LNon()

        self.conv1 = nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.lnon11 = LNon()
        self.conv12 = nn.Conv2d(40, 80, kernel_size=3, padding=1)
        self.lnon12 = LNon()
        self.lnon1 = LNon()

        self.conv2 = nn.Conv2d(80, 160, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(80, 160, kernel_size=3, padding=1)
        self.lnon21 = LNon()
        self.conv22 = nn.Conv2d(80, 160, kernel_size=3, padding=1)
        self.lnon22 = LNon()
        self.lnon2 = LNon()

        self.conv3 = nn.Conv2d(160, 320, kernel_size=3, padding=1)
        self.conv31 = nn.Conv2d(160, 320, kernel_size=3, padding=1)
        self.lnon31 = LNon()
        self.conv32 = nn.Conv2d(160, 320, kernel_size=3, padding=1)
        self.lnon32 = LNon()
        self.lnon3 = LNon()

        self.conv4 = nn.Conv2d(320, 320, kernel_size=1, padding=0)
        self.conv41 = nn.Conv2d(320, 320, kernel_size=1, padding=0)
        self.lnon41 = LNon()
        self.conv42 = nn.Conv2d(320, 320, kernel_size=1, padding=0)
        self.lnon42 = LNon()
        self.lnon4 = LNon()

        self.conv5 = nn.Conv2d(320, 320, kernel_size=1, padding=0)
        self.conv51 = nn.Conv2d(320, 320, kernel_size=1, padding=0)
        self.lnon51 = LNon()
        self.conv52 = nn.Conv2d(320, 320, kernel_size=1, padding=0)
        self.lnon52 = LNon()
        self.lnon5 = LNon()

        self.conv6 = nn.Conv2d(320, 320, kernel_size=1, padding=0)
        self.conv61 = nn.Conv2d(320, 320, kernel_size=1, padding=0)
        self.lnon61 = LNon()
        self.conv62 = nn.Conv2d(320, 320, kernel_size=1, padding=0)
        self.lnon62 = LNon()
        self.lnon6 = LNon()

        self.conv7 = nn.Conv2d(640, 320, kernel_size=1, padding=0)
        self.conv71 = nn.Conv2d(640, 320, kernel_size=1, padding=0)
        self.lnon71 = LNon()
        self.conv72 = nn.Conv2d(640, 320, kernel_size=1, padding=0)
        self.lnon72 = LNon()
        self.lnon7 = LNon()

        self.conv8 = nn.Conv2d(640, 320, kernel_size=1, padding=0)
        self.conv81 = nn.Conv2d(640, 320, kernel_size=1, padding=0)
        self.lnon81 = LNon()
        self.conv82 = nn.Conv2d(640, 320, kernel_size=1, padding=0)
        self.lnon82 = LNon()
        self.lnon8 = LNon()

        self.conv9 = nn.Conv2d(640, 160, kernel_size=3, padding=1)
        self.conv91 = nn.Conv2d(640, 160, kernel_size=3, padding=1)
        self.lnon91 = LNon()
        self.conv92 = nn.Conv2d(640, 160, kernel_size=3, padding=1)
        self.lnon92 = LNon()
        self.lnon9 = LNon()

        self.conva = nn.Conv2d(320, 80, kernel_size=3, padding=1)
        self.conva1 = nn.Conv2d(320, 80, kernel_size=3, padding=1)
        self.lnona1 = LNon()
        self.conva2 = nn.Conv2d(320, 80, kernel_size=3, padding=1)
        self.lnona2 = LNon()
        self.lnona = LNon()

        self.convb = nn.Conv2d(160, 40, kernel_size=3, padding=1)
        self.convb1 = nn.Conv2d(160, 40, kernel_size=3, padding=1)
        self.lnonb1 = LNon()
        self.convb2 = nn.Conv2d(160, 40, kernel_size=3, padding=1)
        self.lnonb2 = LNon()
        self.lnonb = LNon()

        self.convc = nn.Conv2d(80, 20, kernel_size=3, padding=1)
        self.convc1 = nn.Conv2d(80, 20, kernel_size=3, padding=1)
        self.lnonc1 = LNon()
        self.convc2 = nn.Conv2d(80, 20, kernel_size=3, padding=1)
        self.lnonc2 = LNon()
        self.lnonc = LNon()

    def training_step(self, train_batch, batch_idx):
        w = train_batch.view(-1, 20, 64, 64)
        x, y = w[:, :10], w[:, 10:]
        z = self.forward(x)
        loss = F.mse_loss(z, w)
        self.log('train_loss', loss, prog_bar=True)

        mse = F.mse_loss(recover(z[:, 10:]), recover(y))
        self.log('train_mse', mse, prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        w = val_batch.view(-1, 20, 64, 64)
        x, y = w[:, :10], w[:, 10:]
        z = self.forward(x)
        loss = F.mse_loss(z, w)
        self.log('val_loss', loss, prog_bar=True)

        mse = F.mse_loss(recover(z[:, 10:]), recover(y))
        self.log('val_mse', mse, prog_bar=True)

        self.labeled_loss += loss.item() * y.size()[0]
        self.counter += y.size()[0]

    def test_step(self, test_batch, batch_idx):
        w = test_batch.view(-1, 20, 64, 64)
        x, y = w[:, :10], w[:, 10:]
        x = x.view(-1, 10, 64, 64)
        z = self.forward(x)
        mse = F.mse_loss(recover(z[:, 10:]), recover(y))
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
    def downward_calc(self, dnsample, dropout, conv0, conv1, lnon1, conv2, lnon2, lnon0, x):
        y = dnsample(x)
        y = dropout(y)
        x = conv0(y)
        a = conv1(y)
        a = lnon1(a)
        b = conv2(y)
        b = lnon2(b)
        x = lnon0(a * x + b)
        return x

    @th.compile
    def upward_calc(self, upsample, conv0, conv1, lnon1, conv2, lnon2, lnon0, x, y):
        z = upsample(y)
        w = th.cat([z, x], dim=1)
        x = conv0(w)
        a = conv1(w)
        a = lnon1(a)
        b = conv2(w)
        b = lnon2(b)
        x = lnon0(a * x + b)
        return x

    def forward(self, x):
        x = x.view(-1, 10, 64, 64)
        x0 = th.ones_like(x[:, :, :, 0:1])
        x1 = th.ones_like(x[:, :, :, 0:1])
        x = th.cat([x0, x, x1], dim=-1)
        y0 = th.ones_like(x[:, :, 0:1, :])
        y1 = th.ones_like(x[:, :, 0:1, :])
        x = th.cat([y0, x, y1], dim=-2)
        x = x.view(-1, 10, 66, 66)

        x0 = self.conv0(x)
        a0 = self.conv01(x)
        a0 = self.lnon01(a0)
        b0 = self.conv02(x)
        b0 = self.lnon02(b0)
        x0 = self.lnon0(a0 * x0 + b0)

        # 66 -> 33
        x1 = self.downward_calc(self.dnsample, self.dropout, self.conv1, self.conv11, self.lnon11, self.conv12,
                                self.lnon12, self.lnon1, x0)
        # 33 -> 16
        x2 = self.downward_calc(self.dnsample, self.dropout, self.conv2, self.conv21, self.lnon21, self.conv22,
                                self.lnon22, self.lnon2, x1)
        # 16 -> 8
        x3 = self.downward_calc(self.dnsample, self.dropout, self.conv3, self.conv31, self.lnon31, self.conv32,
                                self.lnon32, self.lnon3, x2)
        # 8 -> 4
        x4 = self.downward_calc(self.dnsample, self.dropout, self.conv4, self.conv41, self.lnon41, self.conv42,
                                self.lnon42, self.lnon4, x3)
        # 4 -> 2
        x5 = self.downward_calc(self.dnsample, self.dropout, self.conv5, self.conv51, self.lnon51, self.conv52,
                                self.lnon52, self.lnon5, x4)
        # 2 -> 1
        x6 = self.downward_calc(self.dnsample, self.dropout, self.conv6, self.conv61, self.lnon61, self.conv62,
                                self.lnon62, self.lnon6, x5)

        # 1 -> 2
        x7 = self.upward_calc(self.upsample, self.conv7, self.conv71, self.lnon71, self.conv72, self.lnon72, self.lnon7,
                              x5, x6)
        # 2 -> 4
        x8 = self.upward_calc(self.upsample, self.conv8, self.conv81, self.lnon81, self.conv82, self.lnon82, self.lnon8,
                              x4, x7)
        # 4 -> 8
        x9 = self.upward_calc(self.upsample, self.conv9, self.conv91, self.lnon91, self.conv92, self.lnon92, self.lnon9,
                              x3, x8)
        # 8 -> 16
        xa = self.upward_calc(self.upsample, self.conva, self.conva1, self.lnona1, self.conva2, self.lnona2, self.lnona,
                              x2, x9)
        # 16 -> 33
        xb = self.upward_calc(self.upsample1, self.convb, self.convb1, self.lnonb1, self.convb2, self.lnonb2,
                              self.lnonb, x1, xa)
        # 33 -> 66
        xc = self.upward_calc(self.upsample, self.convc, self.convc1, self.lnonc1, self.convc2, self.lnonc2, self.lnonc,
                              x0, xb)

        x = xc[:, :, 1:-1, 1:-1]
        return x


def _model_():
    return Moving0()
