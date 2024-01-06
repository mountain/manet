import torch as th
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as ltn

from torch import Tensor
from typing import TypeVar, Tuple


U = TypeVar('U', bound='Unit')


class LNon(nn.Module):
    def __init__(self: U,
                 steps: int = 3,
                 length: float = 0.33333,
                 points: int = 5,
                 ) -> None:
        super().__init__()

        self.num_steps = steps
        self.step_length = length
        self.num_points = points

        self.theta = nn.Parameter(
            th.linspace(0, 2 * th.pi, points).view(1, 1, points)
        )
        self.velocity = nn.Parameter(
            th.linspace(0, 1, points).view(1, 1, points)
        )
        self.channel_transform = nn.Parameter(
            th.normal(0, 1, (1, 1, 1))
        )
        self.spatio_transform = nn.Parameter(
            th.normal(0, 1, (1, 1, 1))
        )

    def accessor(self: U,
                 data: Tensor,
                 ) -> Tuple[Tensor, Tensor, Tensor]:

        index = th.sigmoid(data) * self.num_points

        bgn = index.floor().long()
        bgn = bgn * (bgn >= 0)
        bgn = bgn * (bgn <= self.num_points - 2) + (bgn - 1) * (bgn > self.num_points - 2)
        bgn = bgn * (bgn <= self.num_points - 2) + (bgn - 1) * (bgn > self.num_points - 2)
        end = bgn + 1

        return index, bgn, end

    def access(self: U,
               memory: Tensor,
               accessor: Tuple[Tensor, Tensor, Tensor]
               ) -> Tensor:

        index, bgn, end = accessor
        pos = index - bgn
        memory = memory.flatten(0)
        return (1 - pos) * memory[bgn] + pos * memory[end]

    def step(self: U,
             data: Tensor
             ) -> Tensor:

        accessor = self.accessor(data)
        theta = self.access(self.theta, accessor)
        velo = self.access(self.velocity, accessor)

        # by the flow equation of the arithmetic expression geometry
        ds = velo * self.step_length
        dx = ds * th.cos(theta)
        dy = ds * th.sin(theta)
        val = data * (1 + dy) + dx
        return val

    def forward(self: U,
                data: Tensor
                ) -> Tensor:
        shape = data.size()
        data = data.flatten(1)
        data = data.contiguous()
        data = data.view(-1, 1, 1)

        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.channel_transform)
        # data = data * self.channel_transform
        data = data.view(-1, 1, 1)

        for ix in range(self.num_steps):
            data = self.step(data)

        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.spatio_transform)
        # data = data * self.spatio_transform
        data = data.view(-1, 1, 1)

        return data.view(*shape)


class Moving0(ltn.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 1e-3
        self.counter = 0
        self.labeled_loss = 0
        self.dnsample = nn.MaxPool2d(2)
        self.upsample0 = nn.Upsample(scale_factor=64 / 32, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=32 / 16, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=16 / 8, mode='nearest')
        self.upsample3 = nn.Upsample(scale_factor=8 / 4, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=4 / 2, mode='nearest')
        self.upsample5 = nn.Upsample(scale_factor=2 / 1, mode='nearest')

        self.conv0 = nn.Conv2d(10, 20, kernel_size=7, padding=3)
        self.lnon0 = LNon(steps=1, length=1, points=8)
        self.conv1 = nn.Conv2d(20, 20, kernel_size=3, padding=1)
        self.lnon1 = LNon(steps=1, length=1, points=8)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=1, padding=0)
        self.lnon2 = LNon(steps=1, length=1, points=8)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=1, padding=0)
        self.lnon3 = LNon(steps=1, length=1, points=8)
        self.conv4 = nn.Conv2d(20, 20, kernel_size=1, padding=0)
        self.lnon4 = LNon(steps=1, length=1, points=8)
        self.conv5 = nn.Conv2d(20, 20, kernel_size=1, padding=0)
        self.lnon5 = LNon(steps=1, length=1, points=8)
        self.conv6 = nn.Conv2d(20, 20, kernel_size=1, padding=0)
        self.lnon6 = LNon(steps=1, length=1, points=8)

        self.conv7 = nn.Conv2d(40, 20, kernel_size=1, padding=0)
        self.lnon7 = LNon(steps=1, length=1, points=8)
        self.conv8 = nn.Conv2d(40, 20, kernel_size=1, padding=0)
        self.lnon8 = LNon(steps=1, length=1, points=8)
        self.conv9 = nn.Conv2d(40, 20, kernel_size=1, padding=0)
        self.lnon9 = LNon(steps=1, length=1, points=8)
        self.conva = nn.Conv2d(40, 20, kernel_size=1, padding=0)
        self.lnona = LNon(steps=1, length=1, points=8)
        self.convb = nn.Conv2d(40, 20, kernel_size=1, padding=0)
        self.lnonb = LNon(steps=1, length=1, points=8)
        self.convc = nn.Conv2d(40, 10, kernel_size=1, padding=0)
        self.lnonc = LNon(steps=1, length=1, points=8)

    def training_step(self, train_batch, batch_idx):
        batch = train_batch.view(-1, 20, 64, 64)
        x, y = batch[:, :10], batch[:, 10:]
        x = x.view(-1, 10, 64, 64)
        z = self.forward(x)
        loss = F.mse_loss(z, y)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        batch = val_batch.view(-1, 20, 64, 64)
        x, y = batch[:, :10], batch[:, 10:]
        x = x.view(-1, 10, 64, 64)

        z = self.forward(x)
        loss = F.mse_loss(z, y)
        self.log('val_loss', loss, prog_bar=True)

        self.labeled_loss += loss.item() * y.size()[0]
        self.counter += y.size()[0]

    def test_step(self, test_batch, batch_idx):
        batch = test_batch.view(-1, 20, 64, 64)
        x, y = batch[:, :10], batch[:, 10:]
        x = x.view(-1, 10, 64, 64)
        z = self.forward(x)
        loss = F.mse_loss(z, y)
        self.log('test_loss', loss, prog_bar=True)

    def on_save_checkpoint(self, checkpoint) -> None:
        import glob, os

        loss = self.labeled_loss / self.counter
        record = '%2.5f-%03d.ckpt' % (loss, checkpoint['epoch'])
        fname = 'best-%s' % record
        with open(fname, 'bw') as f:
            th.save(checkpoint, f)
        for ix, ckpt in enumerate(sorted(glob.glob('best-*.ckpt'), reverse=True)):
            if ix > 5:
                os.unlink(ckpt)
        self.counter = 0
        self.labeled_loss = 0

    def configure_optimizers(self):
        return [th.optim.Adam(self.parameters(), lr=self.learning_rate)]

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.lnon0(x0)
        x1 = self.dnsample(x0)  # 64 -> 32
        x1 = self.conv1(x1)
        x1 = self.lnon1(x1)
        x2 = self.dnsample(x1)  # 32 -> 16
        x2 = self.conv2(x2)
        x2 = self.lnon2(x2)
        x3 = self.dnsample(x2)  # 16 -> 8
        x3 = self.conv3(x3)
        x3 = self.lnon3(x3)
        x4 = self.dnsample(x3)  # 8 -> 4
        x4 = self.conv4(x4)
        x4 = self.lnon4(x4)
        x5 = self.dnsample(x4)  # 4 -> 2
        x5 = self.conv5(x5)
        x5 = self.lnon5(x5)
        x6 = self.dnsample(x5)  # 2 -> 1
        x6 = self.conv6(x6)
        x6 = self.lnon6(x6)

        x7 = self.upsample5(x6)
        x7 = th.cat([x7, x5], dim=1)
        x7 = self.conv7(x7)
        x7 = self.lnon7(x7)
        x8 = self.upsample4(x7)
        x8 = th.cat([x8, x4], dim=1)
        x8 = self.conv8(x8)
        x8 = self.lnon8(x8)
        x9 = self.upsample3(x8)
        x9 = th.cat([x9, x3], dim=1)
        x9 = self.conv9(x9)
        x9 = self.lnon9(x9)
        xa = self.upsample2(x9)
        xa = th.cat([xa, x2], dim=1)
        xa = self.conva(xa)
        xa = self.lnona(xa)
        xb = self.upsample1(xa)
        xb = th.cat([xb, x1], dim=1)
        xb = self.convb(xb)
        xb = self.lnonb(xb)
        xc = self.upsample0(xb)
        xc = th.cat([xc, x0], dim=1)
        xc = self.convc(xc)
        xc = self.lnonc(xc)

        return xc


def _model_():
    return Moving0()
