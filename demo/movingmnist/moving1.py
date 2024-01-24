import torch as th
import torch.nn.functional as F
import torch.nn as nn
import lightning as ltn

from torch import Tensor


class LNon(nn.Module):

    def __init__(self, points=11):
        super().__init__()
        self.points = points
        self.iscale = nn.Parameter(th.normal(0, 1, (1, 1, 1, 1)))
        self.oscale = nn.Parameter(th.normal(0, 1, (1, 1, 1, 1)))
        self.theta = th.linspace(-th.pi, th.pi, points)
        self.velocity = th.linspace(0, th.e, points)
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
    def forward(self, data: Tensor) -> Tensor:
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


class Moving1(ltn.LightningModule):
    def __init__(self):
        super().__init__()

        self.learning_rate = 1e-3
        self.labeled_counter = 0
        self.labeled_mse = 0
        self.refered_counter = 0
        self.refered_mse = 0

        self.dnsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.ch = [4, 8, 16, 32, 64, 128]

        self.conv0 = nn.Conv2d(2, self.ch[0], kernel_size=7, padding=3, bias=False)  # 64
        self.lnon0 = nn.LeakyReLU(0.2)

        self.conv1 = nn.Conv2d(self.ch[0], self.ch[1], kernel_size=3, padding=1, bias=False)  # 32
        self.lnon1 = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv2d(self.ch[1], self.ch[2], kernel_size=3, padding=1, bias=False)  # 16
        self.lnon2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(self.ch[2], self.ch[3], kernel_size=3, padding=1, bias=False)  # 8
        self.lnon3 = nn.LeakyReLU(0.2)

        self.conv4 = nn.Conv2d(self.ch[3], self.ch[4], kernel_size=3, padding=1, bias=False)  # 4
        self.lnon4 = nn.LeakyReLU(0.2)

        self.conv5 = nn.Conv2d(self.ch[4], self.ch[5], kernel_size=1, padding=0, bias=False)  # 2
        self.lnon5 = nn.LeakyReLU(0.2)

        self.conv6 = nn.Conv2d(self.ch[5], self.ch[5], kernel_size=1, padding=0, bias=False)  # 1
        self.lnon6 = nn.LeakyReLU(0.2)

        self.conv7 = nn.Conv2d(self.ch[5] + self.ch[5], self.ch[5], kernel_size=1, padding=0, bias=False)  # 2
        self.lnon7 = nn.LeakyReLU(0.2)

        self.conv8 = nn.Conv2d(self.ch[5] + self.ch[4], self.ch[4], kernel_size=3, padding=1, bias=False)  # 4
        self.lnon8 = nn.LeakyReLU(0.2)

        self.conv9 = nn.Conv2d(self.ch[4] + self.ch[3], self.ch[3], kernel_size=3, padding=1, bias=False)  # 8
        self.lnon9 = nn.LeakyReLU(0.2)

        self.conva = nn.Conv2d(self.ch[3] + self.ch[2], self.ch[2], kernel_size=3, padding=1, bias=False)  # 16
        self.lnona = nn.LeakyReLU(0.2)

        self.convb = nn.Conv2d(self.ch[2] + self.ch[1], self.ch[1], kernel_size=3, padding=1, bias=False)  # 32
        self.lnonb = nn.LeakyReLU(0.2)

        self.convc = nn.Conv2d(self.ch[1] + self.ch[0], 2, kernel_size=3, padding=1, bias=False)  # 64
        self.lnonc = nn.LeakyReLU(0.2)

        self.conv_1_0 = nn.Conv2d(self.ch[5] * 9, 2 * self.ch[5] * 9, bias=False, kernel_size=1, padding=0)
        self.lnonz6_0 = LNon()
        self.conv_1_1 = nn.Conv2d(2 * self.ch[5] * 9, self.ch[5] * 9, kernel_size=1, padding=0)

        self.conv_1_2 = nn.Conv2d(self.ch[5] * 2, self.ch[5] * 4, bias=False, kernel_size=1, padding=0)
        self.lnonz6_1 = LNon()
        self.conv_1_3 = nn.Conv2d(self.ch[5] * 4, self.ch[5] * 1, kernel_size=1, padding=0)

        self.conv_2_0 = nn.Conv2d(self.ch[5] * 9, 2 * self.ch[5] * 9, bias=False, kernel_size=1, padding=0)
        self.lnonz5_0 = LNon()
        self.conv_2_1 = nn.Conv2d(2 * self.ch[5] * 9, self.ch[5] * 9, kernel_size=1, padding=0)

        self.conv_2_2 = nn.Conv2d(self.ch[5] * 2, self.ch[5] * 4, bias=False, kernel_size=1, padding=0)
        self.lnonz5_1 = LNon()
        self.conv_2_3 = nn.Conv2d(self.ch[5] * 4, self.ch[5] * 1, kernel_size=1, padding=0)

        self.conv_4_0 = nn.Conv2d(self.ch[4] * 9, 2 * self.ch[4] * 9, bias=False, kernel_size=3, padding=1)
        self.lnonz4_0 = LNon()
        self.conv_4_1 = nn.Conv2d(2 * self.ch[4] * 9, self.ch[4] * 9, kernel_size=3, padding=1)

        self.conv_4_2 = nn.Conv2d(self.ch[4] * 2, self.ch[4] * 4, bias=False, kernel_size=3, padding=1)
        self.lnonz4_1 = LNon()
        self.conv_4_3 = nn.Conv2d(self.ch[4] * 4, self.ch[4] * 1, kernel_size=3, padding=1)

        self.conv_8_0 = nn.Conv2d(self.ch[3] * 9, 2 * self.ch[3] * 9, bias=False, kernel_size=3, padding=1)
        self.lnonz3_0 = LNon()
        self.conv_8_1 = nn.Conv2d(2 * self.ch[3] * 9, self.ch[3] * 9, kernel_size=3, padding=1)

        self.conv_8_2 = nn.Conv2d(self.ch[3] * 2, self.ch[3] * 4, bias=False, kernel_size=3, padding=1)
        self.lnonz3_1 = LNon()
        self.conv_8_3 = nn.Conv2d(self.ch[3] * 4, self.ch[3] * 1, kernel_size=3, padding=1)

        self.conv_16_0 = nn.Conv2d(self.ch[2] * 9, 2 * self.ch[2] * 9, bias=False, kernel_size=3, padding=1)
        self.lnonz2_0 = LNon()
        self.conv_16_1 = nn.Conv2d(2 * self.ch[2] * 9, self.ch[2] * 9, kernel_size=3, padding=1)

        self.conv_16_2 = nn.Conv2d(self.ch[2] * 2, self.ch[2] * 4, bias=False, kernel_size=3, padding=1)
        self.lnonz2_1 = LNon()
        self.conv_16_3 = nn.Conv2d(self.ch[2] * 4, self.ch[2] * 1, kernel_size=3, padding=1)

        self.conv_32_0 = nn.Conv2d(self.ch[1] * 9, 2 * self.ch[1] * 9, bias=False, kernel_size=3, padding=1)
        self.lnonz1_0 = LNon()
        self.conv_32_1 = nn.Conv2d(2 * self.ch[1] * 9, self.ch[1] * 9, kernel_size=3, padding=1)

        self.conv_32_2 = nn.Conv2d(self.ch[1] * 2, self.ch[1] * 4, bias=False, kernel_size=3, padding=1)
        self.lnonz1_1 = LNon()
        self.conv_32_3 = nn.Conv2d(self.ch[1] * 4, self.ch[1] * 1, kernel_size=3, padding=1)

        self.conv_64_0 = nn.Conv2d(self.ch[0] * 9, 2 * self.ch[0] * 9, bias=False, kernel_size=3, padding=1)
        self.lnonz0_0 = LNon()
        self.conv_64_1 = nn.Conv2d(2 * self.ch[0] * 9, self.ch[0] * 9, kernel_size=3, padding=1)

        self.conv_64_2 = nn.Conv2d(self.ch[0] * 2, self.ch[0] * 4, bias=False, kernel_size=3, padding=1)
        self.lnonz0_1 = LNon()
        self.conv_64_3 = nn.Conv2d(self.ch[0] * 4, self.ch[0] * 1, kernel_size=3, padding=1)

    def training_step(self, train_batch, batch_idx):
        w = train_batch.view(-1, 20, 64, 64)
        x, y = w[:, :10], w[:, 10:]
        z = self.forward(x)
        loss = F.mse_loss(z, w)
        self.log('train_loss', loss, prog_bar=True)

        mse = F.mse_loss(recover(z[:, 10:]), recover(y))
        self.log('train_mse', mse, prog_bar=True)

        self.refered_mse += mse.item() * y.size()[0]
        self.refered_counter += y.size()[0]

        return loss

    def validation_step(self, val_batch, batch_idx):
        w = val_batch.view(-1, 20, 64, 64)
        x, y = w[:, :10], w[:, 10:]
        z = self.forward(x)
        loss = F.mse_loss(z, w)
        self.log('val_loss', loss, prog_bar=True)

        mse = F.mse_loss(recover(z[:, 10:]), recover(y))
        self.log('val_mse', mse, prog_bar=True)

        self.labeled_mse += mse.item() * y.size()[0]
        self.labeled_counter += y.size()[0]

    def test_step(self, test_batch, batch_idx):
        w = test_batch.view(-1, 20, 64, 64)
        x, y = w[:, :10], w[:, 10:]
        x = x.view(-1, 10, 64, 64)
        z = self.forward(x)
        mse = F.mse_loss(recover(z[:, 10:]), recover(y))
        self.log('test_mse', mse, prog_bar=True)

    def on_save_checkpoint(self, checkpoint) -> None:
        import glob, os

        labeled_loss = self.labeled_mse / self.labeled_counter
        refered_loss = self.refered_mse / self.refered_counter
        record = '%03.5f-%03.5f-%03d.ckpt' % (labeled_loss, refered_loss, checkpoint['epoch'])
        fname = 'best-%s' % record
        with open(fname, 'bw') as f:
            th.save(checkpoint, f)
        for ix, ckpt in enumerate(sorted(glob.glob('best-*.ckpt'))):
            if ix > 5:
                os.unlink(ckpt)
        self.labeled_counter = 0
        self.labeled_mse = 0
        self.refered_counter = 0
        self.refered_mse = 0
        print()

    def configure_optimizers(self):
        optimizer = th.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, 53)
        return [optimizer], [scheduler]

    @th.compile
    def downward_calc(self, dnsample, conv0, lnon0, x):
        y = dnsample(x)
        x = conv0(y)
        x = lnon0(x)
        return x

    @th.compile
    def upward_calc(self, upsample, conv0, lnon0, x, y):
        z = upsample(y)
        w = th.cat([z, x], dim=1)
        x = conv0(w)
        x = lnon0(x)
        return x

    @th.compile
    def downward(self, x):
        x = x.view(-1, 2, 64, 64)
        x0 = self.conv0(x)
        x0 = self.lnon0(x0)
        x1 = self.downward_calc(self.dnsample, self.conv1, self.lnon1, x0)  # 32
        x2 = self.downward_calc(self.dnsample, self.conv2, self.lnon2, x1)  # 16
        x3 = self.downward_calc(self.dnsample, self.conv3, self.lnon3, x2)  # 8
        x4 = self.downward_calc(self.dnsample, self.conv4, self.lnon4, x3)  # 4
        x5 = self.downward_calc(self.dnsample, self.conv5, self.lnon5, x4)  # 2
        x6 = self.downward_calc(self.dnsample, self.conv6, self.lnon6, x5)  # 1
        return x0, x1, x2, x3, x4, x5, x6

    @th.compile
    def upward(self, x0, x1, x2, x3, x4, x5, x6):
        x7 = self.upward_calc(self.upsample, self.conv7, self.lnon7, x5, x6)
        x8 = self.upward_calc(self.upsample, self.conv8, self.lnon8, x4, x7)
        x9 = self.upward_calc(self.upsample, self.conv9, self.lnon9, x3, x8)
        xa = self.upward_calc(self.upsample, self.conva, self.lnona, x2, x9)
        xb = self.upward_calc(self.upsample, self.convb, self.lnonb, x1, xa)
        xc = self.upward_calc(self.upsample, self.convc, self.lnonc, x0, xb)
        return xc

    @th.compile
    def evolve6(self, rep0, rep1, rep2, rep3, rep4, rep5, rep6, rep7, rep8):
        rep = th.cat([rep0, rep1, rep2, rep3, rep4, rep5, rep6, rep7, rep8], dim=1).view(-1, self.ch[5] * 9, 1, 1)
        pred = self.conv_1_1(self.lnonz6_0(self.conv_1_0(rep))).view(-1, self.ch[5] * 9, 1, 1)
        return (pred[:, self.ch[5] * i:self.ch[5] * (i + 1)] for i in range(9))

    @th.compile
    def evolve5(self, rep0, rep1, rep2, rep3, rep4, rep5, rep6, rep7, rep8):
        rep = th.cat([rep0, rep1, rep2, rep3, rep4, rep5, rep6, rep7, rep8], dim=1).view(-1, self.ch[5] * 9, 2, 2)
        pred = self.conv_2_1(self.lnonz5_0(self.conv_2_0(rep))).view(-1, self.ch[5] * 9, 2, 2)
        return (pred[:, self.ch[5] * i:self.ch[5] * (i + 1)] for i in range(9))

    @th.compile
    def evolve4(self, rep0, rep1, rep2, rep3, rep4, rep5, rep6, rep7, rep8):
        rep = th.cat([rep0, rep1, rep2, rep3, rep4, rep5, rep6, rep7, rep8], dim=1).view(-1, self.ch[4] * 9, 4, 4)
        pred = self.conv_4_1(self.lnonz4_0(self.conv_4_0(rep))).view(-1, self.ch[4] * 9, 4, 4)
        return (pred[:, self.ch[4] * i:self.ch[4] * (i + 1)] for i in range(9))

    @th.compile
    def evolve3(self, rep0, rep1, rep2, rep3, rep4, rep5, rep6, rep7, rep8):
        rep = th.cat([rep0, rep1, rep2, rep3, rep4, rep5, rep6, rep7, rep8], dim=1).view(-1, self.ch[3] * 9, 8, 8)
        pred = self.conv_8_1(self.lnonz3_0(self.conv_8_0(rep))).view(-1, self.ch[3] * 9, 8, 8)
        return (pred[:, self.ch[3] * i:self.ch[3] * (i + 1)] for i in range(9))

    @th.compile
    def evolve2(self, rep0, rep1, rep2, rep3, rep4, rep5, rep6, rep7, rep8):
        rep = th.cat([rep0, rep1, rep2, rep3, rep4, rep5, rep6, rep7, rep8], dim=1).view(-1, self.ch[2] * 9, 16, 16)
        pred = self.conv_16_1(self.lnonz2_0(self.conv_16_0(rep))).view(-1, self.ch[2] * 9, 16, 16)
        return (pred[:, self.ch[2] * i:self.ch[2] * (i + 1)] for i in range(9))

    @th.compile
    def evolve1(self, rep0, rep1, rep2, rep3, rep4, rep5, rep6, rep7, rep8):
        rep = th.cat([rep0, rep1, rep2, rep3, rep4, rep5, rep6, rep7, rep8], dim=1).view(-1, self.ch[1] * 9, 32, 32)
        pred = self.conv_32_1(self.lnonz1_0(self.conv_32_0(rep))).view(-1, self.ch[1] * 9, 32, 32)
        return (pred[:, self.ch[1] * i:self.ch[1] * (i + 1)] for i in range(9))

    @th.compile
    def evolve0(self, rep0, rep1, rep2, rep3, rep4, rep5, rep6, rep7, rep8):
        rep = th.cat([rep0, rep1, rep2, rep3, rep4, rep5, rep6, rep7, rep8], dim=1).view(-1, self.ch[0] * 9, 64, 64)
        pred = self.conv_64_1(self.lnonz0_0(self.conv_64_0(rep))).view(-1, self.ch[0] * 9, 64, 64)
        return (pred[:, self.ch[0] * i:self.ch[0] * (i + 1)] for i in range(9))

    @th.compile
    def merge6(self, rep0, rep1):
        rep = th.cat([rep0, rep1], dim=1).view(-1, self.ch[5] * 2, 1, 1)
        pred = self.conv_1_3(self.lnonz6_1(self.conv_1_2(rep))).view(-1, self.ch[5], 1, 1)
        return pred

    @th.compile
    def merge5(self, rep0, rep1):
        rep = th.cat([rep0, rep1], dim=1).view(-1, self.ch[5] * 2, 2, 2)
        pred = self.conv_2_3(self.lnonz5_1(self.conv_2_2(rep))).view(-1, self.ch[5], 2, 2)
        return pred

    @th.compile
    def merge4(self, rep0, rep1):
        rep = th.cat([rep0, rep1], dim=1).view(-1, self.ch[4] * 2, 4, 4)
        pred = self.conv_4_3(self.lnonz4_1(self.conv_4_2(rep))).view(-1, self.ch[4], 4, 4)
        return pred

    @th.compile
    def merge3(self, rep0, rep1):
        rep = th.cat([rep0, rep1], dim=1).view(-1, self.ch[3] * 2, 8, 8)
        pred = self.conv_8_3(self.lnonz3_1(self.conv_8_2(rep))).view(-1, self.ch[3], 8, 8)
        return pred

    @th.compile
    def merge2(self, rep0, rep1):
        rep = th.cat([rep0, rep1], dim=1).view(-1, self.ch[2] * 2, 16, 16)
        pred = self.conv_16_3(self.lnonz2_1(self.conv_16_2(rep))).view(-1, self.ch[2], 16, 16)
        return pred

    @th.compile
    def merge1(self, rep0, rep1):
        rep = th.cat([rep0, rep1], dim=1).view(-1, self.ch[1] * 2, 32, 32)
        pred = self.conv_32_3(self.lnonz1_1(self.conv_32_2(rep))).view(-1, self.ch[1], 32, 32)
        return pred

    @th.compile
    def merge0(self, rep0, rep1):
        rep = th.cat([rep0, rep1], dim=1).view(-1, self.ch[0] * 2, 64, 64)
        pred = self.conv_64_3(self.lnonz0_1(self.conv_64_2(rep))).view(-1, self.ch[0], 64, 64)
        return pred

    @th.compile
    def merge(self, rep0, rep1):
        return (rep0 + rep1) / 2

    @th.compile
    def forward(self, x):
        x0 = x[:, 0:1]
        x1 = x[:, 1:2]
        x2 = x[:, 2:3]
        x3 = x[:, 3:4]
        x4 = x[:, 4:5]
        x5 = x[:, 5:6]
        x6 = x[:, 6:7]
        x7 = x[:, 7:8]
        x8 = x[:, 8:9]
        x9 = x[:, 9:10]
        x01 = x[:, 0:2]
        x12 = x[:, 1:3]
        x23 = x[:, 2:4]
        x34 = x[:, 3:5]
        x45 = x[:, 4:6]
        x56 = x[:, 5:7]
        x67 = x[:, 6:8]
        x78 = x[:, 7:9]
        x89 = x[:, 8:10]

        a0, a1, a2, a3, a4, a5, a6 = self.downward(x01)
        y12 = self.upward(a0, a1, a2, a3, a4, a5, a6)

        b0, b1, b2, b3, b4, b5, b6 = self.downward(x12)
        y23 = self.upward(b0, b1, b2, b3, b4, b5, b6)

        c0, c1, c2, c3, c4, c5, c6 = self.downward(x23)
        y34 = self.upward(c0, c1, c2, c3, c4, c5, c6)

        d0, d1, d2, d3, d4, d5, d6 = self.downward(x34)
        y45 = self.upward(d0, d1, d2, d3, d4, d5, d6)

        e0, e1, e2, e3, e4, e5, e6 = self.downward(x45)
        y56 = self.upward(e0, e1, e2, e3, e4, e5, e6)

        f0, f1, f2, f3, f4, f5, f6 = self.downward(x56)
        y67 = self.upward(f0, f1, f2, f3, f4, f5, f6)

        g0, g1, g2, g3, g4, g5, g6 = self.downward(x67)
        y78 = self.upward(g0, g1, g2, g3, g4, g5, g6)

        h0, h1, h2, h3, h4, h5, h6 = self.downward(x78)
        y89 = self.upward(h0, h1, h2, h3, h4, h5, h6)

        i0, i1, i2, i3, i4, i5, i6 = self.downward(x89)
        y9a = self.upward(i0, i1, i2, i3, i4, i5, i6)

        jp6, kp6, lp6, mp6, np6, op6, pp6, qp6, rp6 = self.evolve6(a6, b6, c6, d6, e6, f6, g6, h6, i6)
        jp5, kp5, lp5, mp5, np5, op5, pp5, qp5, rp5 = self.evolve5(a5, b5, c5, d5, e5, f5, g5, h5, i5)
        jp4, kp4, lp4, mp4, np4, op4, pp4, qp4, rp4 = self.evolve4(a4, b4, c4, d4, e4, f4, g4, h4, i4)
        jp3, kp3, lp3, mp3, np3, op3, pp3, qp3, rp3 = self.evolve3(a3, b3, c3, d3, e3, f3, g3, h3, i3)
        jp2, kp2, lp2, mp2, np2, op2, pp2, qp2, rp2 = self.evolve2(a2, b2, c2, d2, e2, f2, g2, h2, i2)
        jp1, kp1, lp1, mp1, np1, op1, pp1, qp1, rp1 = self.evolve1(a1, b1, c1, d1, e1, f1, g1, h1, i1)
        jp0, kp0, lp0, mp0, np0, op0, pp0, qp0, rp0 = self.evolve0(a0, b0, c0, d0, e0, f0, g0, h0, i0)

        j0, j1, j2, j3, j4, j5, j6 = self.downward(y9a)
        yab = self.upward(self.merge0(j0, jp0), self.merge1(j1, jp1), self.merge2(j2, jp2), self.merge3(j3, jp3),
                          self.merge4(j4, jp4), self.merge5(j5, jp5), self.merge6(j6, jp6))

        k0, k1, k2, k3, k4, k5, k6 = self.downward(yab)
        ybc = self.upward(self.merge0(k0, kp0), self.merge1(k1, kp1), self.merge2(k2, kp2), self.merge3(k3, kp3),
                          self.merge4(k4, kp4), self.merge5(k5, kp5), self.merge6(k6, kp6))

        l0, l1, l2, l3, l4, l5, l6 = self.downward(ybc)
        ycd = self.upward(self.merge0(l0, lp0), self.merge1(l1, lp1), self.merge2(l2, lp2), self.merge3(l3, lp3),
                          self.merge4(l4, lp4), self.merge5(l5, lp5), self.merge6(l6, lp6))

        m0, m1, m2, m3, m4, m5, m6 = self.downward(ycd)
        yde = self.upward(self.merge0(m0, mp0), self.merge1(m1, mp1), self.merge2(m2, mp2), self.merge3(m3, mp3),
                          self.merge4(m4, mp4), self.merge5(m5, mp5), self.merge6(m6, mp6))

        n0, n1, n2, n3, n4, n5, n6 = self.downward(yde)
        yef = self.upward(self.merge0(n0, np0), self.merge1(n1, np1), self.merge2(n2, np2), self.merge3(n3, np3),
                          self.merge4(n4, np4), self.merge5(n5, np5), self.merge6(n6, np6))

        o0, o1, o2, o3, o4, o5, o6 = self.downward(yef)
        yfg = self.upward(self.merge0(o0, op0), self.merge1(o1, op1), self.merge2(o2, op2), self.merge3(o3, op3),
                          self.merge4(o4, op4), self.merge5(o5, op5), self.merge6(o6, op6))

        p0, p1, p2, p3, p4, p5, p6 = self.downward(yfg)
        ygh = self.upward(self.merge0(p0, pp0), self.merge1(p1, pp1), self.merge2(p2, pp2), self.merge3(p3, pp3),
                          self.merge4(p4, pp4), self.merge5(p5, pp5), self.merge6(p6, pp6))

        q0, q1, q2, q3, q4, q5, q6 = self.downward(ygh)
        yhi = self.upward(self.merge0(q0, qp0), self.merge1(q1, qp1), self.merge2(q2, qp2), self.merge3(q3, qp3),
                          self.merge4(q4, qp4), self.merge5(q5, qp5), self.merge6(q6, qp6))

        r0, r1, r2, r3, r4, r5, r6 = self.downward(yhi)
        yij = self.upward(self.merge0(r0, rp0), self.merge1(r1, rp1), self.merge2(r2, rp2), self.merge3(r3, rp3),
                          self.merge4(r4, rp4), self.merge5(r5, rp5), self.merge6(r6, rp6))

        s0, s1, s2, s3, s4, s5, s6 = self.downward(yij)
        yjk = self.upward(s0, s1, s2, s3, s4, s5, s6)

        return th.cat([
            x0, y12[:, 0:1], y23[:, 0:1], y34[:, 0:1], y45[:, 0:1], y56[:, 0:1], y67[:, 0:1], y78[:, 0:1], y89[:, 0:1],
            y9a[:, 0:1],
            yab[:, 0:1], ybc[:, 0:1], ycd[:, 0:1], yde[:, 0:1], yef[:, 0:1], yfg[:, 0:1], ygh[:, 0:1], yhi[:, 0:1],
            yij[:, 0:1], yjk[:, 0:1]
        ], dim=1)


def _model_():
    return Moving1()
