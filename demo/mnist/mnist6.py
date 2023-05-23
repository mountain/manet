import torch as th
import torch.nn.functional as F

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

        self.dnsample = nn.MaxPool2d(2)
        self.upsample0 = nn.Upsample(scale_factor=28 / 14, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=14 / 7, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=7 / 3, mode='nearest')
        self.upsample3 = nn.Upsample(scale_factor=3 / 1, mode='nearest')

        self.learnable_function0 = LearnableFunction(debug_key='lf0')
        self.learnable_function1 = LearnableFunction(debug_key='lf1')
        self.learnable_function2 = LearnableFunction(debug_key='lf2')
        self.learnable_function3 = LearnableFunction(debug_key='lf3')
        self.learnable_function4 = LearnableFunction(debug_key='lf4')
        self.learnable_function5 = LearnableFunction(debug_key='lf5')
        self.learnable_function6 = LearnableFunction(debug_key='lf6')
        self.learnable_function7 = LearnableFunction(debug_key='lf7')

        self.conv0 = nn.Conv2d(1, 5, kernel_size=5, padding=2)
        self.conv1 = nn.Conv2d(5, 15, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(15, 45, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(45, 135, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(135, 405, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(135 + 405, 180, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(45 + 180, 75, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(15 + 75, 30, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(30, 1, kernel_size=3, padding=1)

        self.flat = Reshape(405)
        self.mlp = MLP(405, [10])
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.learnable_function0(x0)
        x1 = self.dnsample(x0)
        x1 = self.conv1(x1)
        x1 = self.learnable_function1(x1)
        x2 = self.dnsample(x1)
        x2 = self.conv2(x2)
        x2 = self.learnable_function2(x2)
        x3 = self.dnsample(x2)
        x3 = self.conv3(x3)
        x3 = self.learnable_function3(x3)
        x4 = self.dnsample(x3)
        x4 = self.conv4(x4)
        x4 = self.learnable_function4(x4)

        x5 = self.upsample3(x4)
        x5 = th.cat([x5, x3], dim=1)
        x5 = self.conv5(x5)
        x5 = self.learnable_function5(x5)
        x6 = self.upsample2(x5)
        x6 = th.cat([x6, x2], dim=1)
        x6 = self.conv6(x6)
        x6 = self.learnable_function6(x6)
        x7 = self.upsample1(x6)
        x7 = th.cat([x7, x1], dim=1)
        x7 = self.conv7(x7)
        x7 = self.learnable_function7(x7)
        x8 = self.upsample0(x7)
        x8 = self.conv8(x8)

        return self.lsm(self.mlp(self.flat(x4))), x8

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view(-1, 1, 28, 28)
        z, x_hat = self(x)
        loss_classify = F.nll_loss(z, y)
        loss_recovery = F.mse_loss(x_hat, x)
        loss = loss_classify + loss_recovery
        self.log('loss_classify', loss_classify, prog_bar=True)
        self.log('loss_recovery', loss_recovery, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return loss


def _model_():
    return MNModel6()
