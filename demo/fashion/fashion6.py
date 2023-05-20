import numpy as np
import torch as th
import lightning as pl

from torch import nn
from torch.nn import functional as F
from torchvision.ops import MLP

from manet.aeg import LogisticFunction
from manet.mac import Reshape


class Fashion6(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = 1e-3
        self.counter = 0
        self.labeled_loss = 0
        self.labeled_correct = 0

        self.dnsample = nn.MaxPool2d(2)
        self.upsample0 = nn.Upsample(scale_factor=28 / 14, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=14 / 7, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=7 / 3, mode='nearest')
        self.upsample3 = nn.Upsample(scale_factor=3 / 1, mode='nearest')
        self.learnable_function0 = LogisticFunction(p=3.8, debug_key='lf0')
        self.learnable_function1 = LogisticFunction(p=3.8, debug_key='lf1')
        self.learnable_function2 = LogisticFunction(p=3.9, debug_key='lf2')
        self.learnable_function3 = LogisticFunction(p=3.8, debug_key='lf3')
        self.learnable_function4 = LogisticFunction(p=3.8, debug_key='lf4')
        self.learnable_function5 = LogisticFunction(p=3.8, debug_key='lf5')
        self.learnable_function6 = LogisticFunction(p=3.8, debug_key='lf6')
        self.learnable_function7 = LogisticFunction(p=3.8, debug_key='lf7')

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

        self.tb_logger = None
        self.learnable_function0.logger = None
        self.learnable_function1.logger = None
        self.learnable_function2.logger = None
        self.learnable_function3.logger = None

    def backward(self, loss, *args, **kwargs):
        loss.backward(*args, **kwargs, retain_graph=True)

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
        x4 = self.learnable_function3(x4)

        x5 = self.upsample3(x4)
        x5 = th.cat([x5, x3], dim=1)
        x5 = self.conv5(x5)
        x5 = self.learnable_function5(x5)
        x6 = self.upsample2(x5)
        x6 = th.cat([x6, x2], dim=1)
        x6 = self.conv5(x6)
        x6 = self.learnable_function6(x6)
        x7 = self.upsample1(x6)
        x7 = th.cat([x7, x1], dim=1)
        x7 = self.conv7(x7)
        x7 = self.learnable_function7(x7)
        x8 = self.upsample0(x7)
        x8 = self.conv8(x8)

        return self.lsm(self.mlp(self.flat(x4))), x8

    def configure_optimizers(self):
        return th.optim.Adam(self.parameters(), lr=self.learning_rate)

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

    def validation_step(self, val_batch, batch_idx):
        if batch_idx % 100 == 0:
            self.learnable_function0.debug = True
            self.learnable_function1.debug = True
            self.learnable_function2.debug = True
            self.learnable_function3.debug = True
            self.learnable_function4.debug = True
            self.learnable_function5.debug = True
            self.learnable_function6.debug = True
            self.learnable_function7.debug = True

            self.learnable_function0.global_step += 1
            self.learnable_function1.global_step += 1
            self.learnable_function2.global_step += 1
            self.learnable_function3.global_step += 1
            self.learnable_function5.global_step += 1
            self.learnable_function6.global_step += 1
            self.learnable_function7.global_step += 1

            import lightning.pytorch.loggers as pl_loggers
            tb_logger = None
            for logger in self.trainer.loggers:
                if isinstance(logger, pl_loggers.TensorBoardLogger):
                    tb_logger = logger.experiment
                    break
            if tb_logger is None:
                raise ValueError('TensorBoard Logger not found')

            self.tb_logger = tb_logger
            self.learnable_function0.logger = tb_logger
            self.learnable_function1.logger = tb_logger
            self.learnable_function2.logger = tb_logger
            self.learnable_function3.logger = tb_logger
            self.learnable_function4.logger = tb_logger
            self.learnable_function5.logger = tb_logger
            self.learnable_function6.logger = tb_logger
            self.learnable_function7.logger = tb_logger

        x, y = val_batch
        x = x.view(-1, 1, 28, 28)

        if batch_idx % 100 == 0:
            y_true = y[:self.learnable_function0.num_samples]
            self.learnable_function0.labels = y_true
            self.learnable_function1.labels = y_true
            self.learnable_function2.labels = y_true
            self.learnable_function3.labels = y_true
            self.learnable_function4.labels = y_true
            self.learnable_function5.labels = y_true
            self.learnable_function6.labels = y_true
            self.learnable_function7.labels = y_true

        z, x_hat = self(x)
        loss_classify = F.nll_loss(z, y)
        loss_recovery = F.mse_loss(x_hat, x)
        loss = loss_classify + loss_recovery
        self.log('loss_classify', loss_classify, prog_bar=True)
        self.log('loss_recovery', loss_recovery, prog_bar=True)
        self.log('val_loss', loss, prog_bar=True)

        pred = z.data.max(1, keepdim=True)[1]
        correct = pred.eq(y.data.view_as(pred)).sum() / y.size()[0]
        self.log('correctness', correct, prog_bar=True)
        self.labeled_loss += loss.item() * y.size()[0]
        self.labeled_correct += correct.item() * y.size()[0]
        self.counter += y.size()[0]

        if batch_idx % 100 == 0:
            imgs = x[:10]
            y_true = y[:10]
            y_pred = pred[:10]
            self.log_tb_images('Ground', (imgs, y_true, y_pred, self.learnable_function0.global_step))
            self.log_tb_images('Recovery', (x_hat, y_true, y_pred, self.learnable_function0.global_step))

        self.learnable_function0.debug = False
        self.learnable_function1.debug = False
        self.learnable_function2.debug = False
        self.learnable_function3.debug = False
        self.learnable_function4.debug = False
        self.learnable_function5.debug = False
        self.learnable_function6.debug = False
        self.learnable_function7.debug = False

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x = x.view(-1, 1, 28, 28)
        z = self(x)
        pred = z.data.max(1, keepdim=True)[1]
        correct = pred.eq(y.data.view_as(pred)).sum() / y.size()[0]
        self.log('correct', correct, prog_bar=True)

    def on_save_checkpoint(self, checkpoint) -> None:
        import pickle, glob, os

        correct = self.labeled_correct / self.counter
        loss = self.labeled_loss / self.counter
        record = '%2.5f-%03d-%1.5f.ckpt' % (correct, checkpoint['epoch'], loss)
        fname = 'best-%s' % record
        with open(fname, 'bw') as f:
            pickle.dump(checkpoint, f)
        for ix, ckpt in enumerate(sorted(glob.glob('best-*.ckpt'), reverse=True)):
            if ix > 5:
                os.unlink(ckpt)

        self.counter = 0
        self.labeled_loss = 0
        self.labeled_correct = 0

    def log_tb_images(self, key, viz_batch) -> None:
        image, y_true, y_pred, idx = viz_batch
        for img_idx, (img, ground, pred) in enumerate(zip(image, y_true, y_pred)):
            self.tb_logger.add_image(f"{key}/{idx}_{img_idx}-{ground.item()}-{pred.item()}", img, idx)


def _model_():
    return Fashion6()
