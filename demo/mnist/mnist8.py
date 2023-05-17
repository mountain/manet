import torch as th
import lightning as pl

from torch import nn
from torch.nn import functional as F

from manet.aeg import LearnableFunction
from manet.mac import MLP, MacSplineUnit, Reshape


class MNModel8(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = 1e-3
        self.counter = 0
        self.labeled_loss = 0
        self.labeled_correct = 0
        self.recognizer = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, padding=2),
            LearnableFunction(),
            Reshape(5, 28, 28),
            nn.MaxPool2d(2),
            nn.Conv2d(5, 10, kernel_size=3, padding=1),
            LearnableFunction(),
            Reshape(10, 14, 14),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, kernel_size=3, padding=1),
            LearnableFunction(),
            Reshape(20, 7, 7),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, kernel_size=1, padding=0),
            LearnableFunction(),
            Reshape(40, 3, 3),
            MLP(40 * 9, [10], mac_unit=MacSplineUnit),
            Reshape(10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.recognizer(x)

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
        self.log('correctness', correct, prog_bar=True)
        self.labeled_loss += loss.item() * y.size()[0]
        self.labeled_correct += correct.item() * y.size()[0]
        self.counter += y.size()[0]
        if batch_idx % 10:  # Log every 10 batches
            imgs = x[:5]
            y_true = y[:5]
            y_pred = pred[:5]
            self.log_tb_images((imgs, y_true, y_pred, batch_idx))

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
        # for ix, ckpt in enumerate(sorted(glob.glob('best-*.ckpt'), reverse=True)):
        #     if ix > 5:
        #         os.unlink(ckpt)

        self.counter = 0
        self.labeled_loss = 0
        self.labeled_correct = 0

    def log_tb_images(self, viz_batch) -> None:
        import pytorch_lightning.loggers as pl_loggers
        # Get tensorboard logger
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break

        if tb_logger is None:
            raise ValueError('TensorBoard Logger not found')

            # Log the images (Give them different names)
        for img_idx, (image, y_true, y_pred, batch_idx) in enumerate(zip(*viz_batch)):
            tb_logger.add_image(f"Image/{batch_idx}_{img_idx}", image, 0)
            tb_logger.add_image(f"GroundTruth/{batch_idx}_{img_idx}", y_true, 0)
            tb_logger.add_image(f"Prediction/{batch_idx}_{img_idx}", y_pred, 0)


def _model_():
    return MNModel8()
