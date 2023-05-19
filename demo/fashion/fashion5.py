import torch as th
import lightning as pl

from torch import nn
from torch.nn import functional as F
from torchvision.ops import MLP

from manet.aeg import LogisticFunction
from manet.mac import Reshape


class Fashion4(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = 1e-3
        self.counter = 0
        self.labeled_loss = 0
        self.labeled_correct = 0
        self.learnable_function0 = LogisticFunction(p=3.8, debug_key='lf0')
        self.learnable_function1 = LogisticFunction(p=3.8, debug_key='lf1')
        self.learnable_function2 = LogisticFunction(p=3.9, debug_key='lf2')
        self.learnable_function3 = LogisticFunction(p=3.8, debug_key='lf3')
        self.recognizer = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=5, padding=2),
            self.learnable_function0,
            Reshape(5, 28, 28),
            nn.MaxPool2d(2),
            nn.Conv2d(5, 20, kernel_size=3, padding=1),
            self.learnable_function1,
            Reshape(20, 14, 14),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 80, kernel_size=3, padding=1),
            self.learnable_function2,
            Reshape(80, 7, 7),
            nn.MaxPool2d(2),
            nn.Conv2d(80, 320, kernel_size=1, padding=0),
            self.learnable_function3,
            Reshape(320 * 3 * 3),
            MLP(320 * 9, [10]),
            Reshape(10),
            nn.LogSoftmax(dim=1)
        )

        self.tb_logger = None
        self.learnable_function0.logger = None
        self.learnable_function1.logger = None
        self.learnable_function2.logger = None
        self.learnable_function3.logger = None

    def backward(self, loss, *args, **kwargs):
        loss.backward(*args, **kwargs, retain_graph=True)

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
        if batch_idx % 100 == 0:
            self.learnable_function0.debug = True
            self.learnable_function1.debug = True
            self.learnable_function2.debug = True
            self.learnable_function3.debug = True

            self.learnable_function0.global_step += 1
            self.learnable_function1.global_step += 1
            self.learnable_function2.global_step += 1
            self.learnable_function3.global_step += 1

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

        x, y = val_batch
        x = x.view(-1, 1, 28, 28)

        if batch_idx % 100 == 0:
            y_true = y[:self.learnable_function0.num_samples]
            self.learnable_function0.labels = y_true
            self.learnable_function1.labels = y_true
            self.learnable_function2.labels = y_true
            self.learnable_function3.labels = y_true

        z = self(x)
        loss = F.nll_loss(z, y)
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
            self.log_tb_images((imgs, y_true, y_pred, self.learnable_function0.global_step))

        self.learnable_function0.debug = False
        self.learnable_function1.debug = False
        self.learnable_function2.debug = False
        self.learnable_function3.debug = False

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

    def log_tb_images(self, viz_batch) -> None:
        image, y_true, y_pred, idx = viz_batch
        for img_idx, (img, ground, pred) in enumerate(zip(image, y_true, y_pred)):
            self.tb_logger.add_image(f"Image/{idx}_{img_idx}-{ground.item()}-{pred.item()}", img, idx)


def _model_():
    return Fashion4()
