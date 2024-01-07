import torch as th
import lightning as ltn
import torch.nn.functional as F

from manet.tools.profiler import reset_profiling_stage, is_profiling, ctx


class BaseModel(ltn.LightningModule):
    def __init__(self):
        super().__init__()
        self.learning_rate = 1e-3

    def forward(self, x):
        raise NotImplementedError

    def configure_optimizers(self):
        return [th.optim.Adam(self.parameters(), lr=self.learning_rate)]


class MNISTModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.labeled_loss = 0
        self.labeled_correct = 0

    def training_step(self, train_batch, batch_idx):
        reset_profiling_stage('train')

        x, y = train_batch
        x = x.view(-1, 1, 28, 28)
        z = self(x)
        loss = F.nll_loss(z, y)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        reset_profiling_stage('valid')

        x, y = val_batch
        x = x.view(-1, 1, 28, 28)

        z = self(x)
        loss = F.nll_loss(z, y)
        self.log('val_loss', loss, prog_bar=True)

        pred = z.data.max(1, keepdim=True)[1]
        correct = pred.eq(y.data.view_as(pred)).sum() / y.size()[0]
        self.log('correct_rate', correct, prog_bar=True)

        self.labeled_loss += loss.item() * y.size()[0]
        self.labeled_correct += correct.item() * y.size()[0]
        self.counter += y.size()[0]

        if is_profiling():
            num_samples = ctx['num_samples']
            ctx['original_input'] = x[:num_samples]
            ctx['ground_truth'] = y[:num_samples]
            ctx['prediction'] = pred[:num_samples]

    def test_step(self, test_batch, batch_idx):
        reset_profiling_stage('test')

        x, y = test_batch
        x = x.view(-1, 1, 28, 28)
        z = self(x)

        pred = z.data.max(1, keepdim=True)[1]
        correct = pred.eq(y.data.view_as(pred)).sum() / y.size()[0]
        self.log('correct_rate', correct, prog_bar=True)

    def on_save_checkpoint(self, checkpoint) -> None:
        import glob, os

        correct = self.labeled_correct / self.counter
        loss = self.labeled_loss / self.counter
        record = '%2.5f-%03d-%1.5f.ckpt' % (correct, checkpoint['epoch'], loss)
        fname = 'best-%s' % record
        with open(fname, 'bw') as f:
            th.save(checkpoint, f)
        for ix, ckpt in enumerate(sorted(glob.glob('best-*.ckpt'), reverse=True)):
            if ix > 5:
                os.unlink(ckpt)

        self.counter = 0
        self.labeled_loss = 0
        self.labeled_correct = 0


class CIFARModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.counter = 0
        self.labeled_loss = 0
        self.labeled_correct = 0

    def training_step(self, train_batch, batch_idx):
        reset_profiling_stage('train')

        x, y = train_batch
        x = x.view(-1, 3, 32, 32)
        z = self.forward(x)
        loss = F.nll_loss(z, y)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        reset_profiling_stage('valid')

        x, y = val_batch
        x = x.view(-1, 3, 32, 32)

        z = self.forward(x)
        loss = F.nll_loss(z, y)
        self.log('val_loss', loss, prog_bar=True)

        pred = z.data.max(1, keepdim=True)[1]
        correct = pred.eq(y.data.view_as(pred)).sum() / y.size()[0]
        self.log('correct_rate', correct, prog_bar=True)

        self.labeled_loss += loss.item() * y.size()[0]
        self.labeled_correct += correct.item() * y.size()[0]
        self.counter += y.size()[0]

        if is_profiling():
            num_samples = ctx['num_samples']
            ctx['original_input'] = x[:num_samples]
            ctx['ground_truth'] = y[:num_samples]
            ctx['prediction'] = pred[:num_samples]

    def test_step(self, test_batch, batch_idx):
        reset_profiling_stage('test')

        x, y = test_batch
        x = x.view(-1, 3, 32, 32)
        z = self.forward(x)

        pred = z.data.max(1, keepdim=True)[1]
        correct = pred.eq(y.data.view_as(pred)).sum() / y.size()[0]
        self.log('correct_rate', correct, prog_bar=True)

    def on_save_checkpoint(self, checkpoint) -> None:
        import glob, os

        correct = self.labeled_correct / self.counter
        loss = self.labeled_loss / self.counter
        record = '%2.5f-%03d-%1.5f.ckpt' % (correct, checkpoint['epoch'], loss)
        fname = 'best-%s' % record
        with open(fname, 'bw') as f:
            th.save(checkpoint, f)
        for ix, ckpt in enumerate(sorted(glob.glob('best-*.ckpt'), reverse=True)):
            if ix > 5:
                os.unlink(ckpt)

        self.counter = 0
        self.labeled_loss = 0
        self.labeled_correct = 0
