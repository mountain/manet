import argparse
from functools import partial
from types import SimpleNamespace
import random
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as tf
import lightning as ltn
import lightning.pytorch as pl

from torch import Tensor
from fastprogress import progress_bar
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchvision.datasets import MNIST


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("-b", "--batch", type=int, default=256, help="batch size of training")
parser.add_argument("-m", "--model", type=str, default='moving0', help="model to execute")
opt = parser.parse_args()

if th.cuda.is_available():
    accelerator = 'gpu'
    th.set_float32_matmul_precision('medium')
elif th.backends.mps.is_available():
    accelerator = 'cpu'
else:
    accelerator = 'cpu'


def to_float(x):
    return x.float()


def recover(x):
    return x * 51.070168 + 12.562058


mnist_stats = ([0.131], [0.308])


def padding(img_size=64, mnist_size=28):
    return (img_size - mnist_size) // 2


def apply_n_times(tf, x, n=1):
    "Apply `tf` to `x` `n` times, return all values"
    sequence = [x]
    for n in range(n):
        sequence.append(tf(sequence[n]))
    return sequence


affine_params = SimpleNamespace(
    angle=(-4, 4),
    translate=((-5, 5), (-5, 5)),
    scale=(.8, 1.2),
    shear=(-3, 3),
)


class RandomTrajectory:
    def __init__(self, affine_params, n=5, **kwargs):
        self.angle = random.uniform(*affine_params.angle)
        self.translate = (random.uniform(*affine_params.translate[0]),
                          random.uniform(*affine_params.translate[1]))
        self.scale = random.uniform(*affine_params.scale)
        self.shear = random.uniform(*affine_params.shear)
        self.n = n
        self.tf = partial(tf.affine, angle=self.angle, translate=self.translate, scale=self.scale, shear=self.shear,
                          **kwargs)

    def __call__(self, img):
        return apply_n_times(self.tf, img, n=self.n)

    def __repr__(self):
        s = ("RandomTrajectory(\n"
             f"  angle:     {self.angle}\n"
             f"  translate: {self.translate}\n"
             f"  scale:     {self.scale}\n"
             f"  shear:     {self.shear}\n)")
        return s


class MovingMNIST:
    def __init__(self, path=".",  # path to store the MNIST dataset
                 affine_params: dict = affine_params,
                 # affine transform parameters, refer to torchvision.transforms.functional.affine
                 num_digits: list[int] = [1, 2],  # how many digits to move, random choice between the value provided
                 num_frames: int = 4,  # how many frames to create
                 img_size=64,  # the canvas size, the actual digits are always 28x28
                 concat=True,  # if we concat the final results (frames, 1, 28, 28) or a list of frames.
                 normalize=False,
                 # scale images in [0,1] and normalize them with MNIST stats. Applied at batch level. Have to take care of the canvas size that messes up the stats!
                 transform=None
                 ):
        self.mnist = MNIST(path, download=True).data
        self.affine_params = affine_params
        self.num_digits = num_digits
        self.num_frames = num_frames
        self.img_size = img_size
        self.pad = padding(img_size)
        self.concat = concat

        # some computation to ensure normalizing correctly-ish
        batch_tfms = [T.ConvertImageDtype(th.float32)]
        if normalize:
            ratio = (28 / img_size) ** 2 * max(num_digits)
            mean, std = mnist_stats
            scaled_mnist_stats = ([mean[0] * ratio], [std[0] * ratio])
            print(f"New computed stats for MovingMNIST: {scaled_mnist_stats}")
            batch_tfms += [T.Normalize(*scaled_mnist_stats)] if normalize else []
        self.batch_tfms = T.Compose(batch_tfms)
        self.transform = transform

    def __len__(self):
        return 100000

    def random_place(self, img):
        "Randomly place the digit inside the canvas"
        x = random.uniform(-self.pad, self.pad)
        y = random.uniform(-self.pad, self.pad)
        return tf.affine(img, translate=(x, y), angle=0, scale=1, shear=(0, 0))

    def random_digit(self):
        "Get a random MNIST digit randomly placed on the canvas"
        img = self.mnist[[random.randrange(0, len(self.mnist))]]
        pimg = tf.pad(img, padding=self.pad)
        return self.random_place(pimg)

    def _one_moving_digit(self):
        digit = self.random_digit()
        traj = RandomTrajectory(self.affine_params, n=self.num_frames - 1)
        return th.stack(traj(digit))

    def __getitem__(self, i):
        moving_digits = [self._one_moving_digit() for _ in range(random.choice(self.num_digits))]
        moving_digits = th.stack(moving_digits)
        moving_digits = self.transform(moving_digits)
        combined_digits = moving_digits.max(dim=0)[0]
        return combined_digits if self.concat else [t.squeeze(dim=0) for t in combined_digits.split(1)]

    def get_batch(self, bs=32):
        "Grab a batch of data"
        batch = th.stack([self[0] for _ in range(bs)])
        return self.batch_tfms(batch) if self.batch_tfms is not None else batch

    def save(self, fname="mmnist.pt", n_batches=2, bs=32):
        data = []
        for _ in progress_bar(range(n_batches)):
            data.append(self.get_batch(bs=bs))

        data = th.cat(data, dim=0)
        print("Saving dataset")
        th.save(data, f"{fname}")


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


class MovingModel(ltn.LightningModule):
    def __init__(self):
        super().__init__()

        self.learning_rate = 1e-3
        self.counter = 0
        self.labeled_mse = 0

        self.dnsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=33 / 16, mode='nearest')

        self.conv0 = nn.Conv2d(10, 40, kernel_size=3, padding=1, bias=False)
        self.lnon0 = LNon()

        self.conv1 = nn.Conv2d(40, 80, kernel_size=3, padding=1, bias=False)
        self.lnon1 = LNon()

        self.conv2 = nn.Conv2d(80, 160, kernel_size=3, padding=1, bias=False)
        self.lnon2 = LNon()

        self.conv3 = nn.Conv2d(160, 320, kernel_size=3, padding=1, bias=False)
        self.lnon3 = LNon()

        self.conv4 = nn.Conv2d(320, 320, kernel_size=1, padding=0, bias=False)
        self.lnon4 = LNon()

        self.conv5 = nn.Conv2d(320, 320, kernel_size=1, padding=0, bias=False)
        self.lnon5 = LNon()

        self.conv6 = nn.Conv2d(320, 320, kernel_size=1, padding=0, bias=False)
        self.lnon6 = LNon()

        self.conv7 = nn.Conv2d(640, 320, kernel_size=1, padding=0, bias=False)
        self.lnon7 = LNon()

        self.conv8 = nn.Conv2d(640, 320, kernel_size=1, padding=0, bias=False)
        self.lnon8 = LNon()

        self.conv9 = nn.Conv2d(640, 160, kernel_size=3, padding=1, bias=False)
        self.lnon9 = LNon()

        self.conva = nn.Conv2d(320, 80, kernel_size=3, padding=1, bias=False)
        self.lnona = LNon()

        self.convb = nn.Conv2d(160, 40, kernel_size=3, padding=1, bias=False)
        self.lnonb = LNon()

        self.convc = nn.Conv2d(80, 20, kernel_size=3, padding=1, bias=False)
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

        self.labeled_mse += mse.item() * y.size()[0]
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

        loss = self.labeled_mes / self.counter
        record = '%03.5f-%03d.ckpt' % (loss, checkpoint['epoch'])
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
        x0 = self.lnon0(x0)

        # 66 -> 33
        x1 = self.downward_calc(self.dnsample, self.conv1, self.lnon1, x0)
        # 33 -> 16
        x2 = self.downward_calc(self.dnsample, self.conv2, self.lnon2, x1)
        # 16 -> 8
        x3 = self.downward_calc(self.dnsample, self.conv3, self.lnon3, x2)
        # 8 -> 4
        x4 = self.downward_calc(self.dnsample, self.conv4, self.lnon4, x3)
        # 4 -> 2
        x5 = self.downward_calc(self.dnsample, self.conv5, self.lnon5, x4)
        # 2 -> 1
        x6 = self.downward_calc(self.dnsample, self.conv6, self.lnon6, x5)

        # 1 -> 2
        x7 = self.upward_calc(self.upsample, self.conv7, self.lnon7, x5, x6)
        # 2 -> 4
        x8 = self.upward_calc(self.upsample, self.conv8, self.lnon8, x4, x7)
        # 4 -> 8
        x9 = self.upward_calc(self.upsample, self.conv9, self.lnon9, x3, x8)
        # 8 -> 16
        xa = self.upward_calc(self.upsample, self.conva, self.lnona, x2, x9)
        # 16 -> 33
        xb = self.upward_calc(self.upsample1, self.convb, self.lnonb, x1, xa)
        # 33 -> 66
        xc = self.upward_calc(self.upsample, self.convc, self.lnonc, x0, xb)

        x = xc[:, :, 1:-1, 1:-1]
        return x


if __name__ == '__main__':

    print('loading data...')
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from demo.movingmnist.data import MovingMNIST

    mean, std = 12.562058, 51.070168
    transform = transforms.Compose([
                transforms.Lambda(to_float),
                transforms.Normalize((mean,), (std,))
    ])

    mnist_train = MovingMNIST(path='datasets/movingmnist', num_frames=20, transform=transform)

    mnist_test = MovingMNIST(path='datasets/movingmnist', num_frames=20, transform=transform)

    train_loader = DataLoader(mnist_train, shuffle=True, batch_size=opt.batch, num_workers=16, persistent_workers=True)
    val_loader = DataLoader(mnist_test, batch_size=opt.batch, num_workers=16, persistent_workers=True)
    test_loader = DataLoader(mnist_test, batch_size=opt.batch, num_workers=16, persistent_workers=True)

    # training
    print('construct trainer...')
    trainer = pl.Trainer(max_epochs=opt.n_epochs, log_every_n_steps=1,
                         callbacks=[EarlyStopping(monitor="val_mse", mode="min", patience=30)])

    import importlib
    print('construct model...')
    model = MovingModel()

    print('training...')
    trainer.fit(model, train_loader, val_loader)

    print('testing...')
    trainer.test(model, test_loader)
