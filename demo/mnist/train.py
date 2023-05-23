import matplotlib
matplotlib.use('Agg')

import argparse
import torch
import lightning.pytorch as pl

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from manet.tools.profiler import bind_profiling_context


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("-b", "--batch", type=int, default=32, help="batch size of training")
parser.add_argument("-m", "--model", type=str, default='mnist9', help="model to execute")
opt = parser.parse_args()

if torch.cuda.is_available():
    accelerator = 'gpu'
elif torch.backends.mps.is_available():
    accelerator = 'cpu'
else:
    accelerator = 'cpu'


if __name__ == '__main__':

    print('loading data...')
    from torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision import transforms

    mnist_train = MNIST('datasets', train=True, download=True, transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ]))

    mnist_test = MNIST('datasets', train=False, download=True, transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ]))

    train_loader = DataLoader(mnist_train, shuffle=True, batch_size=opt.batch, num_workers=8)
    val_loader = DataLoader(mnist_test, batch_size=opt.batch, num_workers=8)
    test_loader = DataLoader(mnist_test, batch_size=opt.batch, num_workers=8)

    # training
    print('construct trainer...')
    trainer = pl.Trainer(accelerator=accelerator, precision=32, max_epochs=opt.n_epochs,
                         callbacks=[EarlyStopping(monitor="correctness", mode="max", patience=30)])
    bind_profiling_context(trainer)

    import importlib
    print('construct model...')
    mdl = importlib.import_module('demo.mnist.%s' % opt.model, package=None)
    model = mdl._model_()

    # fname = 'seed.ckpt'
    # with open(fname, 'rb') as f:
    #     import pickle
    #     checkpoint = pickle.load(f)
    #     model.load_state_dict(checkpoint['state_dict'], strict=False)

    print('training...')
    trainer.fit(model, train_loader, val_loader)

    print('testing...')
    trainer.test(model, test_loader)
