import argparse
import torch
import lightning.pytorch as pl

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("-b", "--batch", type=int, default=512, help="batch size of training")
parser.add_argument("-m", "--model", type=str, default='mnist2', help="model to execute")
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
    from torch.utils.data import random_split
    from torchvision.datasets import MNIST
    from torchvision import transforms

    dataset = MNIST('datasets', train=True, download=True, transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ]))

    mnist_test = MNIST('datasets', train=False, download=True, transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ]))

    mnist_train, mnist_val = random_split(dataset, [55000, 5000])
    train_loader = DataLoader(mnist_train, shuffle=True, batch_size=opt.batch, num_workers=64)
    val_loader = DataLoader(mnist_val, batch_size=opt.batch, num_workers=64)
    test_loader = DataLoader(mnist_val, batch_size=opt.batch, num_workers=64)

    # training
    print('construct trainer...')
    trainer = pl.Trainer(accelerator=accelerator, precision=32, max_epochs=opt.n_epochs)

    import importlib
    print('construct model...')
    mdl = importlib.import_module('demo.mnist.%s' % opt.model, package=None)
    model = mdl._model_()

    print('training...')
    trainer.fit(model, train_loader, val_loader)

    print('testing...')
    trainer.test(model, test_loader)
