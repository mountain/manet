import argparse
import torch
import lightning.pytorch as pl

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("-b", "--batch", type=int, default=64, help="batch size of training")
parser.add_argument("-m", "--model", type=str, default='nmist2', help="model to execute")
opt = parser.parse_args()

if torch.cuda.is_available():
    accelerator = 'gpu'
else:
    accelerator = 'cpu'


if __name__ == '__main__':

    print('loading data...')

    from torch.utils.data import DataLoader
    from torch.utils.data import random_split
    from torchvision.datasets import MNIST
    from torchvision import transforms

    dataset = MNIST('', train=True, download=True, transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ]))

    mnist_test = MNIST('', train=False, download=True, transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ]))

    mnist_train, mnist_val = random_split(dataset, [55000, 5000])
    train_loader = DataLoader(mnist_train, shuffle=True, batch_size=128)
    val_loader = DataLoader(mnist_val, shuffle=True, batch_size=128)
    test_loader = DataLoader(mnist_val, shuffle=True, batch_size=128)

    # training
    print('construct trainer...')
    trainer = pl.Trainer(accelerator=accelerator, precision=16, max_epochs=opt.n_epochs)

    import importlib
    print('construct model...')
    mdl = importlib.import_module('demo.mnist.%s' % opt.model, package=None)
    model = mdl._model_()

    print('training...')
    trainer.fit(model, train_loader, val_loader)

    print('testing...')
    trainer.test(model, test_loader)
