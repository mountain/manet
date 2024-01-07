import argparse
import torch
import lightning.pytorch as pl

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("-b", "--batch", type=int, default=32, help="batch size of training")
parser.add_argument("-m", "--model", type=str, default='fashion2', help="model to execute")
opt = parser.parse_args()

if torch.cuda.is_available():
    accelerator = 'gpu'
elif torch.backends.mps.is_available():
    accelerator = 'mps'
else:
    accelerator = 'cpu'


if __name__ == '__main__':

    print('loading data...')
    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR100
    from torchvision import transforms


    cifar_raw = CIFAR100('datasets', train=True, download=True, transform=transforms.Compose([
                                   transforms.ToTensor()
                                 ]))
    raw_loader = DataLoader(cifar_raw, shuffle=False, batch_size=opt.batch, num_workers=8)
    counter, total, vairance = 0, 0, 0
    for raw_batch in raw_loader:
        x, y = raw_batch
        x = x.view(-1, 3, 32, 32)
        counter += (32 * 32 * 3 * x.shape[0])
        total += x.sum()

    mean = (total / counter).item()
    for raw_batch in raw_loader:
        x, y = raw_batch
        x = x.view(-1, 3, 32, 32)
        vairance += ((x - mean).pow(2)).sum()

    std = torch.sqrt(vairance / counter).item()
    print('mean: %f, std: %f' % (mean, std))

    cifar_train = CIFAR100('datasets', train=True, download=True, transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                     (mean,), (std,))
                                 ]))

    cifar_test = CIFAR100('datasets', train=False, download=True, transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize(
                                     (mean,), (std,))
                                 ]))


    train_loader = DataLoader(cifar_train, shuffle=True, batch_size=opt.batch, num_workers=8, persistent_workers=True)
    val_loader = DataLoader(cifar_test, batch_size=opt.batch, num_workers=8, persistent_workers=True)
    test_loader = DataLoader(cifar_test, batch_size=opt.batch, num_workers=8, persistent_workers=True)

    # training
    print('construct trainer...')
    trainer = pl.Trainer(max_epochs=opt.n_epochs, log_every_n_steps=1,
                         callbacks=[EarlyStopping(monitor="correct_rate", mode="max", patience=30)])

    import importlib
    print('construct model...')
    mdl = importlib.import_module('demo.cifar100.%s' % opt.model, package=None)
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
