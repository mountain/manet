import argparse
import torch
import pickle
import lightning.pytorch as pl
from torchtext.transforms import ToTensor

from demo.wikitext.dataset import ContextDataset


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("-b", "--batch", type=int, default=128, help="batch size of training")
parser.add_argument("-m", "--model", type=str, default='diffusion', help="model to execute")
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

    wiki_train, wiki_valid, wiki_test = (
        ContextDataset('train', transform=ToTensor()),
        ContextDataset('valid', transform=ToTensor()),
        ContextDataset('test', transform=ToTensor())
    )
    train_loader = DataLoader(wiki_train, batch_size=opt.batch, shuffle=True, num_workers=64)
    val_loader = DataLoader(wiki_valid, batch_size=opt.batch, num_workers=64)
    test_loader = DataLoader(wiki_test, batch_size=opt.batch, num_workers=64)

    # training
    print('construct trainer...')
    trainer = pl.Trainer(accelerator=accelerator, precision=64, max_epochs=opt.n_epochs)

    import importlib
    print('construct model...')
    mdl = importlib.import_module('demo.wikitext.emb.%s' % opt.model, package=None)
    model = mdl._model_()

    # fname = 'best-8.03624-5.ckpt'
    # with open(fname, 'rb') as f:
    #     checkpoint = pickle.load(f)
    #     model.load_state_dict(checkpoint['state_dict'], strict=False)

    print('training...')
    trainer.fit(model, train_loader, val_loader)

    print('testing...')
    trainer.test(model, test_loader)
