import matplotlib

from demo.brach.dataset import TrainDataset, ValidDataset, TestDataset

matplotlib.use('Agg')

import argparse
import torch
import lightning.pytorch as pl

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from manet.tools.profiler import bind_profiling_context


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("-b", "--batch", type=int, default=32, help="batch size of training")
parser.add_argument("-m", "--model", type=str, default='brach2', help="model to execute")
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

    ds_train, ds_val, ds_test = TrainDataset(), ValidDataset(), TestDataset()

    train_loader = DataLoader(ds_train, shuffle=True, batch_size=opt.batch, num_workers=8)
    val_loader = DataLoader(ds_val, batch_size=opt.batch, num_workers=8)
    test_loader = DataLoader(ds_test, batch_size=opt.batch, num_workers=8)

    # training
    print('construct trainer...')
    trainer = pl.Trainer(accelerator=accelerator, precision=32, max_epochs=opt.n_epochs,
                         callbacks=[EarlyStopping(monitor="val_loss", mode="max", patience=30)])
    bind_profiling_context(trainer=trainer)

    import importlib
    print('construct model...')
    mdl = importlib.import_module('demo.brach.%s' % opt.model, package=None)
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
