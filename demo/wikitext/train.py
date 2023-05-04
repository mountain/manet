import argparse
import torch as th
from torchtext.transforms import ToTensor

from demo.wikitext.dataset import ContextDataset
from demo.wikitext.emb.diffusion import default_steps

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("-b", "--batch", type=int, default=128, help="batch size of training")
parser.add_argument("-m", "--model", type=str, default='diffusion', help="model to execute")
opt = parser.parse_args()

device = (
    "cuda"
    if th.cuda.is_available()
    else "mps"
    if th.backends.mps.is_available()
    else "cpu"
)
print(f"using {device} device...")


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, X in enumerate(dataloader):
        y = X[:, default_steps // 3:]
        # Compute prediction and loss
        pred = model(X.to(device))
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with th.no_grad():
        for X in dataloader:
            y = X[:, default_steps // 3:]
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(th.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':

    print('loading data...')

    from torch.utils.data import DataLoader

    wiki_train, wiki_valid, wiki_test = (
        ContextDataset('train', transform=ToTensor()),
        ContextDataset('valid', transform=ToTensor()),
        ContextDataset('test', transform=ToTensor())
    )
    train_loader = DataLoader(wiki_train, batch_size=opt.batch, shuffle=True, num_workers=16)
    val_loader = DataLoader(wiki_valid, batch_size=opt.batch, num_workers=16)
    test_loader = DataLoader(wiki_test, batch_size=opt.batch, num_workers=16)

    import importlib
    print('construct model...')
    mdl = importlib.import_module('demo.wikitext.emb.%s' % opt.model, package=None)
    model = mdl._model_()
    model.to(device)

    # fname = 'best-8.03624-5.ckpt'
    # with open(fname, 'rb') as f:
    #     checkpoint = pickle.load(f)
    #     model.load_state_dict(checkpoint['state_dict'], strict=False)

    from torch.nn import functional as F
    loss_fn = F.nll_loss
    optimizer = th.optim.Adam(model.parameters(), lr=1e-3)
    epochs = opt.n_epochs
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader, model, loss_fn, optimizer)
        test_loop(val_loader, model, loss_fn)
    test_loop(test_loader, model, loss_fn)

    print("Done!")
