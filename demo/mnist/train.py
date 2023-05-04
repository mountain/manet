import argparse
import torch as th

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("-b", "--batch", type=int, default=512, help="batch size of training")
parser.add_argument("-m", "--model", type=str, default='mnist2', help="model to execute")
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
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.to(device)).flatten()
        loss = loss_fn(pred, y * th.ones_like(pred, dtype=th.long))

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
        for X, y in dataloader:
            pred = model(X.to(device)).flatten()
            test_loss += loss_fn(pred, y * th.ones_like(pred, dtype=th.long)).item()
            correct += (pred.argmax(1) == y).type(th.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


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
    train_loader = DataLoader(mnist_train, shuffle=True, batch_size=opt.batch, num_workers=0)
    val_loader = DataLoader(mnist_val, batch_size=opt.batch, num_workers=0)
    test_loader = DataLoader(mnist_val, batch_size=opt.batch, num_workers=0)

    import importlib
    print('construct model...')
    mdl = importlib.import_module('demo.mnist.%s' % opt.model, package=None)
    model = mdl._model_()
    model.to(device)

    from torch.nn import functional as F
    loss_fn = F.nll_loss
    optimizer = th.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(mnist_train, model, loss_fn, optimizer)
        test_loop(mnist_val, model, loss_fn)
    test_loop(mnist_test, model, loss_fn)

    print("Done!")
