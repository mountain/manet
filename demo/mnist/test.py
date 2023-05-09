import torch as th
import pickle
import lightning.pytorch as pl

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import demo.mnist.mnist6 as mdl

mnist_test = MNIST('datasets', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,), (0.3081,))
]))

test_loader = DataLoader(mnist_test, batch_size=32, num_workers=8)

model = mdl._model_()

trainer = pl.Trainer(accelerator='cpu', precision=32, max_epochs=1)

fname = 'best-1.00000-049-0.35214.ckpt'
with open(fname, 'rb') as f:
    checkpoint = pickle.load(f)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    with th.no_grad():
        trainer.test(model, test_loader)
