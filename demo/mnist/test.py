import torch as th
import pickle
import lightning.pytorch as pl

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

import demo.mnist.mnist8 as mdl

mnist_test = MNIST('datasets', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.1307,), (0.3081,))
]))

test_loader = DataLoader(mnist_test, batch_size=1, num_workers=1)

model = mdl._model_()

trainer = pl.Trainer(accelerator='cpu', precision=32, max_epochs=1)

if __name__ == '__main__':
    fname = 'best-1.00000-011-0.00869.ckpt'
    with open(fname, 'rb') as f:
        checkpoint = pickle.load(f)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.eval()

        with th.no_grad():
            counter, success = 0, 0
            for test_batch in test_loader:
                x, y = test_batch
                x = x.view(-1, 1, 28, 28)
                z = model(x)
                pred = z.data.max(1, keepdim=True)[1]
                correct = pred.eq(y.data.view_as(pred)).sum() / y.size()[0]
                print('.', end='', flush=True)
                if counter % 100 == 0:
                    print('')
                success += correct.item()
                counter += 1
        print('')
        print('Accuracy: %2.5f' % (success / counter))
