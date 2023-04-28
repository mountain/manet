import os.path as pth
import torch as th
import pickle

from torch.utils.data import DataLoader
from torchtext.datasets import WikiText2

import demo.wikitext.emb.direct as mdl

dataset = WikiText2('')
_, _, wiki_test = dataset
test_loader = DataLoader(wiki_test, batch_size=1)

model = mdl._model_()

fname = '%s/best-0.000000039121783-194.ckpt' % pth.dirname(__file__)
with open(fname, 'rb') as f:
    checkpoint = pickle.load(f)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()

    with th.no_grad():
        for items in test_loader:
            for paragraph in items:
                paragraph = paragraph.split(' ')
                if len(paragraph) > 40:
                    print(' '.join(paragraph[:20]))
                    print(' '.join(paragraph[20:40]))
                    print(model.complete(paragraph[:20]))
                    # press any key to continue
                    input()
