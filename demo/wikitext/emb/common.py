import os
import pickle
import glob

import torch as th
import torch.nn as nn

from typing import Dict, Any



class EmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()

        vname = 'datasets/vocab.txt'
        with open(vname) as f:
            self.vocabulary = list([ln.strip() for ln in f if len(ln.strip()) > 0])
        with open(vname) as f:
            self.dictionary = dict({ln.strip(): ix for ix, ln in enumerate(f) if len(ln.strip()) > 0})
            self.dictionary[''] = 0
        self.word_count = len(self.vocabulary)

        self.word_dim = 3
        self.embedding = nn.Parameter(th.normal(0, 1, (self.word_dim * self.word_count,)))

        self.labeled_loss = None

    def forward(self, x):
        raise NotImplementedError()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        record = '%2.5f-%03d.ckpt' % (self.labeled_loss, checkpoint['epoch'])
        if len(record) < 12:
            record = '0' * (12 - len(record)) + record
        fname = 'best-%s' % record
        with open(fname, 'bw') as f:
            pickle.dump(checkpoint, f)
        for ix, ckpt in enumerate(sorted(glob.glob('best-*.ckpt'))):
            if ix > 3:
                os.unlink(ckpt)
