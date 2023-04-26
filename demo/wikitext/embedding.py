import numpy as np
import torch
import lightning as pl
import os.path as pth

from torch import nn

from manet.mac import MLP


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.lookup = {'\n': 0}
        fname = '%s/vocabulary.txt' % pth.dirname(__file__)
        with open(fname, mode='r') as f:
            for ix, wd in enumerate(f):
                wd = wd.strip()
                self.lookup[wd] = ix
        self.word_count = ix + 1
        self.embedding = nn.Parameter(torch.normal(0, np.sqrt(self.word_count), (1, self.word_count)))

        self.solver = MLP(4, [8, 16, 8, 4])
        self.predictor = MLP(8, [16, 32, 16, 8, 4])

        import collections
        self.q = collections.deque(maxlen=3)
        self.p = collections.deque(maxlen=2)

    def word_embedding(self, word):
        try:
            ix = self.lookup[word]
        except KeyError:
            ix = self.lookup['<unk>']
        return self.embedding[:, ix:ix+1]

    def solve(self, ctx, emb1, emb2, emb3):
        return self.solver(torch.concat((ctx, emb1, emb2, emb3), dim=1))

    def predict(self, ctx, emb1, emb2, emb3, rels):
        rels = self.predictor(torch.concat((ctx, emb1, emb2, emb3, rels), dim=1))
        pred4 = (emb3 + rels[:, 0:1]) * rels[:, 1:2]
        pred5 = (pred4 + rels[:, 2:3]) * rels[:, 3:4]
        return pred4, pred5

    def learn(self, last_length, loss_rel, loss_emb, paragraph):
        words = paragraph.split(' ')
        length = len(words)
        empty = self.word_embedding('')
        error_rel, error_emb = torch.zeros_like(empty), torch.zeros_like(empty)
        if length < 7:
            length = last_length
        else:
            ctx = torch.zeros_like(empty)
            self.q.append(self.word_embedding(words[0]))
            ctx = (ctx + self.q[-1]) / 2
            self.q.append(self.word_embedding(words[1]))
            ctx = (ctx + self.q[-1]) / 2
            self.q.append(self.word_embedding(words[2]))
            ctx = (ctx + self.q[-1]) / 2
            self.p.append(self.word_embedding(words[3]))
            self.p.append(self.word_embedding(words[4]))
            for ix in range(5, length):
                rels = self.solve(ctx, self.q[0], self.q[1], self.q[2])
                error12 = (self.q[0] + rels[:, 0:1]) * rels[:, 1:2] - self.q[1]
                error23 = (self.q[1] + rels[:, 2:3]) * rels[:, 3:4] - self.q[2]
                error_rel = error_rel + error12 * error12 + error23 * error23

                pred4, pred5 = self.predict(ctx, self.q[0], self.q[1], self.q[2], rels)
                error34 = pred4 - self.p[0]
                error45 = pred5 - self.p[1]
                error_emb = error_emb + error34 * error34 + error45 * error45

                self.q.append(self.p.popleft())
                self.p.append(self.word_embedding(words[ix]))
                ctx = (ctx + self.q[-1]) / 2

        length = last_length + length
        loss_rel, loss_emb = loss_rel + torch.sum(error_rel), loss_emb + torch.sum(error_emb)
        return length, loss_rel, loss_emb

    def log_messages(self, key, loss_rel, loss_emb, loss):
        self.log(key, loss, prog_bar=True, batch_size=1)
        self.log('loss_rel', loss_rel, prog_bar=True, batch_size=1)
        self.log('loss_emb', loss_emb, prog_bar=True, batch_size=1)
        self.log('max', self.embedding.max().item(), prog_bar=True, batch_size=1)
        self.log('min', self.embedding.min().item(), prog_bar=True, batch_size=1)
        self.log('mean', self.embedding.mean().item(), prog_bar=True, batch_size=1)
        self.log('std', torch.std(self.embedding).item(), prog_bar=True, batch_size=1)

    def forward(self, x):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def step(self, key, batch):
        empty = self.word_embedding('')
        length, loss_rel, loss_emb = 0, torch.sum(empty.clone()), torch.sum(empty.clone())
        for paragraph in batch:
            length, loss_rel, loss_emb = self.learn(length, loss_rel, loss_emb, paragraph)

        if length == 0:
            loss = loss_rel + loss_emb
        else:
            loss_rel = loss_rel / length
            loss_emb = loss_emb / length
            loss = loss_rel + loss_emb

        self.log_messages('%s_loss' % key, loss_rel, loss_emb, loss)
        return loss

    def training_step(self, train_batch, batch_idx):
        return self.step('train', train_batch)

    def validation_step(self, val_batch, batch_idx):
        self.step('valid', val_batch)

    def test_step(self, test_batch, batch_idx):
        self.step('test', test_batch)


def _model_():
    return Model()
