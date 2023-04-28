import numpy as np
import torch
import lightning as pl
import os.path as pth

from torch import nn

from manet.mac import MLP


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        vname = '%s/vocabulary.txt' % pth.dirname(__file__)
        with open(vname) as f:
            self.vocabulary = list([ln.strip() for ln in f if len(ln.strip()) > 0])
        self.word_count = len(self.vocabulary)

        self.embedding = nn.Parameter(torch.normal(0, 1, (self.word_count,)))

        self.solver = MLP(4, [8, 16, 8, 4])
        self.predictor = MLP(8, [16, 32, 16, 8, 4])

    def solve(self, ctx, emb1, emb2, emb3):
        return self.solver(torch.concat((ctx, emb1, emb2, emb3), dim=1))

    def predict(self, ctx, emb1, emb2, emb3, rels):
        rels = self.predictor(torch.concat((ctx, emb1, emb2, emb3, rels), dim=1))
        pred4 = (emb3 + rels[:, 0:1]) * rels[:, 1:2]
        pred5 = (pred4 + rels[:, 2:3]) * rels[:, 3:4]

        return pred4, pred5, rels

    def log_messages(self, key, loss_rel, loss_emb, loss_amb, loss):
        self.log(key, loss, prog_bar=True, batch_size=64)
        self.log('loss_rel', loss_rel, prog_bar=True, batch_size=64)
        self.log('loss_emb', loss_emb, prog_bar=True, batch_size=64)
        self.log('loss_amb', loss_amb, prog_bar=True, batch_size=64)
        self.log('max', self.embedding.max().item(), prog_bar=True, batch_size=64)
        self.log('min', self.embedding.min().item(), prog_bar=True, batch_size=64)
        self.log('mean', self.embedding.mean().item(), prog_bar=True, batch_size=64)
        self.log('std', torch.std(self.embedding).item(), prog_bar=True, batch_size=64)

    def forward(self, x):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def step(self, key, batch):
        embedding = self.embedding[batch]
        ctx = embedding[:, 0:1]
        for ix in range(1, 7):
            ctx = (ctx + embedding[:, ix:ix+1]) / 2

        rels = self.solve(ctx, embedding[:, 7:8], embedding[:, 8:9], embedding[:, 9:10])
        error12 = (embedding[:, 7:8] + rels[:, 0:1]) * rels[:, 1:2] - embedding[:, 8:9]
        error23 = (embedding[:, 8:9] + rels[:, 2:3]) * rels[:, 3:4] - embedding[:, 9:10]
        loss_rel = torch.mean(error12 * error12 + error23 * error23)

        ctx = (ctx + embedding[:, 7:8]) / 2
        ctx = (ctx + embedding[:, 8:9]) / 2
        ctx = (ctx + embedding[:, 9:10]) / 2
        pred4, pred5, rels = self.predict(ctx, embedding[:, 7:8], embedding[:, 8:9], embedding[:, 9:10], rels)
        error34 = pred4 - embedding[:, 10:11]
        error45 = pred5 - embedding[:, 11:12]
        loss_emb = torch.mean(error34 * error34 + error45 * error45)

        nrels = self.solve(ctx, embedding[:, 9:10], pred4, pred5)
        loss_amb = torch.mean((rels - nrels) * (rels - nrels))

        loss = loss_rel + loss_emb + loss_amb
        self.log_messages('%s_loss' % key, loss_rel, loss_emb, loss_amb, loss)

        return loss

    def training_step(self, train_batch, batch_idx):
        return self.step('train', train_batch)

    def validation_step(self, val_batch, batch_idx):
        self.step('valid', val_batch)

    def test_step(self, test_batch, batch_idx):
        self.step('test', test_batch)


def _model_():
    return Model()
