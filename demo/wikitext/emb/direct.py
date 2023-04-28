import os
from typing import Dict, Any

import torch
import lightning as pl
import os.path as pth

from torch import nn

from manet.mac import MLP
from demo.wikitext.emb.common import EmbeddingModel


def search(nuclears, x):
    dist = (nuclears - x) * (nuclears - x)
    return torch.argmin(dist)


class DirectModel(EmbeddingModel):
    def __init__(self):
        super().__init__()

        fname = 'datasets/frequency-train.txt'
        self.cooccurrence = {}
        self.frequency = {}
        with open(fname) as f:
            for ln in f:
                for k, v in eval(ln).items():
                    if len(k) == 1:
                        frst = k[0]
                        if frst not in self.cooccurrence:
                            self.cooccurrence[frst] = set()
                    if len(k) == 2:
                        frst, scnd = k
                        if frst not in self.cooccurrence:
                            self.cooccurrence[frst] = set()
                        self.cooccurrence[frst].add(scnd)
                        self.frequency[(frst, scnd)] = v

        self.solver = MLP(4, [8, 16, 8, 4])
        self.predictor = MLP(8, [16, 32, 16, 8, 4])

    def solve(self, ctx, emb1, emb2, emb3):
        return self.solver(torch.concat((ctx, emb1, emb2, emb3), dim=1))

    def predict(self, ctx, emb1, emb2, emb3, rels):
        rels = self.predictor(torch.concat((ctx, emb1, emb2, emb3, rels), dim=1))
        pred4 = (emb3 + rels[:, 0:1]) * rels[:, 1:2]
        pred5 = (pred4 + rels[:, 2:3]) * rels[:, 3:4]

        return pred4, pred5, rels

    def generate(self, ctx, ix1, ix2, ix3):
        while True:
            emb1, emb2, emb3 = self.embedding[ix1], self.embedding[ix2], self.embedding[ix3]
            ctx, emb1, emb2, emb3 = ctx.view(1, 1), emb1.view(1, 1), emb2.view(1, 1), emb3.view(1, 1)
            _ctx = ctx.clone()
            rels = self.solve(ctx, emb1, emb2, emb3)
            ctx = (ctx + emb1) / 2
            ctx = (ctx + emb2) / 2
            ctx = (ctx + emb3) / 2
            pred4, pred5, rels = self.predict(ctx, emb1, emb2, emb3, rels)
            nuclears, indexes = self.make_nuclears(self.vocabulary[ix3])
            ix4 = indexes[search(nuclears, pred4)].item()
            ctx = (_ctx + emb1) / 2
            yield ix4
            nuclears, indexes = self.make_nuclears(self.vocabulary[ix4])
            ix5 = indexes[search(nuclears, pred5)].item()
            ctx = (ctx + emb2) / 2
            yield ix5
            ix1, ix2, ix3 = ix3, ix4, ix5

    def make_nuclears(self, first):
        nuclears, indexes = [], []
        dictionary = {self.dictionary[second]: self.embedding[self.dictionary[second]] for second in self.cooccurrence[first]}
        for k, v in sorted(dictionary.items(), key=lambda x: x[1]):
            nuclears.append(v)
            indexes.append(k)
        return torch.stack(nuclears), torch.tensor(indexes)

    def complete(self, prompt):
        ctx = 0
        for wd in prompt[:-3]:
            ix = self.dictionary[wd]
            emb = self.embedding[ix]
            ctx = (ctx + emb) / 2
        wd1, wd2, wd3 = prompt[-3:]
        ix1, ix2, ix3 = self.dictionary[wd1], self.dictionary[wd2], self.dictionary[wd3]
        counter = 0
        for ix4 in self.generate(ctx, ix1, ix2, ix3):
            print(self.vocabulary[ix4], end=' ')
            counter += 1
            if counter > 20:
                break

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


def _model_():
    return DirectModel()
