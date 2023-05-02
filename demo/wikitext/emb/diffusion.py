import collections

import torch as th
import torch.nn.functional as F
from torch.nn import NLLLoss
from transformers import AutoTokenizer

from manet.mac import MLP
from demo.wikitext.emb.common import EmbeddingModel

nll = NLLLoss()


default_steps = 18


class DiffusionModel(EmbeddingModel):
    def __init__(self):
        super().__init__()
        l = default_steps * self.word_dim
        self.handler = MLP(l, [3 * 4 * l, 3 * 16 * l, 3 * 4 * l, 3 * self.word_dim])
        self.pmemory = collections.deque(maxlen=default_steps // 3)
        self.qmemory = collections.deque(maxlen=default_steps // 3)
        self.rmemory = collections.deque(maxlen=default_steps // 3)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def clear(self, token):
        for ix in range(default_steps // 3):
            self.pmemory.append(th.zeros_like(token))
            self.qmemory.append(th.zeros_like(token))
            self.rmemory.append(th.zeros_like(token))

    def diffuse_step(self, ctx, theta, emb):
        self.pmemory.append(emb)
        self.qmemory.append(theta)
        self.rmemory.append(ctx)
        memory = th.cat(list(self.pmemory) + list(self.qmemory) + list(self.rmemory), dim=-1)
        result = self.handler(memory).view(-1, 1, 3, self.word_dim)
        dctx = th.tanh(result[:, :, 0:1, :])
        dtheta = th.tanh(result[:, :, 1:2, :]) * th.pi
        velocity = th.sigmoid(result[:, :, 2:3, :])
        next_ctx = ctx + dctx
        next_theta = theta + dtheta
        next_emb = emb + th.cos(next_theta) * velocity + emb * th.sin(next_theta) * velocity
        return next_ctx, next_theta, next_emb

    def step(self, key, batch):
        tokens = self.embedding[batch].view(-1, 1, default_steps, self.word_dim)
        token = tokens[:, :, 0:1, :].view(-1, 1, 1, self.word_dim)
        theta = th.zeros_like(token)
        ctx = th.zeros_like(token)
        self.clear(token)

        sequence = []
        for ix in range(default_steps):
            if ix < default_steps // 3:
                token = tokens[:, :, ix:ix+1, :]
            sequence.append(token)
            ctx, theta, token = self.diffuse_step(ctx, theta, token)

        sequence = th.cat(sequence[default_steps // 3:], dim=2)
        embedding = self.embedding.view(1, -1, 1, self.word_dim)
        dist = th.sum((sequence - embedding) ** 2, dim=-1)
        pred = F.log_softmax(3 * (1 - th.tanh(dist)), dim=1)
        penalty = (th.std(pred, dim=2).mean() - th.std(tokens[:, :, default_steps // 3:], dim=2).mean()) ** 2
        loss = nll(pred, batch[:, default_steps // 3:]) + penalty

        self.log_messages(key, loss=loss, penalty=penalty, batch_size=batch.shape[0])
        return loss

    def get_embedding(self, word):
        embedding = self.embedding.view(1, -1, 1)
        try:
            ix = self.dictionary[word]
        except KeyError:
            ix = 0
        return embedding[0:1, ix:ix+1, 0:1]

    def generate(self, ctx, theta, emb):
        embedding = self.embedding.view(1, -1, 1)
        while True:
            ctx, theta, emb = self.diffuse_step(ctx, theta, emb)
            ix = th.argmin((embedding - emb) ** 2, dim=1).item()
            emb = embedding[0:1, ix:ix+1, 0:1]
            yield ix

    def complete(self, prompt):
        ctx = th.zeros(1, 1, 1)
        theta = th.zeros(1, 1, 1)
        emb = th.zeros(1, 1, 1)
        self.clear(emb)
        tokens = self.tokenizer.tokenize(' '.join(prompt))
        for token in tokens:
            emb = self.get_embedding(token)
            ctx, theta, token = self.diffuse_step(ctx, theta, emb)

        counter = 0
        ctx, theta, token = self.diffuse_step(ctx, theta, emb)
        for ix in self.generate(ctx, theta, token):
            print(self.vocabulary[ix], end=' ')
            counter += 1
            if counter > 20:
                break


def _model_():
    return DiffusionModel()
