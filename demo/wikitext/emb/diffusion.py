import collections

import torch as th
import torch.nn.functional as F
from torch.nn import NLLLoss, LogSoftmax
from transformers import AutoTokenizer

from manet.mac import MLP
from demo.wikitext.emb.common import EmbeddingModel


nll = NLLLoss()
log_softmax = LogSoftmax(dim=2)


default_steps = 18


class DiffusionModel(EmbeddingModel):
    def __init__(self):
        super().__init__()
        self.l = default_steps // 3 * 2
        self.c = self.l
        self.encoder = MLP(self.l, [2 * self.l, 4 * self.l, self.c], spatio_dims=self.word_dim)
        self.decoder = MLP(self.c, [2 * self.l, 4 * self.l, 2], spatio_dims=self.word_dim)
        self.ulearner = MLP(3 * self.c, [8 * self.c, 4 * self.c, 8 * self.c], spatio_dims=self.word_dim)
        self.pmemory = collections.deque(maxlen=default_steps // 3)
        self.qmemory = collections.deque(maxlen=default_steps // 3)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def memory(self):
        return th.cat(list(self.pmemory) + list(self.qmemory), dim=1)

    def clear(self, token):
        token = token.view(-1, 1, self.word_dim)
        for ix in range(default_steps // 3):
            self.pmemory.append(th.zeros_like(token))
            self.qmemory.append(th.zeros_like(token))

    def diffuse_step(self, context, theta, emb):
        emb = emb.view(-1, 1, self.word_dim)
        theta = theta.view(-1, 1, self.word_dim)


        self.pmemory.append(emb)
        self.qmemory.append(theta)

        inputs = self.encoder(self.memory())
        if context is None:
            context = th.zeros_like(inputs)
        context = context.view(-1, self.c, self.word_dim)
        inputs = inputs.view(-1, self.c, self.word_dim)


        lastr, lasts = th.ones_like(context), th.ones_like(context)
        dc, do = th.clone(context), th.clone(context)
        context, output = th.zeros_like(context), th.zeros_like(context)
        for _ in range(3):
            state = th.sigmoid(self.ulearner(th.cat((context, inputs, output), dim=1)))
            state = state.view(-1, 8, self.c, self.word_dim)
            p, r, t, v = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
            q, s, u, w = state[:, 4], state[:, 5], state[:, 6], state[:, 7]
            p, q = 4 * p, 4 * q

            r, lastr = r * lastr, r
            s, lasts = s * lasts, s

            dc = th.fmod((1 - dc) * dc * p + inputs, 1) * r + dc * (1 - r)
            dc = dc * t * (1 - r) + dc * r
            do = th.fmod((1 - do) * do * q + inputs, 1) * s + do * (1 - s)
            do = do * r * (1 - s) + do * s
            context = context + dc
            output = output + do


        result = self.decoder(output).view(-1, 2, self.word_dim)

        dtheta = th.tanh(result[:, 0:1, :]) * th.pi
        velocity = th.sigmoid(result[:, 1:2, :])
        next_theta = theta + dtheta
        next_emb = emb + th.cos(next_theta) * velocity + emb * th.sin(next_theta) * velocity

        return context.view(-1, self.c, 1, self.word_dim), next_theta.view(-1, 1, 1, self.word_dim), next_emb.view(-1, 1, 1, self.word_dim)

    def step(self, key, batch):
        batch = batch.view(-1, default_steps)
        rebatch = (batch * self.word_dim).view(-1, default_steps, 1)
        batch = th.cat([rebatch + ix for ix in range(self.word_dim)], dim=2)
        batch = batch.view(-1, default_steps * self.word_dim)

        tokens = self.embedding[batch].view(-1, 1, default_steps, self.word_dim)
        token = tokens[:, :, 0:1, :].view(-1, 1, 1, self.word_dim)
        theta = th.zeros_like(token)
        self.clear(token)

        sequence = []
        context = None
        for ix in range(default_steps):
            if ix < default_steps // 3:
                token = tokens[:, :, ix:ix+1, :]
            sequence.append(token)
            context, theta, token = self.diffuse_step(context, theta, token)

        sequence = th.cat(sequence[default_steps // 3:], dim=2)
        embedding = self.embedding.view(1, -1, 1, self.word_dim)
        dist = th.sum((sequence - embedding) ** 2, dim=3)
        pred = F.log_softmax(3 * (1 - th.tanh(dist)))
        batch = batch.view(-1, default_steps, self.word_dim)[:, :, 0] // self.word_dim
        loss = nll(pred, batch[:, default_steps // 3:])

        self.log_messages(key, loss=loss, batch_size=batch.size()[0])
        return loss

    def get_embedding(self, word):
        embedding = self.embedding.view(1, 1, -1, 1)
        try:
            ix = self.dictionary[word]
        except KeyError:
            ix = 0
        return embedding[0:1, 0:1, ix:ix+1, 0:1]

    def generate(self, ctx, theta, emb):
        embedding = self.embedding.view(1, -1, 1, 1)
        while True:
            ctx, theta, emb = self.diffuse_step(ctx, theta, emb)
            ix = th.argmin((embedding - emb) ** 2, dim=1).item()
            emb = embedding[0:1, ix:ix+1, 0:1, 0:1]
            yield ix

    def complete(self, prompt):
        ctx = th.zeros(1, 1, 1, 1)
        theta = th.zeros(1, 1, 1, 1)
        emb = th.zeros(1, 1, 1, 1)
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
