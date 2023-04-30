import collections

import torch as th
import torch.nn.functional as F
from torch.nn import NLLLoss

from manet.mac import MLP
from demo.wikitext.emb.common import EmbeddingModel

nll = NLLLoss()


default_steps = 18


class DiffusionModel(EmbeddingModel):
    def __init__(self):
        super().__init__()
        l = default_steps
        self.handler = MLP(l, [3 * 4 * l, 3 * 16 * l, 3 * 4 * l, 3])
        self.pmemory = collections.deque(maxlen=default_steps // 3)
        self.qmemory = collections.deque(maxlen=default_steps // 3)
        self.rmemory = collections.deque(maxlen=default_steps // 3)

    def diffuse_step(self, ctx, theta, emb):
        self.pmemory.append(emb)
        self.qmemory.append(theta)
        self.rmemory.append(ctx)
        memory = th.cat(list(self.pmemory) + list(self.qmemory) + list(self.rmemory), dim=-1)
        result = self.handler(memory).view(-1, 1, 3)
        dctx = th.tanh(result[:, :, 0:1])
        dtheta = th.tanh(result[:, :, 1:2]) * th.pi
        velocity = th.sigmoid(result[:, :, 2:3])
        next_ctx = ctx + dctx
        next_theta = theta + dtheta
        next_emb = emb + th.cos(next_theta) * velocity + emb * th.sin(next_theta) * velocity
        return next_ctx, next_theta, next_emb

    def step(self, key, batch):
        tokens = self.embedding[batch].view(-1, 1, default_steps)
        token = tokens[:, :, 0:1].view(-1, 1, 1)
        theta = th.zeros_like(token)
        ctx = th.zeros_like(token)

        for ix in range(default_steps // 3):
            self.pmemory.append(th.zeros_like(token))
            self.qmemory.append(th.zeros_like(token))
            self.rmemory.append(th.zeros_like(token))

        sequence = []
        for ix in range(default_steps):
            ctx, theta, token = self.diffuse_step(ctx, theta, token)
            sequence.append(token)

        sequence = th.cat(sequence[default_steps // 3:], dim=2)
        embedding = self.embedding.view(1, -1, 1)
        pred = F.log_softmax(1 - th.tanh((sequence - embedding) ** 2), dim=1)
        loss = nll(pred, batch[:, default_steps // 3:])

        self.log_messages(key, loss=loss)
        return loss


def _model_():
    return DiffusionModel()
