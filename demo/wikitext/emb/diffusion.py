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
        self.rotate = MLP(2, [8, 32, 8, 1])
        self.context = MLP(3, [12, 48, 12, 1])

    def diffuse_step(self, ctx, theta, emb):
        next_theta = theta + self.rotate(th.cat((ctx, emb), dim=-1)).view(-1, 1, 1)
        next_emb = emb + th.cos(next_theta) + emb * th.sin(next_theta)
        next_ctx = self.context(th.cat((ctx, next_theta, next_emb), dim=-1)).view(-1, 1, 1)
        return next_ctx, next_theta, next_emb

    def step(self, key, batch):
        tokens = self.embedding[batch].view(-1, 1, default_steps)
        token = tokens[:, :, 0:1].view(-1, 1, 1)
        theta = th.zeros_like(token)
        ctx = th.zeros_like(token)
        sequence = []
        for ix in range(default_steps):
            ctx, theta, token = self.diffuse_step(ctx, theta, token)
            sequence.append(token)
        sequence = th.cat(sequence, dim=2)
        embedding = self.embedding.view(1, -1, 1)
        pred = F.log_softmax(1 - th.tanh((sequence - embedding) ** 2), dim=1)

        loss = nll(pred, batch)
        self.log_messages(key, loss=loss)
        return loss


def _model_():
    return DiffusionModel()
