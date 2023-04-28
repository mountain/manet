import torch as th
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from manet.mac import MLP
from demo.wikitext.emb.common import EmbeddingModel

cross_entropy = CrossEntropyLoss()


class DiffusionModel(EmbeddingModel):
    def __init__(self):
        super().__init__()
        self.context = MLP(2, [8, 32, 16, 4, 1])

    def diffuse_step(self, ctx, emb):
        theta = self.context(th.cat((ctx, emb), dim=-1)).view(-1, 1, 1)
        next_emb = emb + th.cos(theta) + emb * th.sin(theta)
        return theta, next_emb

    def step(self, key, batch):
        tokens = self.embedding[batch].view(-1, 1, 12)
        token = tokens[:, :, 0:1].view(-1, 1, 1)
        ctx = th.zeros_like(token)
        sequence = []
        for ix in range(12):
            ctx, token = self.diffuse_step(ctx, token)
            sequence.append(token)
        sequence = th.cat(sequence, dim=2)
        embedding = self.embedding.view(1, -1, 1)
        onehot = F.gumbel_softmax(1 - th.tanh((sequence - embedding) ** 2), dim=1)

        return cross_entropy(onehot, batch)


def _model_():
    return DiffusionModel()
