import torch as th

from torch import nn

from typing import TypeVar, Type

It: Type = TypeVar('It', bound='IterativeMap')


class IterativeMap(nn.Module):
    def __init__(self: It, num_steps: int = 3) -> None:
        super().__init__()
        self.num_steps = num_steps

    def post_forward(self: It, data: th.Tensor) -> th.Tensor:
        return data

    def pre_mapping(self: It, data: th.Tensor) -> th.Tensor:
        return data

    def post_mapping(self: It, data: th.Tensor) -> th.Tensor:
        return data

    def mapping(self: It, data: th.Tensor) -> th.Tensor:
        raise NotImplemented()

    def forward(self: It, data: th.Tensor) -> th.Tensor:
        data = self.pre_forward(data)

        for ix in range(self.num_steps):
            data = self.pre_mapping(data)
            data = self.mapping(data)
            data = self.post_mapping(data)

        data = self.post_forward(data)

        return data
