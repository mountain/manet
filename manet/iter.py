import torch as th

from torch import nn

from typing import TypeVar, Type

It: Type = TypeVar('It', bound='IterativeMap')


class IterativeMap(nn.Module):
    def __init__(self: It, num_steps: int = 3) -> None:
        super().__init__()
        self.num_steps = num_steps

    def before_forward(self: It, data: th.Tensor) -> th.Tensor:
        return data

    def pre_transform(self: It, data: th.Tensor) -> th.Tensor:
        return data

    def befoer_mapping(self: It, data: th.Tensor) -> th.Tensor:
        return data

    def mapping(self: It, data: th.Tensor) -> th.Tensor:
        raise NotImplemented()

    def after_mapping(self: It, data: th.Tensor) -> th.Tensor:
        return data

    def post_transform(self: It, data: th.Tensor) -> th.Tensor:
        return data

    def after_forward(self: It, data: th.Tensor) -> th.Tensor:
        return data

    def forward(self: It, data: th.Tensor) -> th.Tensor:
        data = self.before_forward(data)

        data = self.pre_transform(data)
        for ix in range(self.num_steps):
            data = self.befor_mapping(data)
            data = self.mapping(data)
            data = self.after_mapping(data)
        data = self.post_transform(data)

        data = self.after_forward(data)
        return data
