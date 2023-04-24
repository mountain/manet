import torch

from torch import nn
from torch import Tensor

from torch.nn import functional as F
from typing import Optional, List, Union
from torch.nn.common_types import _size_2_t

from manet.mac import MacUnit


class Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        padding: Union[str, _size_2_t] = 0,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                         bias=False, padding_mode=padding_mode, device=device, dtype=dtype)
        kw, kh = self.kernel_size
        self.kernel = MacUnit(in_channels, out_channels, kw * kh, 1)
        self.device = device
        self.dtype = dtype

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        b, i, w, h = input.size()
        kw, kh = self.kernel_size
        p = self.padding[0]
        if self.padding_mode == 'zeros':
            padded = F.pad(input, (p, p, p, p), 'constant', 0)
        else:
            padded = F.pad(input, (p, p, p, p), self.padding_mode)
        result = torch.zeros(b, self.out_channels, w, h).to(self.device).to(self.dtype)
        for m in range(w):
            for n in range(h):
                pointer = torch.zeros(1, 1, w, h).to(self.device).to(self.dtype)
                pointer[0, 0, m, n] = 1
                piece = padded[:, :, m:m+kw, n:n+kh]
                piece = piece.reshape(b, self.in_channels, kw * kh)
                result += self.kernel(piece).reshape(b, self.out_channels, 1, 1) * pointer
        return result
