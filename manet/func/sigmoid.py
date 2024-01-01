import numpy as np
import torch as th
from torch import Tensor

###
# Sigmoid functions
#
# The sigmoid functions are used to map the output of the network to a value between 0 and 1.
#
# The sigmoid functions are ordered by their curve steepness. The first is the most steep one.
#   - alg1: Algebraic sigmoid function
#   - natan: Normalized arctangent
#   - alg2: Algebraic sigmoid function
#   - ngd: Normalized Gudermannian function
#   - ntanh: hyperbolic tangent
#   - nerf: Normalized error function


def alg1(x: float | Tensor) -> float | Tensor:
    return (x / (1 + th.abs(x)) + 1) / 2


def natan(x: float | Tensor) -> float | Tensor:
    return (th.atan(x * th.pi / 2) / th.pi * 2 + 1) / 2


def alg2(x: float | Tensor) -> float | Tensor:
    return (x / th.sqrt(1 + x * x) + 1) / 2


def gd(x: float | Tensor) -> float | Tensor:
    return 2 * th.atan(th.tanh(x / 2))


def ngd(x: float | Tensor) -> float | Tensor:
    return (gd(x * th.pi / 2) / th.pi * 2 + 1) / 2


def ntanh(x: float | Tensor) -> float | Tensor:
    return (th.tanh(x) + 1) / 2


c = np.sqrt(np.pi) / 2


def nerf(x: float | Tensor) -> float | Tensor:
    return (th.erf(x * c) + 1) / 2


functions = {
    'alg1': alg1,
    'natan': natan,
    'alg2': alg2,
    'ngd': ngd,
    'ntanh': ntanh,
    'nerf': nerf
}
