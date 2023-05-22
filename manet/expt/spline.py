import torch as th

from torch import nn
from manet.iter import IterativeMap

from torch import Tensor
from typing import TypeVar, Type, Tuple

from manet.tools.ploter import plot_iterative_function, plot_image, plot_histogram

Sp: Type = TypeVar('Sp', bound='SplineFunction')


class SplineFunction(IterativeMap):

    def __init__(self: Sp, num_steps: int = 3, num_points: int = 5, debug: bool = False, debug_key: str = None, logger: TensorBoardLogger = None) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.num_points = num_points
        self.alpha = nn.Parameter(th.ones(1, 1, 1))
        self.beta = nn.Parameter(th.zeros(1, 1, 1))
        self.value = nn.Parameter(th.linspace(0, 1, num_points).view(num_points))
        self.derivative = nn.Parameter(th.ones(num_points).view(num_points) * 2 * th.pi / num_points)
        self.channel_transform = nn.Parameter(th.normal(0, 1, (1, 1)))
        self.spatio_transform = nn.Parameter(th.normal(0, 1, (1, 1)))

        self.debug = debug
        self.debug_key = debug_key
        self.logger = logger
        self.global_step = 0
        self.labels = None
        self.num_samples = 20

    def begin_end_of(self: Sp, handler: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        bgn = handler.floor().long()
        end = bgn + 1
        bgn = (bgn * (bgn + 1 < self.num_points) + (bgn - 1) * (bgn + 1 >= self.num_points)) * (bgn >= 0)
        end = (end * (end < self.num_points) + (end - 1) * (end == self.num_points)) * (end >= 0)
        t = handler - bgn
        return bgn, end, t

    @plot_iterative_function
    def before_forward(self: Sp, data: Tensor) -> Tensor:
        self.size = data.size()
        return data

    @plot_image
    @plot_histogram
    def pre_transform(self: Sp, data: Tensor) -> Tensor:
        sz = self.size
        data = data.view(-1, sz[1], sz[2] * sz[3])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.channel_transform)
        data = data.view(-1, sz[2] * sz[3], sz[1])
        data = th.sigmoid(data)
        return data

    def befoer_mapping(self: Sp, data: th.Tensor) -> th.Tensor:
        self.handle = th.sigmoid(data * self.alpha + self.beta) * self.num_points
        return data

    def mapping(self: Sp, data: th.Tensor) -> th.Tensor:
        bgn, end, t = self.begin_end_of(self.handle)
        p0, p1 = self.value[bgn], self.value[end]
        m0, m1 = self.derivative[bgn], self.derivative[end]

        # Cubic Hermite spline
        q1 = (2 * t ** 3 - 3 * t ** 2 + 1) * p0
        q2 = (t ** 3 - 2 * t ** 2 + t) * m0
        q3 = (-2 * t ** 3 + 3 * t ** 2) * p1
        q4 = (t ** 3 - t ** 2) * m1

        return q1 + q2 + q3 + q4

    @plot_image
    @plot_histogram
    def post_transform(self: Sp, data: Tensor) -> Tensor:
        sz = self.size
        data = data.view(-1, sz[2] * sz[3], sz[1])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.spatio_transform)
        data = data.view(*sz)
        return data
