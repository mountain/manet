

class SplineFunction(ExprFlow):

    def __init__(self: U, num_steps: int = 3, num_points: int = 5, debug: bool = False, debug_key: str = None, logger: TensorBoardLogger = None) -> None:
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

    def handler(self: P, data: Tensor) -> Tensor:
        return th.sigmoid(data * self.alpha + self.beta) * self.num_points

    def begin_end_of(self: P, handler: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        bgn = handler.floor().long()
        end = bgn + 1
        bgn = (bgn * (bgn + 1 < self.num_points) + (bgn - 1) * (bgn + 1 >= self.num_points)) * (bgn >= 0)
        end = (end * (end < self.num_points) + (end - 1) * (end == self.num_points)) * (end >= 0)
        t = handler - bgn
        return bgn, end, t

    def spline(self: U, handler: Tensor) -> Tensor:
        bgn, end, t = self.begin_end_of(handler)
        p0, p1 = self.value[bgn], self.value[end]
        m0, m1 = self.derivative[bgn], self.derivative[end]

        # Cubic Hermite spline
        q1 = (2 * t ** 3 - 3 * t ** 2 + 1) * p0
        q2 = (t ** 3 - 2 * t ** 2 + t) * m0
        q3 = (-2 * t ** 3 + 3 * t ** 2) * p1
        q4 = (t ** 3 - t ** 2) * m1

        return q1 + q2 + q3 + q4

    def plot_total_function(self: F) -> Tensor:
        line = th.linspace(0, 1, 1000).view(1, 1000)
        curve = self.spline(line)

        import matplotlib.pyplot as plt
        x, y = line[0].detach().cpu().numpy(), curve[0].detach().cpu().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, y)
        return fig

    def forward(self: F, data: Tensor) -> Tensor:
        sz = data.size()

        if self.debug and self.logger is not None:
            if sz[0] > self.num_samples:
                self.logger.add_figure('%s:0:function:total' % self.debug_key, self.plot_total_function(), self.global_step)

        data = data.view(-1, sz[1], sz[2] * sz[3])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.channel_transform)
        data = data.view(-1, sz[2] * sz[3], sz[1])

        if self.debug and self.logger is not None:
            if sz[0] > self.num_samples:
                for ix in range(self.num_samples):
                    self.logger.add_histogram('%s:1:before:%d:histo' % (self.debug_key, self.labels[ix]), data[ix], self.global_step)
                    image = data[ix].view(sz[2], sz[3] * sz[1])
                    image = (image - image.min()) / (image.max() - image.min())
                    self.logger.add_image('%s:1:before:%d:image' % (self.debug_key, self.labels[ix]), image, self.global_step, dataformats='HW')

        for ix in range(self.num_steps):
            handler = self.handler(data)
            data = self.spline(handler)

        if self.debug and self.logger is not None:
            if sz[0] > self.num_samples:
                for ix in range(self.num_samples):
                    self.logger.add_histogram('%s:2:after:%d:histo' % (self.debug_key, self.labels[ix]), data[ix], self.global_step)
                    image = data[ix].view(sz[2], sz[3] * sz[1])
                    image = (image - image.min()) / (image.max() - image.min())
                    self.logger.add_image('%s:2:after:%d:image' % (self.debug_key, self.labels[ix]), image, self.global_step, dataformats='HW')

        data = data.view(-1, sz[2] * sz[3], sz[1])
        data = th.permute(data, [0, 2, 1]).reshape(-1, 1)
        data = th.matmul(data, self.spatio_transform)
        data = data.view(*sz)

        return data
