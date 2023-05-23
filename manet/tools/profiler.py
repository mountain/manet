
counter = 0


class Profiler:
    def __init__(self, debug: bool = False, dkey: str = None, num_samples: int = 20):
        self.tb_logger = None
        self.debug = debug
        self.dkey = dkey
        self.order = None
        self.global_step = 0
        self.sampling_steps = 10000
        self.num_samples = num_samples
        self.labels = None

    def initialize(self):
        global counter
        if self.order is None:
            self.order = counter
            counter += 1

        import lightning.pytorch.loggers as pl_loggers
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break
        if tb_logger is None:
            # raise ValueError('TensorBoard Logger not found')
            pass
        else:
            self.tb_logger = tb_logger
