counter = 0
global_step = 0
tb_logger = None


def bind_profiling_context(trainer):
    import lightning.pytorch.loggers as pl_loggers

    global counter
    global global_step
    global tb_logger

    counter = 0
    global_step = 0

    for logger in trainer.loggers:
        if isinstance(logger, pl_loggers.TensorBoardLogger):
            tb_logger = logger.experiment
            break

    if tb_logger is None:
        raise ValueError('TensorBoard Logger not found')

    return counter, global_step, tb_logger


class Profiler:
    def __init__(self, debug: bool = False, dkey: str = None, num_samples: int = 20):
        self.tb_logger = None
        self.debug = debug
        self.dkey = dkey
        self.order = None
        self.sampling_steps = 10000
        self.num_samples = num_samples
        self.labels = None

    def initialize(self):
        global counter
        if self.order is None:
            self.order = counter
            counter += 1
