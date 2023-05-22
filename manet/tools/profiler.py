from aspectlib import Aspect, Proceed, Return


class Profiler:
    def __init__(self):
        self.tb_logger = None
        self.global_step = 0
        self.debug = False

    def find_tb_logger(self):
        import lightning.pytorch.loggers as pl_loggers
        tb_logger = None
        for logger in self.trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                tb_logger = logger.experiment
                break
        if tb_logger is None:
            raise ValueError('TensorBoard Logger not found')
        else:
            self.tb_logger = tb_logger

        return tb_logger
