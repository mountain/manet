ctx = {
    'stage': None,
    'debug': False,
    'counter': 0,
    'global_step': 0,
    'sampling_steps': 10000,
    'num_samples': 20,
    'tb_logger': None,
    'trainer': None,
    'ground_truth': None,
    'prediction': None,
}


def bind_profiling_context(**kwargs):
    import lightning as pl
    import lightning.pytorch.loggers as pl_loggers
    ctx.update(kwargs)
    ctx['counter'] = 0
    ctx['global_step'] = 0

    trainer: pl.Trainer = ctx['trainer']
    if trainer is None:
        raise ValueError('Trainer not found')
    else:
        for logger in trainer.loggers:
            if isinstance(logger, pl_loggers.TensorBoardLogger):
                ctx['tb_logger'] = logger.experiment
                break
        if ctx['tb_logger'] is None:
            raise ValueError('TensorBoard Logger not found')

    return ctx


def reset_profiling_stage(stage, **kwargs):
    ctx['stage'] = stage
    ctx['global_step'] = 0
    ctx.update(kwargs)


class Profiler:
    def __init__(self, dkey: str = None):
        self.dkey = dkey
        self.order = None

    def initialize(self):
        if self.order is None:
            self.order = ctx['counter']
            ctx['counter'] += 1
