import aspectlib

from aspectlib import Aspect, Proceed, Return


@Aspect
def strip_return_value(*args, **kwargs):
    result = yield Proceed
    yield Return(result.strip())
