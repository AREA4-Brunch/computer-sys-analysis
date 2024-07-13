import inspect
import functools


def num_remaining_args(func: callable) -> int:
    """ Returns number of args `func` takes in that are not already
        bound to it, in case of method bound to instance or func
        created using functools.partial.
    """
    provided_args_cnt = 0
    if isinstance(func, (functools.partial, functools.partialmethod)):
        provided_args_cnt = len(func.args) + len(func.keywords)
        func = func.func
    # everything that was bound to func will not be in parameters
    singature = inspect.signature(func)
    total_num_args = len(singature.parameters.values())
    return total_num_args - provided_args_cnt
