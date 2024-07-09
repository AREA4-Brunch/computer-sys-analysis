from typing import Callable


class CallableWithArgs(Callable[[any], any]):
    """ Utility wrapper for delaying function call with args, kwargs or both.
        Useful when args and kwargs should be stored only as long as needed.
    """
    # _method: Callable[[any], tuple[float, any]]
    # stores either args, kwargs or both if provided with method
    # _args_kwargs: tuple[tuple, dict] | tuple[tuple] | tuple[dict]

    def __init__(
        self,
        method: Callable[[any], any],
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self._method = method
        # do not store 2 ptrs to args and kwargs if only 1 is provided
        # so store a tuple with 1 or both
        self._args_kwargs = tuple()
        if len(args): self._args_kwargs += (args, )
        if len(kwargs): self._args_kwargs += (kwargs, )

    def __call__(self, *args, **kwargs) -> any:
        """ Additional args and kwargs can be appended to args and kwargs
            provided in constructor.
        """
        meth_args, meth_kwargs = self._unpack_args_kwargs()
        return self._method(*meth_args, *args, **meth_kwargs, **kwargs)

    def _unpack_args_kwargs(self) -> tuple[tuple, dict]:
        if len(self._args_kwargs) == 2: return self._args_kwargs;
        if isinstance(self._args_kwargs, dict):  # only kwargs was stored
            return (tuple(), self._args_kwargs[0])
        return (self._args_kwargs[0], dict())
