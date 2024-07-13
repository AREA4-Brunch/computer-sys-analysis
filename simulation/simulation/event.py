import abc
import functools
from typing import Callable
from .scheduler import ITasksScheduler
from ..utils.callable import num_remaining_args


class ISimulatedEvent(abc.ABC):
    @abc.abstractmethod
    def execute(self, scheduler: ITasksScheduler, *args, **kwargs):
        pass

    @abc.abstractmethod
    def then(self, *args, **kwargs) -> 'ISimulatedEvent':
        """ Should allow for taking in an existing ISimulatedEvent
            or just passing args and creating one within the method.
        """
        pass

    @abc.abstractmethod
    def then_(
        self,
        first_last: tuple['ISimulatedEvent', 'ISimulatedEvent'],
        *args,
        **kwargs
    ) -> 'ISimulatedEvent':
        """ Performs `then` on self using `first` and given args
            and returns `last`.
            Useful to make some code cleaner.
        """
        pass

    @abc.abstractmethod
    def does_take_args(self) -> bool:
        pass


class SimulatedEvent(ISimulatedEvent):
    """ Executes given func and allows other simulated events
        to be executed right after it with the result of original
        func passed to them.

        Represent node in linked list of simulated events.
    """
    # _func: Callable[[any], tuple[float, any]]
    # _nexts: list[ISimulatedEvent]

    def __init__(
        self,
        func: Callable[[any], tuple[float, any]],
        *args,
        **kwargs
    ) -> None:
        """ Given func will be executed with given arguments
            and should return the tuple with duration of its
            execution and its return value in that order.
        """
        super().__init__()
        # create memory efficient wrappers to store args and kwargs
        # to avoid adding attributes to this class if none were provided
        args_missing = num_remaining_args(func) - len(args) - len(kwargs)
        if args_missing > 1:
            raise ValueError(
                f'Given function is yet to be provided {args_missing} args'
                + f', but at most 1 can be returned via event chaining.'
            )
        self._args_kwargs = (args_missing,)
        if len(args): self._args_kwargs += (args, )
        if len(kwargs): self._args_kwargs += (kwargs, )
        self._func = func
        self._nexts = []

    def execute(self, scheduler: ITasksScheduler, *prev_args, **prev_kwargs):
        """ Executes the func provided in constructor with given args
            and kwargs prepended to the ones provided in the
            constructor. Then it executes all other events chained
            via `then` at same scheduled time and to each of them
            prepends the func's result to their args provided in their
            constructors.

            To fetch the result of execution chain a `then`
            before calling this function.
        """
        # do not check self._func takes args to pass prev_args or not;
        # only way for prev_args to be provided if the func does not
        # take any is that if user explicitly provided them in first
        # call of execute so user should handle the error
        fargs, fkwargs = self._unpack_args_kwargs()
        duration, ret_val = self._func(*fargs, *prev_args, **fkwargs, **prev_kwargs)
        for next in self._nexts:
            cur_ret = (ret_val,) if next.does_take_args() else tuple()
            if duration == 0:  # no need to overload scheduler
                next.execute(scheduler, *cur_ret)
            else:
                scheduler.add(duration, next.execute, scheduler, *cur_ret)

    def then(self, *args, **kwargs) -> ISimulatedEvent:
        """ Given event executes after the func provided in
            constructor, there can be any number of such events
            provided.
            Returns the given event as the next one that `then`
            should be called on to chain more events to execute
            after the ones that were linked/`then`ed to this event.
            Instead of providing an event you can provide all
            constructor args and kwargs and `SimulatedEvent` will be
            created immediately; makes code cleaner.
        """
        super().then()
        if not args: raise ValueError(f'then requires arguments');
        is_provided = isinstance(args[0], ISimulatedEvent)
        next = args[0] if is_provided else SimulatedEvent(*args, **kwargs)
        self._nexts.append(next)
        return next

    def then_(
        self,
        first_last: tuple[ISimulatedEvent, ISimulatedEvent],
        *args,
        **kwargs
    ) -> ISimulatedEvent:
        self.then(first_last[0], *args, **kwargs)
        return first_last[1]

    def does_take_args(self) -> bool:
        return self._args_kwargs[0] > 0

    def _unpack_args_kwargs(self) -> tuple[tuple, dict]:
        if len(self._args_kwargs) == 2: 
            if isinstance(self._args_kwargs[1], dict):  # only kwargs stored
                return (tuple(), self._args_kwargs[1])
            return (self._args_kwargs[1], dict())
        if len(self._args_kwargs) == 3: return self._args_kwargs;
        return (tuple(), dict())


def simulated_func(duration: float):
    """ Returns decorator that creates a simulated
        function out of given function.
    """
    def decorator(func):
        @functools.wraps(func)  # !crucial for object methods
        def wrapper(*args, **kwargs):
            return duration, func(*args, **kwargs)
        return wrapper
    return decorator


def simulated_events_chain_provider():
    """ Returns decorator that creates an event
        from given simulated function and returns
        that event as first and last events in the
        simulated execution of given sim func.
    """
    def decorator(func):
        @functools.wraps(func)  # !crucial for object methods
        def wrapper(*args, **kwargs):
            event = SimulatedEvent(func, *args, **kwargs)
            return event, event
        return wrapper
    return decorator
