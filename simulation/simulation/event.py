import abc
import functools
from typing import Callable
from simulation.scheduler import ITasksScheduler
from simulation.utils.callable_wrappers import CallableWithArgs


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
        first: 'ISimulatedEvent',
        last: 'ISimulatedEvent',
        *args,
        **kwargs
    ) -> 'ISimulatedEvent':
        """ Performs `then` on self using `first` and given args
            and returns `last`.
            Useful to make some code cleaner.
        """
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
        if args or kwargs:
            func = CallableWithArgs(func, args, kwargs)

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
        duration, ret_val = self._func(*prev_args, **prev_kwargs)
        for next in self._nexts:
            if duration == 0:  # no need to overload scheduler
                next.execute(scheduler, ret_val)
            else:
                scheduler.add(duration, next.execute, scheduler, ret_val)

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
        first: ISimulatedEvent,
        last: ISimulatedEvent,
        *args,
        **kwargs
    ) -> ISimulatedEvent:
        self.then(first, *args, **kwargs)
        return last


def simulated_func(duration: float):
    """ Returns decorator that creates a simulated
        function out of given function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return duration, func(*args, **kwargs)
        return wrapper
    return decorator


def simulated_events_chain(no_args: bool=False):
    """ Returns decorator that creates an event
        from given simulated function and returns
        that event as first and last events in the
        simulated execution of given sim func.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if no_args:
                event = SimulatedEvent(
                    lambda *_, **__: func(*args, **kwargs)
                )
            else:
                event = SimulatedEvent(func, *args, **kwargs)
            return event, event
        return wrapper
    return decorator
