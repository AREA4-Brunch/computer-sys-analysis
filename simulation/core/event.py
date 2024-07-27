import abc
import functools
from typing import Callable, ParamSpec, TypeVar, Generic, Union
from .scheduler import ITasksScheduler
from ..utils.callable import num_remaining_args


# types of args, kwargs passed to `ISimulatedEvent.execute`
TExecParams = ParamSpec('TExecParams')
TRet = TypeVar('TRet')  # type of return value of the SimFunc
TAllParams = ParamSpec('TAllParams')  # types of args, kwargs of SimFunc.__call__
type _TDuration = float   # type of duration of execution of sim func
type SimFunc = Callable[TAllParams, tuple[_TDuration, TRet]]
# types of args, kwargs passed to the SimFunc, but provided before TExecParams
TPartialParams = ParamSpec('TPartialParams')


class ISimulatedEvent(Generic[TExecParams, TRet], abc.ABC):
    @abc.abstractmethod
    def execute(
        self,
        scheduler: ITasksScheduler,
        *args: TExecParams.args,
        **kwargs: TExecParams.kwargs,
    ) -> None:
        pass

    @abc.abstractmethod
    def then(self, *args, **kwargs) -> Union[
        'ISimulatedEvent[[], any]',
        'ISimulatedEvent[[TRet], any]'
    ]:
        """ Should allow for taking in an existing ISimulatedEvent
            or just passing args and creating one within the method.
        """
        pass

    def then_(
        self,
        first_last: tuple[any, 'ISimulatedEvent'],
        *args,
        **kwargs
    ) -> 'ISimulatedEvent':
        """ Performs `then` on self using `first` and given args
            and returns `last`.
            Useful to make some code cleaner.
        """
        self.then(first_last[0], *args, **kwargs)
        return first_last[1]

    @abc.abstractmethod
    def num_expected_args(self) -> int:
        """ Number of arguments to be provided on `execute`. """
        pass


class SimulatedEvent(
    Generic[TExecParams, TRet],
    ISimulatedEvent[TExecParams, TRet],
):
    """ Executes given func and allows other simulated events
        to be executed right after it with the result of original
        func passed to them.

        Represent node in linked list of simulated events.
    """
    # _func: Callable[[any], tuple[float, any]]
    # _nexts: list[ISimulatedEvent]

    def __init__(
        self,
        func: SimFunc,
        *args,
        **kwargs
    ) -> None:
        """ Given func will be executed with given arguments
            and optionally other ones appended to them if provided
            on `execute`, and should return the tuple with duration
            of its execution and its return value in that order.
        """
        super().__init__()
        # create memory efficient wrappers to store args and kwargs
        # to avoid adding attributes to this class if none were provided
        args_missing = num_remaining_args(func) - len(args) - len(kwargs)
        self._args_kwargs = (args_missing,)
        if len(args): self._args_kwargs += (args, )
        if len(kwargs): self._args_kwargs += (kwargs, )
        self._func = func
        self._nexts = []

    def execute(
        self,
        scheduler: ITasksScheduler,
        *prev_args: TExecParams.args,
        **prev_kwargs: TExecParams.kwargs,
    ) -> None:
        """ Executes the func provided in constructor with given args
            and kwargs appended to the ones provided in the
            constructor. Then it executes all other events chained
            via `then` at same scheduled time and to each of them
            appends the func's result to their args provided in their
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
            does_take_args = next.num_expected_args() > 0
            if duration == 0:  # no need to overload scheduler
                if does_take_args: next.execute(scheduler, ret_val);
                else:              next.execute(scheduler);
            elif does_take_args:
                scheduler.add(duration, next.execute, scheduler, ret_val)
            else:
                scheduler.add(duration, next.execute, scheduler)

    def then(self, *args, **kwargs) -> (
        ISimulatedEvent[[], any]
      | ISimulatedEvent[[TRet], any]
    ):
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
        next_args_missing = next.num_expected_args()
        if next_args_missing > 1:
            raise ValueError(
                f'Given function is yet to be provided {next_args_missing} args'
              + f', but at most 1 can be returned from previous `SimulatedEvent`.'
            )
        self._nexts.append(next)
        return next

    def num_expected_args(self) -> int:
        return self._args_kwargs[0]

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
    def decorator(func) -> SimFunc:
        @functools.wraps(func)  # !crucial for object methods
        def wrapper(*args: TAllParams.args, **kwargs: TAllParams.kwargs):
            return duration, func(*args, **kwargs)
        return wrapper
    return decorator


def simulated_events_chain(sim_func: SimFunc) -> tuple[
    SimulatedEvent[TExecParams, TRet],
    SimulatedEvent[TExecParams, TRet]
]:
    """ Creates an event from given simulated function
        and returns that event as first and last events
        in the simulated execution chain of given sim func.
    """
    @functools.wraps(sim_func)  # !crucial for object methods
    def wrapper(
        *args: TPartialParams.args,
        **kwargs: TPartialParams.kwargs
    ):
        event = SimulatedEvent(sim_func, *args, **kwargs)
        return event, event
    return wrapper
