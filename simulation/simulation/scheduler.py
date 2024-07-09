import abc
import heapq
from simulation.utils.timer import ITimer


class ITasksScheduler(abc.ABC):
    """ Provides the interface for registering into the
        scheduler but not for manipulating it.
    """
    @abc.abstractmethod
    def add(self, schedule_in: float, func: callable, *args, **kwargs):
        pass


class TasksScheduler(ITasksScheduler):
    # _timer: ITimer
    # _tasks: heap of tuples: (scheduled_at, (func, args if any, kwargs if any))

    def __init__(self, timer: ITimer) -> None:
        """ Because of singleton only first constructor call
            will execute, others will be ignored and will not
            require any args to be passed.
        """
        self._timer = timer
        self._tasks = heapq.heapify([])

    def __bool__(self):
        return bool(self._tasks)

    def add(self, schedule_in: float, func: callable, *args, **kwargs):
        # store args and kwargs in tuple only if they were provided
        schedule_at = self._timer._now + schedule_in
        task_desc = (func)
        if (len(args)): task_desc += (args,)
        if len(kwargs): task_desc += (kwargs,)
        heapq.heappush(self._tasks, (schedule_at, task_desc))

    def next(self) -> tuple[callable, tuple, dict]:
        schedule_at, task_desc = heapq.heappop(self._tasks)
        # unpack task_desc and where missing add empty args, kwargs
        return self._unpack_task_desc(task_desc)

    def next_scheduled_at(self) -> float:
        schedule_at = self._tasks[0][0]
        return schedule_at

    def has_next(self) -> bool:
        return bool(self._tasks)

    def _unpack_task_desc(self, task_desc) -> tuple[callable, tuple, dict]:
        if len(task_desc) == 3: return task_desc;
        func = task_desc[0]
        if len(task_desc) == 1: return (func, tuple(), dict());
        has_kwargs = isinstance(task_desc[1], dict)
        args = tuple() if has_kwargs else task_desc[1]
        kwargs = task_desc[1] if has_kwargs else dict()
        return (func, args, kwargs)
