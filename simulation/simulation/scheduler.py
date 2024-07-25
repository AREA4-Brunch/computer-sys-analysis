import heapq
from ..core.scheduler import ITasksScheduler
from ..core.timer import ITimer


class TasksScheduler(ITasksScheduler):
    # _timer: ITimer
    # _tasks: SortedDict - _tasks[time] = deque[(func, args, kwargs)]

    def __init__(self, timer: ITimer) -> None:
        """ Because of singleton only first constructor call
            will execute, others will be ignored and will not
            require any args to be passed.
        """
        self._timer = timer
        self._tasks = []
        self._next_task_id = 1

    def __bool__(self):
        return bool(self._tasks)

    def add(self, schedule_in: float, func: callable, *args, **kwargs):
        # store args and kwargs in tuple only if they were provided
        schedule_at = self._timer.now() + schedule_in
        task_desc = (func,)
        if len(args): task_desc += (args,)
        if len(kwargs): task_desc += (kwargs,)
        heapq.heappush(
            self._tasks, (schedule_at, self._next_task_id, task_desc)
        )
        self._next_task_id += 1

    def next(self) -> tuple[callable, tuple, dict]:
        schedule_at, task_id, task_desc = heapq.heappop(self._tasks)
        if not self._tasks: self._next_task_id = 0;
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

    def __str__(self) -> str:
        out = f'Scheduler[\n\ttime: {self._timer.now()}\n'
        tasks = [ el for el in self._tasks ]
        while tasks:
            out += f'\t{heapq.heappop(tasks)}\n\n'
        return out + '\n]'
