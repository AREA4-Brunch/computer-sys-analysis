from collections import deque
from typing import Callable
from simulation.utils.timer import ITimer
from simulation.utils.observable import Observable
from simulation.jobs.job import IJob
from simulation.resources.resource_interfaces import (
    ISimulatedResource,
)
from simulation.simulation.event import (
    simulated_func,
    simulated_events_chain,
)


class StandardResource(ISimulatedResource):
    """ Subject in decorator design pattern. """
    # _jobs: deque[IJob]
    # _proc_time: float - avg processing time of a job
    # _timer: ITimer
    # __observable: IObservable

    def __init__(self, serv_time: float):
        super().__init__()
        self._jobs = deque()
        self._proc_time = serv_time
        self._timer = None
        self.__observable = Observable()

    def init(self, timer: ITimer) -> 'StandardResource':
        self._timer = timer
        return self

    @simulated_events_chain()
    @simulated_func(duration=0)
    def insert_job(self, job: IJob) -> None:
        self._jobs.append(job)

    @simulated_events_chain()
    @simulated_func(duration=0)
    def is_idle(self) -> bool:
        return bool(self._jobs)

    @simulated_events_chain()
    @simulated_func(duration=0)
    def has_jobs_waiting(self) -> int:
        return len(self._jobs) - 1

    @simulated_events_chain()
    def process_cur_job(self) -> tuple[float, IJob]:
        job = self._jobs.popleft()
        return self._proc_time, job

    def subscribe(self, event: str, notify_strategy: Callable) -> Callable:
        self.__observable.subscribe(event, notify_strategy)

    def unsubscribe(self, event: str, notify_strategy: Callable) -> Callable:
        self.__observable.unsubscribe(event, notify_strategy)

    def _notify(self, event: str, *args, **kwargs):
        self.__observable.notify(event, *args, **kwargs)

    def _num_subscribers(self, event: str) -> int:
        return self.__observable.num_subscribers(event)
