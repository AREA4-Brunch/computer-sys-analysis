import numpy as np
from typing import Callable
from simulation.utils.timer import ITimer
from simulation.utils.observable import Observable
from simulation.jobs.job import IJob, Job
from simulation.resources.resource_interfaces import (
    ISimulatedResource,
)
from simulation.simulation.event import (
    simulated_events_chain_provider,
    simulated_func,
)


class JobGenerator(ISimulatedResource):
    # _alpha: float - generator speed
    # _timer: ITimer
    # _psrng: np.random.RandomState
    # __observable: IObservable

    def __init__(
        self,
        alpha: float,
        timer: ITimer,
        psrng: np.random.RandomState,
    ):
        super().__init__()
        self._alpha = alpha
        self._timer = timer
        self._psrng = psrng
        self.__observable = Observable()

    def init(self, timer: ITimer) -> 'JobGenerator':
        self._timer = timer
        return self

    @simulated_events_chain_provider()
    @simulated_func(duration=0)
    def insert_job(self, job: IJob):
        """ Generator does not take in any jobs. """
        pass

    @simulated_events_chain_provider()
    @simulated_func(duration=0)
    def is_idle(self) -> bool:
        return False

    @simulated_events_chain_provider()
    @simulated_func(duration=0)
    def has_jobs_waiting(self) -> int:
        return 1

    @simulated_events_chain_provider()
    def process_cur_job(self) -> tuple[float, IJob]:
        proc_time, job = self._gen_new_job()
        return proc_time, job

    def subscribe(self, event: str, notify_strategy: Callable) -> Callable:
        self.__observable.subscribe(event, notify_strategy)

    def unsubscribe(self, event: str, notify_strategy: Callable) -> Callable:
        self.__observable.unsubscribe(event, notify_strategy)

    def _notify(self, event: str, *args, **kwargs):
        self.__observable.notify(event, *args, **kwargs)

    def _num_subscribers(self, event: str) -> int:
        return self.__observable.num_subscribers(event)

    def _gen_new_job(self) -> tuple[float, IJob]:
        generation_duration = self._psrng.exponential(1 / self._alpha)
        job = Job(self._timer.now())
        return generation_duration, job
