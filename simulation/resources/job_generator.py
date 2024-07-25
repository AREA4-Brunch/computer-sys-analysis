import numpy as np
from ..core.timer import ITimer
from ..jobs.job import IJob, Job
from .resource_interfaces import (
    ISimulatedResource,
)
from ..core.event import (
    simulated_func,
    simulated_events_chain_provider,
    ISimulatedEvent,
    SimulatedEvent,
)


class JobGenerator(ISimulatedResource):
    # _alpha: float - generator speed
    # _timer: ITimer
    # _psrng: np.random.RandomState

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
        self._is_idle = True

    @simulated_events_chain_provider()
    @simulated_func(duration=0)
    def insert_job(self, job: IJob):
        """ Generator does not take in any jobs. """
        return

    @simulated_events_chain_provider()
    @simulated_func(duration=0)
    def is_idle(self) -> bool:
        return self._is_idle

    @simulated_events_chain_provider()
    @simulated_func(duration=0)
    def has_jobs(self) -> int:
        return 1

    def process_cur_job(self) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        first = SimulatedEvent(self._on_start_process_cur_job)
        last = first.then(self._on_end_process_cur_job)
        return first, last

    def num_jobs(self) -> int:
        return 1

    def _on_start_process_cur_job(self) -> tuple[float, None]:
        self._is_idle = False
        gen_time = self._psrng.exponential(1 / self._alpha)
        return gen_time, None

    @simulated_func(duration=0)
    def _on_end_process_cur_job(self) -> IJob:
        job = Job(self._timer.now())
        self._is_idle = True
        return job

    def __str__(self):
        out = '[\n'
        out += f'\tis_idle: {self._is_idle}'
        return out + '\n]'
