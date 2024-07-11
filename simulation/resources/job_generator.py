import numpy as np
from simulation.utils.timer import ITimer
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

    def init(self, timer: ITimer) -> 'JobGenerator':
        self._timer = timer
        return self

    @simulated_events_chain_provider()
    @simulated_func(duration=0)
    def insert_job(self, job: IJob):
        """ Generator does not take in any jobs. """
        return

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

    def _gen_new_job(self) -> tuple[float, IJob]:
        generation_duration = self._psrng.exponential(1 / self._alpha)
        job = Job(self._timer.now())
        return generation_duration, job
