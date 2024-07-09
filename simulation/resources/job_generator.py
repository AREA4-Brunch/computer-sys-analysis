import numpy as np
from simulation.utils.timer import ITimer
from simulation.utils.observable import Observable
from simulation.jobs.job import IJob
from simulation.resources.resource_interfaces import (
    ISimulatedResource,
)
from simulation.simulation.event import (
    ISimulatedEvent,
    simulated_events_chain,
    simulated_func,
)


class JobGenerator(ISimulatedResource):
    def __init__(
        self,
        alpha: float,
        timer: ITimer,
        psrng: np.random.RandomState,
    ):
        super().__init__()
        self._alpha = alpha
        self._psrng = psrng
        self._timer = timer
        self.insert_job(self._gen_new_job())

    def init(self, timer: ITimer) -> 'JobGenerator':
        self._timer = timer
        return self

    @simulated_events_chain()
    @simulated_func(duration=0)
    def insert_job(self, job: IJob):
        pass

    def process_cur_job(self) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        cur_job = self._cur_job
        self._on_processed_job(self._cur_job_arrival_time)
        return cur_job

    def _gen_eta(self):
        generation_duration = self._psrng.exponential(1 / self._alpha)
        eta = self._timer._now() + generation_duration
        return eta

    def _gen_new_job(self) -> IJob:
        return Job(self._timer._now())

    def _process_cur_job(self, dummy) -> tuple[float, tuple[IJob, float]]:
        job, start_processing_at = self._jobs.popleft()
        return self._proc_time, (job, start_processing_at)
