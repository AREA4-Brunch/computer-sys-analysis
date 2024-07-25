from collections import deque
from ..jobs.job import IJob
from .resource_interfaces import ISimulatedResource
from ..core.event import (
    simulated_func,
    simulated_events_chain_provider,
    ISimulatedEvent,
    SimulatedEvent,
)


class StandardResource(ISimulatedResource):
    """ Subject in decorator design pattern. """
    # _jobs: deque[IJob]
    # _proc_time: float - avg processing time of a job

    def __init__(self, serv_time: float):
        super().__init__()
        self._jobs = deque()
        self._proc_time = serv_time
        self._is_idle = True

    @simulated_events_chain_provider()
    @simulated_func(duration=0)
    def insert_job(self, job: IJob) -> None:
        self._jobs.append(job)

    @simulated_events_chain_provider()
    @simulated_func(duration=0)
    def is_idle(self) -> bool:
        return self._is_idle

    @simulated_events_chain_provider()
    @simulated_func(duration=0)
    def has_jobs(self) -> int:
        return len(self._jobs)

    def process_cur_job(self) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        first = SimulatedEvent(self._on_start_process_cur_job)
        last = first.then(self._on_end_process_cur_job)
        return first, last

    def num_jobs(self) -> int:
        return len(self._jobs)

    def _on_start_process_cur_job(self) -> tuple[float, None]:
        self._is_idle = False
        return self._proc_time, None

    @simulated_func(duration=0)
    def _on_end_process_cur_job(self) -> IJob:
        job = self._jobs.popleft()
        self._is_idle = True
        return job

    def __str__(self):
        out = '[\n'
        out += f'\tis_idle: {self._is_idle}\n'
        out += f'\tjobs: '
        for job in self._jobs: out += f'{job.created_at()}, ';
        return out + '\n]'
