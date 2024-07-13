from ..utils.timer import ITimer
from ..jobs.job import IJob
from .resource_interfaces import (
    ISimulatedResource,
)
from ..simulation.event import (
    ISimulatedEvent,
    simulated_func,
    simulated_events_chain_provider,
)
from ..metrics.resource_metrics import ISimulatedResourceMetrics


class ProcessingTimeCounter(ISimulatedResource):
    """ Decorater in decorater design pattern. """
    # _resource: ISimulatedResource
    # _timer: ITimer
    # _metrics: ISimulatedResourceMetrics

    def __init__(
        self,
        subject_resource: ISimulatedResource,
        timer: ITimer,
        metrics: ISimulatedResourceMetrics,
    ) -> None:
        super().__init__()
        self._resource = subject_resource
        self._timer = timer
        self._metrics = metrics

    def insert_job(
        self,
        *args,
        **kwargs
    ) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        return self._resource.insert_job(*args, **kwargs)

    def is_idle(self) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        return self._resource.is_idle()

    def has_jobs_waiting(self) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        """ Returns positive number if any jobs are waiting.
        """
        return self._resource.has_jobs_waiting()

    def process_cur_job(self) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        metrics = { 'start_time': 0 }
        first, last = self._before_process_cur_job(metrics)
        last = last.then_(self._resource.process_cur_job())
        last = last.then(self._after_process_cur_job, metrics)
        return first, last

    def num_jobs(self) -> int:
        return self._resource.num_jobs()

    @simulated_events_chain_provider()
    @simulated_func(duration=0)
    def _before_process_cur_job(self, metrics: dict, job: IJob):
        metrics['start_time'] = self._timer.now()
        return job

    @simulated_func(duration=0)
    def _after_process_cur_job(self, metrics: dict, job: IJob):
        proc_time = self._timer.now() - metrics['start_time']
        self._metrics.add_processing_jobs_cnt(1)
        self._metrics.add_processing_time_of_jobs(proc_time)
        return job
