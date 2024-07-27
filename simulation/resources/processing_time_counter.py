from ..core.timer import ITimer
from ..jobs.job import IJob
from .resource_interfaces import (
    ISimulatedResource,
)
from ..core.event import (
    ISimulatedEvent,
    simulated_func,
    simulated_events_chain,
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

    def insert_job(self, *job: tuple[IJob]) -> tuple[
        ISimulatedEvent[[IJob], any] | ISimulatedEvent[[], any],
        ISimulatedEvent[..., None]
    ]:
        return self._resource.insert_job(*job)

    def is_idle(self) -> tuple[
        ISimulatedEvent[[], any], ISimulatedEvent[..., bool]
    ]:
        return self._resource.is_idle()

    def has_jobs(self) -> tuple[
        ISimulatedEvent[[], any], ISimulatedEvent[..., int]
    ]:
        """ Returns positive number if any jobs are waiting.
        """
        return self._resource.has_jobs()

    def process_cur_job(self) -> tuple[
        ISimulatedEvent[[], None], ISimulatedEvent[[IJob], IJob]
    ]:
        metrics = dict()
        first, last = self._before_process_cur_job(metrics)
        last = last.then_(self._resource.process_cur_job())
        last = last.then(self._after_process_cur_job, metrics)
        return first, last

    def num_jobs(self) -> int:
        return self._resource.num_jobs()

    @simulated_events_chain
    @simulated_func(duration=0)
    def _before_process_cur_job(self, metrics: dict):
        metrics['start_time'] = self._timer.now()

    @simulated_func(duration=0)
    def _after_process_cur_job(self, metrics: dict, job: IJob):
        proc_time = self._timer.now() - metrics['start_time']
        self._metrics.add_processing_jobs_cnt(1)
        self._metrics.add_processing_time_of_jobs(proc_time)
        return job

    def __str__(self):
        return self._resource.__str__()
