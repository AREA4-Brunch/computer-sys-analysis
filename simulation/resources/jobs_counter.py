from simulation.utils.timer import ITimer
from simulation.jobs.job import IJob
from simulation.resources.resource_interfaces import (
    ISimulatedResource,
)
from simulation.simulation.event import (
    ISimulatedEvent,
    simulated_func,
    simulated_events_chain_provider,
)
from simulation.metrics.resource_metrics import ISimulatedResourceMetrics


class JobsCounter(ISimulatedResource):
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
        return self._process_sim_func(
            self._resource.insert_job, *args, **kwargs
        )

    def process_cur_job(self) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        return self._process_sim_func(self._resource.process_cur_job)

    def is_idle(self) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        return self._process_sim_func(self._resource.is_idle)

    def has_jobs_waiting(self) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        return self._process_sim_func(self._resource.has_jobs_waiting)

    def _process_sim_func(
        self,
        sim_events_chain_provider,
        *args,
        **kwargs
    ) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        metrics = { 'start_time': 0 }
        first, last = self._before_resource_sim_func(metrics)
        last = last.then_(
            sim_events_chain_provider(*args, **kwargs)
        ).then(
            self._after_resource_sim_func, metrics
        )
        return first, last

    @simulated_events_chain_provider()
    @simulated_func(duration=0)
    def _before_resource_sim_func(self, metrics: dict, job: IJob):
        metrics['start_time'] = self._timer.now()
        metrics['num_jobs'] = self._resource.num_jobs()
        return job

    @simulated_func(duration=0)
    def _after_resource_sim_func(self, metrics: dict, job: IJob):
        proc_time = self._timer.now() - metrics['start_time']
        self._metrics.add_jobs_cnt_during_time(
            proc_time * metrics['num_jobs']
        )
        return job
