from ..utils.timer import ITimer
from .resource_interfaces import (
    ISimulatedResource,
)
from ..simulation.event import (
    ISimulatedEvent,
    simulated_func,
    simulated_events_chain_provider,
)
from ..metrics.resource_metrics import ISimulatedResourceMetrics


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

    def is_idle(self) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        return self._process_sim_func(self._resource.is_idle)

    def has_jobs_waiting(self) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        return self._process_sim_func(self._resource.has_jobs_waiting)

    def process_cur_job(self) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        return self._process_sim_func(self._resource.process_cur_job)

    def num_jobs(self) -> int:
        return self._resource.num_jobs()

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
    def _before_resource_sim_func(self, metrics: dict, prev_ret_val=None):
        """ prev_ret_val=None because there may have been no events before
            and no args were provided in `ISimulatedEvent.execute`
        """
        metrics['start_time'] = self._timer.now()
        metrics['num_jobs'] = self._resource.num_jobs()
        return prev_ret_val

    @simulated_func(duration=0)
    def _after_resource_sim_func(self, metrics: dict, prev_ret_val):
        proc_time = self._timer.now() - metrics['start_time']
        self._metrics.add_jobs_cnt_during_time(
            proc_time * metrics['num_jobs']
        )
        return prev_ret_val
