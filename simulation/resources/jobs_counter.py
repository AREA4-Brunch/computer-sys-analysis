from ..core.timer import ITimer
from .resource_interfaces import (
    ISimulatedResource,
)
from ..core.event import (
    ISimulatedEvent,
    simulated_func,
    simulated_events_chain,
)
from ..metrics.resource_metrics import ISimulatedResourceMetrics
from ..jobs.job import IJob


class JobsCounter(ISimulatedResource):
    """ Decorater in decorater design pattern. """
    # _resource: ISimulatedResource
    # _timer: ITimer
    # _metrics: ISimulatedResourceMetrics
    # _last_measurement_time: float
    # _last_measurement_num_jobs: int

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
        self._last_measurement_time = self._timer.now()
        self._last_measurement_num_jobs = 0

    def insert_job(self, *job: tuple[IJob]) -> tuple[
        ISimulatedEvent[[IJob], any] | ISimulatedEvent[[], any],
        ISimulatedEvent[[None], None]
    ]:
        return self._process_sim_func(
            self._resource.insert_job, *job
        )

    def is_idle(self) -> tuple[
        ISimulatedEvent[[], any], ISimulatedEvent[[bool], bool]
    ]:
        return self._process_sim_func(self._resource.is_idle)

    def has_jobs(self) -> tuple[
        ISimulatedEvent[[], any], ISimulatedEvent[[int], int]
    ]:
        return self._process_sim_func(self._resource.has_jobs)

    def process_cur_job(self) -> tuple[
        ISimulatedEvent[[], any], ISimulatedEvent[[IJob], IJob]
    ]:
        return self._process_sim_func(self._resource.process_cur_job)

    def num_jobs(self) -> int:
        return self._resource.num_jobs()

    def _process_sim_func(
        self,
        sim_events_chain_provider,
        *args,
        **kwargs
    ) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        first, last = self._before_resource_sim_func()
        last = last.then_(
            sim_events_chain_provider(*args, **kwargs)
        ).then(
            self._after_resource_sim_func
        )
        return first, last

    @simulated_events_chain
    @simulated_func(duration=0)
    def _before_resource_sim_func(self, prev_ret_val=None):
        """ prev_ret_val=None because there may have been no events before
            and no args were provided in `ISimulatedEvent.execute`
        """
        elapsed_time = self._timer.now() - self._last_measurement_time
        jobs_time = elapsed_time * self._last_measurement_num_jobs
        self._metrics.add_jobs_cnt_during_time(jobs_time)
        self._last_measurement_time = self._timer.now()
        self._last_measurement_num_jobs = self._resource.num_jobs()
        return prev_ret_val

    @simulated_func(duration=0)
    def _after_resource_sim_func(self, prev_ret_val):
        elapsed_time = self._timer.now() - self._last_measurement_time
        jobs_time = elapsed_time * self._last_measurement_num_jobs
        self._metrics.add_jobs_cnt_during_time(jobs_time)
        self._last_measurement_time = self._timer.now()
        self._last_measurement_num_jobs = self._resource.num_jobs()
        return prev_ret_val

    def __str__(self):
        return self._resource.__str__()
