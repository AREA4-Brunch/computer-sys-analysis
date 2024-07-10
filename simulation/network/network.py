import abc
import numpy as np
import logging
import bisect
from typing import Iterable
from simulation.resources.resource_interfaces import ISimulatedResource
from simulation.simulation.scheduler import ITasksScheduler
from simulation.jobs.job import IJob
from simulation.utils.timer import ITimer
from simulation.simulation.event import (
    ISimulatedEvent,
    simulated_func,
)


class INetwork(abc.ABC):
    @abc.abstractmethod
    def start(self, scheduler: ITasksScheduler):
        pass


class SimulatedNetwork(INetwork):
    # _resources: list[ISimulatedResource]
    # _probs: list[tuple[float, int]] - probs[src] = [ (prob, dst)... ]
    # _psrng: np.random.RandomState
    # _name: str
    # _logger: logging.Logger
    # _log_prefix: str - prefix prepended to every logger call

    def __init__(
        self,
        resources: list[ISimulatedResource],
        probs: Iterable,
        psrng: np.random.RandomState,
        logger: logging.Logger=None,
        name: str='',
    ) -> None:
        super().__init__()
        self._logger = logger
        self._name = name
        self._log_prefix = f'{self._name}'
        self._resources = resources
        self._probs = self._construct_transition_probs(probs)
        self._psrng = psrng
        self._init_resources_etas()

    def init(self, timer: ITimer) -> 'SimulatedNetwork':
        for resource in self._resources:
            resource.init(timer)
        return self

    def start(self, scheduler: ITasksScheduler):
        """ Starts processing on all resources that are ready.
        """
        @simulated_func(duration=0)
        def process_if_idle(resource, idx, is_idle: bool):
            if is_idle:
                first, last = resource.process_cur_job()
                last.then(self._on_resource_processed_job, idx)
                first.execute(scheduler)

        for idx, resource in enumerate(self._resources):
            start, end = resource.is_idle()
            end.then(process_if_idle, resource, idx)
            start.execute(scheduler)

    def _construct_transition_probs(self, probs: Iterable):
        has_not_warned_about_sys_leave = True
        probs_offseted = []
        for src, dst_probs in enumerate(probs):
            cur_probs = []
            for dst, prob in enumerate(dst_probs):
                if prob <= 0: continue;
                if len(cur_probs):
                    cur_probs.append((prob + cur_probs[-1][0], dst))
                else:
                    cur_probs = [ (prob, dst) ]

            probs_offseted.append(cur_probs)

            # check validity of given probabilities
            has_not_warned_about_sys_leave = self._validate_cur_probs(
                cur_probs,
                src,
                has_not_warned_about_sys_leave,
            )

        return probs_offseted

    def _validate_cur_probs(
        self,
        cur_probs: list[float],
        src: int,
        do_log_independent_warning: bool,
    ) -> bool:
        """ Returns if given probs row is valid and expected.
            If invalid raises an exception, if unexptected logs warnings.
        """
        prob_leaving_sys = 1. - cur_probs[-1][0] if len(cur_probs) else 1.

        if prob_leaving_sys < 0:
            raise Exception(f'Given probabilities in each row must be <= 1!')

        if prob_leaving_sys < 1e-8:  # tolerate rounding error
            # provided probabilities sum up to 1
            return True

        if self._logger is None:  # cannot log warnings
            return False

        # provided probabilities sum up to less than 1 and leftover prob
        # will be treated as prob of leaving the system from `src` resource
        if do_log_independent_warning:
            self._logger.warning(
                f'{self._log_prefix} Warning: All leftover probabilities'
              + f'(1 - sum(row_probs)) of each prob matrix row will represent'
              + f' probability of leaving the system.'
            )

        self._logger.warning(
            f'{self._log_prefix} Warning: prob of leaving system from resource #{src}'
          + f' is {prob_leaving_sys:.5f}'
        )

        return False

    def _next_resource_idx(self, src: int) -> int:
        """ Returns idx of resource in self._resources for a job
            to go to next.
            If job after src leaves the net returns -1.
        """
        prob = self._psrng.uniform(0., 1.)
        idx = bisect.bisect_right(self._probs[src], (prob, 0))
        if idx >= len(self._probs[src]):
            return -1  # leaves the system after src
        dst = self._probs[src][idx][1]
        return dst

    @simulated_func(duration=0)
    def _on_resource_processed_job(
        self,
        resource_idx: int,
        job: IJob,
    ) -> ISimulatedEvent:
        next_resource_idx = self._next_resource_idx(resource_idx)
        self._process_next_job_if_any(resource_idx)
        if next_resource_idx < 0: return;  # job leaves the system now
        next_resource = self._resources[next_resource_idx]
        first_event, last_event = next_resource.is_idle()
        @simulated_func(duration=0)
        def insert_then_process_if_idle(is_idle: bool):
            first, last = next_resource.insert_job(job)
            if not is_idle:
                last.then_(next_resource.process_cur_job())
            first.execute()
        last_event.then(insert_then_process_if_idle)
        first_event.execute()

    def _process_next_job_if_any(self, resource_idx):
        resource: ISimulatedResource = self._resources[resource_idx]
        first_event, last_event = resource.has_jobs_waiting()
        @simulated_func(duration=0)
        def process_if_any(jobs_waiting: int):
            if jobs_waiting > 0:
                resource.process_cur_job()[0].execute()
        last_event.then(process_if_any)
        first_event.execute()
