import abc
import numpy as np
import logging
import bisect
from typing import Iterable, Callable
from ..resources.resource_interfaces import ISimulatedResource
from ..core.scheduler import ITasksScheduler
from ..jobs.job import IJob
from ..core.timer import ITimer
from ..core.event import simulated_func
from ..utils.observable import Observable


class INetworkObservable(abc.ABC):
    @abc.abstractmethod
    def subscribe(self, event: any, notify_strategy: Callable):
        pass

    @abc.abstractmethod
    def unsubscribe(self, event: any, notify_strategy: Callable):
        pass


class INetwork(INetworkObservable):
    class Event:
        ON_JOB_LEAVE_NETWORK = 1  # push: (job: IJob, left_net_time: float)

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
    # __observable: IObservable

    def __init__(
        self,
        resources: list[ISimulatedResource],
        probs: Iterable[float],
        psrng: np.random.RandomState,
        timer: ITimer,
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
        self._timer = timer
        self._scheduler: ITasksScheduler = None
        self.__observable = Observable()

    def subscribe(self, event: any, notify_strategy: Callable):
        self.__observable.subscribe(event, notify_strategy)

    def unsubscribe(self, event: any, notify_strategy: Callable):
        self.__observable.unsubscribe(event, notify_strategy)

    def start(self, scheduler: ITasksScheduler):
        """ Starts processing on all resources that have
            a job to process.
        """
        self._scheduler = scheduler
        for idx in range(len(self._resources)):
            self._process_next_job_if_any(idx)

    def _notify(self, event: any, *args, **kwargs):
        self.__observable.notify(event, *args, **kwargs)

    def _construct_transition_probs(self, probs: Iterable[float]):
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
        # will be treated as prob of leaving the network from `src` resource
        if do_log_independent_warning:
            self._logger.warning(
                f'{self._log_prefix} Warning: All leftover probabilities'
              + f'(1 - sum(row_probs)) of each prob matrix row will represent'
              + f' probability of leaving the network.'
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
            return -1  # leaves the network after src
        dst = self._probs[src][idx][1]
        return dst

    @simulated_func(duration=0)
    def _on_resource_processed_job(
        self,
        resource_idx: int,
        job: IJob,
    ):
        next_resource_idx = self._next_resource_idx(resource_idx)
        if next_resource_idx < 0:  # job leaves the network now
            self._on_job_leave_network(job)
        else:  # job leaves to next resource
            next_resource = self._resources[next_resource_idx]
            first_event, last_event = next_resource.is_idle()
            @simulated_func(duration=0)
            def insert_then_process_if_were_idle(is_idle: bool):
                first, last = next_resource.insert_job(job)
                if is_idle and resource_idx != next_resource_idx:
                    last.then_(
                        next_resource.process_cur_job()
                    ).then(
                        self._on_resource_processed_job,
                        next_resource_idx,
                    )
                first.execute(self._scheduler)
            last_event.then(insert_then_process_if_were_idle)
            first_event.execute(self._scheduler)
        # !important process resource_idx after possible insertion
        # of next_resource since possible resource_idx == next_resource_idx
        self._process_next_job_if_any(resource_idx)

    def _process_next_job_if_any(self, resource_idx):
        """ Called when resource is IDLE to process next if any. """
        resource: ISimulatedResource = self._resources[resource_idx]
        first_event, last_event = resource.has_jobs()
        @simulated_func(duration=0)
        def process_if_any(has_jobs: int):
            if has_jobs > 0:
                first, last = resource.process_cur_job()
                last.then(self._on_resource_processed_job, resource_idx)
                first.execute(self._scheduler)
        last_event.then(process_if_any)
        first_event.execute(self._scheduler)

    def _on_job_leave_network(self, job: IJob):
        self._notify(
            INetwork.Event.ON_JOB_LEAVE_NETWORK,
            job,
            self._timer.now(),
        )

    def __str__(self):
        out = '[\n'
        out += f'Time: {self._timer.now()}\n'
        for idx, resource in enumerate(self._resources):
            out += f'Resource #{idx}:\n{resource}\n'
        out += f'\n{self._scheduler}'
        return out + '\n]'
