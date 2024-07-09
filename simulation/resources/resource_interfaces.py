import abc
from typing import Callable
from simulation.jobs.job import IJob
from simulation.utils.observable import IObservable
from simulation.simulation.event import ISimulatedEvent
from simulation.utils.timer import ITimer


class IResourceObservable(abc.ABC):
    @abc.abstractmethod
    def subscribe(self, event: any, notify_strategy: Callable) -> Callable:
        pass

    @abc.abstractmethod
    def unsubscribe(self, event: any, notify_strategy: Callable) -> Callable:
        pass

    @abc.abstractmethod
    def _notify(self, event: any, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _num_subscribers(self, event: any) -> int:
        pass


class IResource(IResourceObservable):
    @abc.abstractmethod
    def init(self, timer: ITimer) -> 'IResource':
        pass

    @abc.abstractmethod
    def insert_job(self, job: IJob):
        pass

    @abc.abstractmethod
    def has_cur_job(self) -> bool:
        pass

    @abc.abstractmethod
    def has_jobs_waiting(self) -> int:
        """ Returns positive number if any jobs are waiting.
        """
        pass

    @abc.abstractmethod
    def process_cur_job(self) -> IJob:
        pass


class ISimulatedResource(IResourceObservable):
    @abc.abstractmethod
    def init(self, timer: ITimer) -> 'ISimulatedResource':
        pass

    @abc.abstractmethod
    def insert_job(
        self,
        job: IJob,
    ) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        pass

    @abc.abstractmethod
    def process_cur_job(self) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        pass

    @abc.abstractmethod
    def is_idle(self) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        pass

    @abc.abstractmethod
    def has_jobs_waiting(self) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        """ Returns positive number if any jobs are waiting.
        """
        pass
