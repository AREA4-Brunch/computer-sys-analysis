import abc
from ..jobs.job import IJob
from ..simulation.event import ISimulatedEvent


class IResource(abc.ABC):
    @abc.abstractmethod
    def insert_job(self, job: IJob) -> None:
        pass

    @abc.abstractmethod
    def is_idle(self) -> bool:
        pass

    @abc.abstractmethod
    def has_jobs_waiting(self) -> int:
        """ Returns positive number if any jobs are waiting.
        """
        pass

    @abc.abstractmethod
    def process_cur_job(self) -> IJob:
        pass

    @abc.abstractmethod
    def num_jobs(self) -> int:
        """ Returns total number of jobs waiting or being processed.
        """
        pass


class ISimulatedResource(abc.ABC):
    @abc.abstractmethod
    def insert_job(
        self,
        *args,
        **kwargs
    ) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        pass

    @abc.abstractmethod
    def is_idle(self) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        pass

    @abc.abstractmethod
    def has_jobs_waiting(self) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        """ Returns positive number if any jobs are waiting.
        """
        pass

    @abc.abstractmethod
    def process_cur_job(self) -> tuple[ISimulatedEvent, ISimulatedEvent]:
        pass

    @abc.abstractmethod
    def num_jobs(self) -> int:
        """ Returns total number of jobs waiting or being processed.
            Not a step in the simulation; used for describing the state
            during the simulation.
        """
        pass
