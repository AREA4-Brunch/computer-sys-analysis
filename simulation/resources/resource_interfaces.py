import abc
from simulation.jobs.job import IJob
from simulation.simulation.event import ISimulatedEvent


class IResource(abc.ABC):
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
    def num_jobs(self) -> int:
        """ Returns total number of jobs waiting or being processed.
        """
        pass

    @abc.abstractmethod
    def process_cur_job(self) -> IJob:
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

    @abc.abstractmethod
    def num_jobs(self) -> int:
        """ Returns total number of jobs waiting or being processed.
            Not a step in the simulation; used for describing the state
            during the simulation.
        """
        pass
