import abc
from ..jobs.job import IJob
from ..core.event import ISimulatedEvent


class IResource(abc.ABC):
    @abc.abstractmethod
    def insert_job(self, job: IJob) -> None:
        pass

    @abc.abstractmethod
    def is_idle(self) -> bool:
        pass

    @abc.abstractmethod
    def has_jobs(self) -> int:
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
    def insert_job(self, *job: tuple[IJob]) -> tuple[
        ISimulatedEvent[[IJob], any] | ISimulatedEvent[[], any],
        ISimulatedEvent[..., None]
    ]:
        """ Arg `job` is optionally passed for in case
            it is instead provided via `ISimulatedEvent.execute`
            directly or through event chain.
        """
        pass

    @abc.abstractmethod
    def is_idle(self) -> tuple[
        ISimulatedEvent[[], any], ISimulatedEvent[..., bool]
    ]:
        pass

    @abc.abstractmethod
    def has_jobs(self) -> tuple[
        ISimulatedEvent[[], any], ISimulatedEvent[..., int]
    ]:
        """ Returns positive number if any jobs are waiting.
        """
        pass

    @abc.abstractmethod
    def process_cur_job(self) -> tuple[
        ISimulatedEvent[[], any], ISimulatedEvent[..., IJob]
    ]:
        pass

    @abc.abstractmethod
    def num_jobs(self) -> int:
        """ Returns total number of jobs waiting or being processed.
            Not a step in the simulation; used for describing the state
            during the simulation.
        """
        pass
