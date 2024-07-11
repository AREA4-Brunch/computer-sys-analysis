import abc


class IJob(abc.ABC):
    @abc.abstractmethod
    def created_at(self) -> float:
        pass


class Job(IJob):
    # created_at - time job has arrived in the net

    def __init__(self, created_at: float):
        self.created_at = created_at

    def created_at(self) -> float:
        return self.created_at
