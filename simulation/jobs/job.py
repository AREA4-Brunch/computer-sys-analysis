import abc


class IJob(abc.ABC):
    @abc.abstractmethod
    def created_at(self) -> float:
        pass


class Job(IJob):
    # creation_time - time job has arrived in the net

    def __init__(self, created_at: float):
        self.creation_time = created_at

    def created_at(self) -> float:
        return self.creation_time
