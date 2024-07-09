import abc


class IJob(abc.ABC):
    pass


class Job(IJob):
    # created_at - time job has arrived in the net

    def __init__(self, created_at: float):
        self.created_at = created_at
